"""Gibbs sampling functionality."""

from typing import Optional
from jaxtyping import PRNGKeyArray
import jax
from jax import numpy as jnp
import equinox as eqx
from ..models import ebm_models, factor_graphs
from ..custom_types import VariableStates, State, VariableMapping
from energy_jax.sampling import sampler


class GibbsSampler(eqx.Module, strict=True):
    r"""
    Gibbs sample over a factor graph.

    Iteratively computes:

    $$ x_j^{(i + 1)} \sim p(x_j^{(i + 1)} | x_0^{(i + 1)} ... x_{j - 1}^{(i + 1)},
        x_j^{(i)} ... x_n^{(i)}) $$

    for each variable.

    Note, due to the nature of its interactions with factor graphs

    Attributes:
        - num_chains: the number of chains to use with sample_chains
        - reps: the number of repetitions to do over the variables
        - factor_graph: the factor graph the gibbs sampling is done over
        - is_evidence: whether each state is a free variable or a clamped
            variable
        - traversal order: the order in which to go over the variables
    """

    samplers: dict[factor_graphs.VariableNode, sampler.AbstractSampler]
    num_chains: Optional[int] = None
    reps: Optional[int] = None
    is_evidence: Optional[VariableMapping] = None
    traversal_order: Optional[list[factor_graphs.VariableNode]] = None
    squeeze: Optional[bool] = False

    def sample_single_variable(
        self,
        factor_graph: factor_graphs.FactorGraph,
        variable: factor_graphs.VariableNode,
        values: VariableStates,
        key: PRNGKeyArray,
    ) -> State:
        r"""
        Sample a single variable.

        Computes the subdistribution based on the values incidence to the variable.

        $$ x_j^{(i + 1)} \sim p(x_j^{(i + 1)} | x_0^{(i + 1)} ... x_{j - 1}^{(i + 1)},
        x_j^{(i)} ... x_n^{(i)}) $$

        Args:
            - variable: the variable to sample
            - values: the current values of the variables
            - key: the random key to use

        Returns:
            - the sample for that variable
        """
        factors = factor_graph.get_incident_factors(variable)
        ebm = ebm_models.FactorEBM(
            factors,
            values,
            is_clamped={var: var != variable for var in factor_graph.variables},
        )
        samples = self.samplers[variable].run_chain(
            ebm,
            values[variable],
            key=key,
        )
        # TODO: rethink energy-jax and these positions? Maybe an abstract state?
        if self.squeeze:
            return jnp.squeeze(samples["position"], axis=0)
        return samples["position"]

    def step(
        self,
        factor_graph: factor_graphs.FactorGraph,
        current_value: VariableStates,
        list_variables: list[factor_graphs.VariableNode],
        key: PRNGKeyArray,
    ) -> VariableStates:
        """
        Compute one Gibbs iteration (over all variables).

        This loops over each variable and clamps every other variable and iterates
        this process in the standard gibbs manner.

        Args:
            - current_value: the input values
            - list_variables: the list of variables to sample
            - key: the random key to use

        Returns:
            - the updated state
        """
        for var in list_variables:
            key, subkey = jax.random.split(key)
            new_state = self.sample_single_variable(
                factor_graph, var, current_value, subkey
            )
            current_value[var] = new_state

        return current_value

    def run_chain(
        self,
        factor_graph: factor_graphs.FactorGraph,
        current_values: VariableStates,
        key: PRNGKeyArray,
        is_evidence: Optional[
            VariableStates
        ] = None,  # TODO: weird to have this input and a member variable??
    ) -> VariableStates:
        """
        Perform Gibbs sampling for a given number of repetitions.

        This takes self.reps steps.

        Args:
            - factor_graph: the factor graph to use
            - current_values: the input values
            - key: the random key to use
            - is_evidence: optionally provide new evidence values to use

        Returns:
            - the updated state
        """
        if self.reps is None:
            raise ValueError(
                "You cannot call run_chain without specifying the number of repetitions!"
            )
        arr, const = eqx.partition(current_values, eqx.is_array)

        if is_evidence is None:
            if self.is_evidence is None:
                evidence = {i: False for i in factor_graph.variables}
            else:
                evidence = self.is_evidence
        else:
            evidence = is_evidence

        if self.traversal_order is not None:
            _vars = [var for var in self.traversal_order if not evidence[var]]
        else:
            _vars = sorted(
                k for k in factor_graph.variables if not evidence[k]
            )  # This enforces the same order since dict
            # is by default unstructed. I am not sure how much this matters for Gibbs sampling
            # but this hidden randomness can make it very hard to debug

        def inner(carry, subkey) -> tuple:  # type: ignore[no-untyped-def]
            """Inner function for scan."""
            state = self.step(factor_graph, eqx.combine(carry, const), _vars, subkey)
            state, _ = eqx.partition(state, eqx.is_array)
            return state, _

        out = jax.lax.scan(inner, arr, jax.random.split(key, self.reps))[0]
        return eqx.combine(out, const)

    def sample_chains(
        self,
        factor_graph: factor_graphs.FactorGraph,
        state: VariableStates,
        key: PRNGKeyArray,
        is_evidence: Optional[VariableStates] = None,
    ) -> VariableStates:
        """
        Vectorize over keys to get multiple gibbs chains.

        Note: if you want to randomly initialize the variables (rather than providing an initial
        state for each) you can set each value to None (e.g. a dictionary mapping each Variable to
        None).

        Args:
            - factor_graph: the factor graph to use
            - state: the initial variable values
            - key: the random key to use
            - is_evidence: optionally provide new evidence values to use

        Returns:
            - the dictionary
        """
        if self.num_chains is None:
            raise ValueError(
                "You cannot call sample_chains without specifying the number of chains!"
            )

        keys = jax.random.split(key, self.num_chains)

        return eqx.filter_vmap(self.run_chain, in_axes=(None, None, 0, None))(
            factor_graph, state, keys, is_evidence
        )
