"""EBM Joint Factor Models."""

from typing import Any
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree
from ..models.factor_graphs import FactorNode
from ..custom_types import VariableStates, VariableMapping, State
from energy_jax.ebms import ebm


class FactorEBM(ebm.AbstractEBM):
    """
    Joined Factor EBM.

    Attributes:
        - factors: the list of factors this represents
        - values: the values of all relevant variables
        - is_clamped: the clamped value of all relevant variables
    """

    factors: list[FactorNode]
    values: VariableStates
    is_clamped: VariableMapping

    # TODO(energy-jax): Fix annoying inits
    def __init__(
        self,
        factors: list[FactorNode],
        values: VariableStates,
        is_clamped: VariableMapping,
    ):
        """Init."""
        # super().__init__()  # TODO: so bad smh
        self.factors = factors
        self.values = values
        self.is_clamped = is_clamped

    def energy_function(
        self,
        x: State,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """
        Calculate the curried energy function (of the unclamped value) based on the state.

        Assumptions:
            - same order of variables
            - if the input is an array, then this is a single variable sample

        Args:
            - x: the input to evaluate for the unclamped variables.
        """
        energy = jnp.array(0.0)
        for fn in self.factors:
            variables = fn.connected_variables
            if fn.array_inputs:  # if factor is array EBM
                vals = [
                    self.values[var] if self.is_clamped[var] else x for var in variables
                ]
                x_combined: PyTree = jnp.concatenate(vals)
            else:  # dict EBM
                if fn.mapping is None:
                    raise ValueError(f"mapping function of {fn} cannot be None!")
                x_combined = {
                    fn.mapping[var]: self.values[var] if self.is_clamped[var] else x
                    for var in variables
                }
            energy += fn.factor.energy_function(x_combined)
        return jnp.squeeze(energy)

    def param_count(self) -> int:
        """Compute number of trainable parameters in EBM."""
        return sum(
            fn.factor.param_count() for fn in self.factors if fn.factor is not None
        )
