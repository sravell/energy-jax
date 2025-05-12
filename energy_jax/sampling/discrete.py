"""Discrete sampling tools."""

from typing import Dict, Optional, Tuple, Union, Any, Callable
from functools import partial, reduce
import operator
import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
import equinox as eqx  # type: ignore[import]
from jaxtyping import PyTree, Int, Array, Float, PRNGKeyArray
from ..ebms.ebm import AbstractDiscreteEBM
from ..ebms.rbms import CategoricalRBM
from .sampler import (
    AbstractDiscreteSampler,
    AbstractDiscreteRunChain,
)
from ..utils import scan_wrapper, random_categorical_pytree


@partial(jax.jit, static_argnames=["num"])
def random_categorical_hamming_distance(
    key: PRNGKeyArray,
    position: PyTree,
    num: int,
    maxval: int = 2,
) -> PyTree:
    """
    Generate a random sample a certain hamming distance from the input.

    Args:
        - key (PRNGKey): the random key used for the sample
        - position (pytree): the input/current binary position
        - num (int): the hamming distance of the new position
        - maxval: the maximum value to generate

    Returns:
        - (pytree): a new sample of the same shape as the input
    """
    pos, unravel_fn = ravel_pytree(position)
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(
        subkey, jnp.arange(len(pos)), shape=(num,), replace=False
    )
    key, subkey = jax.random.split(key)
    vals = jax.random.randint(subkey, minval=0, maxval=maxval, shape=(num,))
    return unravel_fn(pos.at[indices].set(vals))


def approx_difference_function_multi_dim(
    x: Int[Array, "xshape num_categories"], model: AbstractDiscreteEBM
) -> Float[Array, "xshape"]:
    r"""
    First order Taylor expansion of exp(f(x') - f(x)) distribution.

    $$ f(x') - f(x) = \nabla_x f(x) - x^T \nabla_x f(x) $$

    This computes d, where d is a matrix. $d_{i, j}$ approximates the log-likelihood ratio
    of flipping the i-th dimension of x from its current value to the value j.

    This is equation (4) in the GWG paper (see GWG docstring for more).

    Note that the model is sign flipped, since this expected f(x), but we are
    working with EBMs that give us -f(x)

    Args:
        - x: the current (multicategory discrete) state
        - model: the EBM

    Returns:
        - the approximate difference function
    """
    gx = eqx.filter_grad(lambda inp: -1 * model.energy_function(inp))(x)
    gx_cur = jnp.sum(gx * x, axis=-1)[..., None]
    return gx - gx_cur


class OneHotCategorical(eqx.Module, strict=True):
    """
    A pared down distribution adapted from distrax.

    Distrax license available here: https://github.com/google-deepmind/distrax/blob/master/LICENSE.

    Much of the code is adapted directly from distrax. We choose not to
    include a dependency since this is the only distribution we use.

    Attributes:
        - logits: the unnormalized log probabilities
        - probs: the probability distribution
        - num_categories: the number of categories the distribution is over
    """

    logits: jnp.ndarray
    probs: jnp.ndarray
    num_categories: int

    def __init__(self, logits: Float[Array, "xshape num_categories"]) -> None:
        """
        Compute the logits, probs, and num_categories.

        Args:
            - logits: the unnormalized log-probs. Can be any shape, but whatever shape they
                are is assumed to be the distribution shape. E.g. (28, 28, 2) indicates that
                the distribution is over (28, 28) and there are 2 categories.

        Returns:
            - None
        """
        self.num_categories = logits.shape[-1]
        self.logits = jax.nn.log_softmax(logits, axis=-1)  # normalize
        self.probs = jax.nn.softmax(self.logits, axis=-1)

    def sample(
        self, key: PRNGKeyArray, sample_shape: Tuple[int]
    ) -> Int[Array, "sample_shape xshape"]:
        """
        Generate samples from the categorical distribution.

        Note: this is not jit-able!

        Args:
            - key: the random key to use for sampling
            - sample_shape: just the leading dimensions of the samples. E.g. if
                your logits are (28, 28) and you want 100 samples, you
                would set sample shape = (100,) not (100, 28, 28).

        Returns:
            - the random samples
        """
        num_samples = reduce(operator.mul, sample_shape, 1)
        samples = self._sample_n(key, num_samples)
        return jax.tree_util.tree_map(
            lambda x: x.reshape(sample_shape + x.shape[1:]), samples
        )

    def sample_one(self, key: PRNGKeyArray) -> Int[Array, "xshape"]:
        """
        Jit friendly method of generating a single sample.

        Args:
            - key: the random key to use in the sample

        Returns:
            - the sample
        """
        samples = self._sample_n(key, 1)
        return jax.tree_util.tree_map(lambda x: jnp.squeeze(x), samples)

    def _sample_n(self, key: PRNGKeyArray, n: int) -> Int[Array, "n xshape"]:
        """
        Subroutine to generate n samples.

        Note: `_sample_n` is not jit-able, due to n being used in shaping!

        Args:
            - key: the random key to use for sampling
            - n: the number of samples to generate

        Returns:
            - an array of the samples
        """
        new_shape = (n,) + self.logits.shape[:-1]
        is_valid = jnp.logical_and(
            jnp.all(jnp.isfinite(self.probs), axis=-1, keepdims=True),
            jnp.all(self.probs >= 0, axis=-1, keepdims=True),
        )
        draws = jax.random.categorical(
            key=key, logits=self.logits, axis=-1, shape=new_shape
        )
        draws_one_hot = jax.nn.one_hot(draws, num_classes=self.num_categories).astype(
            self.logits.dtype
        )
        return jnp.where(is_valid, draws_one_hot, jnp.ones_like(draws_one_hot) * -1)

    def log_prob(
        self, value: Int[Array, "xshape num_categoires"]
    ) -> Float[Array, "xshape"]:
        """
        Compute the log probability, assuming value is one hot encoded.

        Args:
            - value: the one hot categorical array

        Returns:
            - the log probabilities
        """
        value_one_hot = value
        value_args = jnp.argmax(value, axis=-1)
        mask_outside_domain = jnp.logical_or(
            value_args < 0, value_args > self.num_categories - 1
        )
        return jnp.where(
            mask_outside_domain, -jnp.inf, jnp.sum(self.logits * value_one_hot, axis=-1)
        )

    def log_prob_categories(
        self, value: Int[Array, "xshape"]
    ) -> Float[Array, "xshape"]:
        """
        Compute the log probability, assuming value is not one hot encoded.

        Args:
            - value: the non one hot categorical array

        Returns:
            - the log probabilities
        """
        value_one_hot = jax.nn.one_hot(
            value, self.num_categories, dtype=self.logits.dtype
        )
        mask_outside_domain = jnp.logical_or(value < 0, value > self.num_categories - 1)
        return jnp.where(
            mask_outside_domain, -jnp.inf, jnp.sum(self.logits * value_one_hot, axis=-1)
        )


class GibbsWithGradient(AbstractDiscreteSampler, strict=True):
    """
    Approximate Gibbs sampling using gradient information.

    Based on this paper: https://arxiv.org/abs/2102.04509
    and associated implementation: https://github.com/wgrathwohl/GWG_release/tree/main

    TODO: add exact diff

    Note: does not currently support PyTree inputs, only arrays.

    Attributes:
        - temperature: the temperature used when generating the gradient, specifically, this
            modifies the computation so d(x) -> d(x) / temperature (see Algorithm 1 for more
            details).
        - penalize_current: how much to penalize a current proposal (to incentive a new proposal)
        - diff_fn: the method of computing d(x)
    """

    temperature: float
    penalize_current: float
    diff_fn: Callable
    burn_in_steps: int
    chain_steps: int
    maxval: Union[int, None]
    xshape: Union[Tuple[int], PyTree, None]
    num_chains: Union[int, None]

    def __init__(
        self,
        xshape: Union[Tuple[int], None],
        burn_in_steps: int,
        chain_steps: int,
        num_chains: int,
        maxval: Optional[int] = 2,
        temp: float = 1.0,
        penalize_current: float = 0.0,
        diff_fn: str = "approx",
    ) -> None:
        """
        Initialize member variables.

        Args:
            - xshape: the shape of the input to the EBM (the shape of the samples), or a pytree of
                the desired format
            - burn_in_steps: the number of burn in steps (i.e. steps that are discarded)
            - chain_steps: the number of steps per chain to use post burn in
            - num_chains: the number of chains to vectorize over
            - maxval: the maximum value of the categorical distribution
            - temp: the temperature of the d(x) calculation
            - penalize_current: how much to penalize the current proposal
            - diff_fn: the method to compute d(x)

        Returns:
            - None
        """
        self.xshape = xshape
        self.num_chains = num_chains
        self.burn_in_steps = burn_in_steps
        self.chain_steps = chain_steps
        self.maxval = maxval
        self.temperature = temp
        self.penalize_current = penalize_current
        if diff_fn == "approx":
            self.diff_fn = approx_difference_function_multi_dim
        else:
            raise ValueError("exact diff_fn is not yet supported.")

    def step(
        self,
        model: AbstractDiscreteEBM,  # type: ignore[override]
        state: Dict[str, PyTree],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict[str, PyTree]:
        r"""
        Compute a one step update proposal.

        Follows https://github.com/wgrathwohl/GWG_release/blob/main/samplers.py#L275.

        Computes
        $$ d = \nabla_x f(x) - x^T \nabla_x f(x) $$ -> forward_delta
        $$ q(i|x) = Categorical(Softmax(\frac{d}{temperature})) $$ -> cd_forward
        $$ sample i ~ q(i|x) $$ -> change
        $$ x' = changedims(x, i) $$ -> x_delta
        $$ d = \nabla_x f(x') - x'^T \nabla_x f(x') $$ -> reverse_delta
        $$ q(i|x') = Categorical(Softmax(\frac{d}{temperature})) $$ -> cd_reverse
        $$ A = \min \left ( \exp (-E(x') + E(x)) \frac{q(i|x')}{q(i|x)} \right ) $$ -> a

        where A is the acceptance probability.

        See GWG paper for more information.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the key to use

        Returns:
            - (dict): contains the "position" and "energy" (log prob)
        """
        x_cur = state["position"]
        if self.maxval is None:
            raise ValueError("maxval must be set for this sampler!")
        x_cur = jax.nn.one_hot(x_cur, self.maxval, dtype="float32")

        key, subkey1, subkey2 = jax.random.split(key, 3)
        temperature = self.temperature
        forward_delta = self.diff_fn(x_cur, model) / temperature
        penalize_current = (
            self.penalize_current
            if "penalize_current" not in kwargs
            else kwargs["penalize_current"]
        )
        # make sure we dont choose to stay where we are!
        forward_logits = forward_delta - penalize_current * x_cur
        cd_forward = OneHotCategorical(logits=forward_logits)

        change = cd_forward.sample_one(subkey1)

        # compute probability of sampling this change
        lp_forward = cd_forward.log_prob(change)
        # get binary indicator (xshape,) indicating which dim was changed
        changed_ind = jnp.sum(change, axis=-1)
        # mask out changed dim and add in the change
        x_delta = x_cur.copy() * (1.0 - changed_ind[..., None]) + change

        reverse_delta = self.diff_fn(x_delta, model) / temperature
        reverse_logits = reverse_delta - penalize_current * x_delta
        cd_reverse = OneHotCategorical(logits=reverse_logits)
        reverse_changes = x_cur * changed_ind[..., None]

        lp_reverse = cd_reverse.log_prob(reverse_changes)

        x_cur_energy = model.energy_function(x_cur)
        m_term = (
            -1 * model.energy_function(x_delta) + x_cur_energy
        )  # since model is an EBM
        la = m_term + lp_reverse - lp_forward
        a = jnp.array(
            jnp.exp(la) > jax.random.uniform(subkey2, shape=la.shape),
            dtype="float32",
        )[..., None]
        x_cur = x_delta * a + x_cur * (1.0 - a)

        output_state = {"position": jnp.argmax(x_cur, axis=-1), "energy": x_cur_energy}
        return output_state

    def run_chain(  # type: ignore[override]
        self,
        model: AbstractDiscreteEBM,
        state: Union[PyTree, Int[Array, "xshape"], None],
        key: PRNGKeyArray,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, PyTree]:
        """
        Run a full chain of MCMC.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the random key to use

        Returns:
            - the results of the chain
        """
        key, subkey = jax.random.split(key, 2)
        if state is not None:
            in_state = state
        elif self.xshape is None or self.maxval is None:
            raise ValueError(
                "If you don't provide an xshape and maxval you must provide the input to the sampler!"
            )
        elif isinstance(self.xshape, tuple):
            in_state = jax.random.randint(
                key=subkey, shape=self.xshape, minval=0, maxval=self.maxval
            )
        else:
            in_state = self._random_initial_state(subkey)

        initial_state = {"position": in_state}
        initial_state["energy"] = model.energy_function(
            initial_state["position"], **kwargs
        )

        one_step = scan_wrapper(
            lambda state, key: self.step(
                model, state, key, temperature=temperature, **kwargs
            )
        )

        keys = jax.random.split(key, self.burn_in_steps + self.chain_steps)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return jax.tree_util.tree_map(lambda x: x[self.burn_in_steps :], states)


class DiscreteUniformMH(AbstractDiscreteRunChain, strict=True):
    """Uniform proposal based MH MCMC."""

    burn_in_steps: int
    chain_steps: int
    maxval: Union[int, None]
    xshape: Union[Tuple[int], PyTree, None]
    num_chains: Union[int, None]

    def __init__(
        self,
        xshape: Union[Tuple[int], PyTree],
        burn_in_steps: int,
        chain_steps: int,
        num_chains: int,
        maxval: Optional[int] = 2,
    ):
        """Initialize member variables."""
        self.xshape = xshape
        self.num_chains = num_chains
        self.burn_in_steps = burn_in_steps
        self.chain_steps = chain_steps
        self.maxval = maxval

    def step(
        self,
        model: AbstractDiscreteEBM,  # type: ignore[override]
        state: Dict[str, PyTree],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict[str, PyTree]:
        """
        Uniform proposal.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the key to use

        Returns:
            - (dict): contains the "position" and "energy" (log prob)
        """
        old_position = state["position"]
        key, subkey = jax.random.split(key)
        if self.maxval is None:
            raise ValueError("maxval must be set for this sampler!")
        new_position = random_categorical_pytree(subkey, old_position, self.maxval)

        new_log_prob = model.energy_function(new_position, **kwargs)
        old_log_prob = state["energy"]
        acceptance = jnp.minimum(1.0, jnp.exp(-new_log_prob + old_log_prob))

        key, subkey = jax.random.split(key)
        next_state = jax.lax.cond(
            jax.random.uniform(subkey) >= acceptance,
            lambda x: {"energy": old_log_prob, "position": old_position},
            lambda x: {"energy": new_log_prob, "position": new_position},
            None,
        )
        return next_state


class DiscreteHammingMH(AbstractDiscreteRunChain, strict=True):
    """
    Hamming proposal based MH MCMC.

    In non-binary situations, the hamming distance is across a position, not just plus or minus
    one from a given category.

    Attributes:
        - hamming_distance: the maximum hamming distance to step
    """

    hamming_distance: int
    burn_in_steps: int
    chain_steps: int
    maxval: Union[int, None]
    xshape: Union[Tuple[int], PyTree, None]
    num_chains: Union[int, None]

    def __init__(
        self,
        xshape: Tuple[int],
        burn_in_steps: int,
        chain_steps: int,
        num_chains: int,
        maxval: Optional[int] = 2,
        hamming_distance: Optional[int] = 1,
    ):
        """Initialize member variables."""
        if hamming_distance is None:
            raise TypeError("hamming_distance must be an int.")
        self.hamming_distance = hamming_distance
        self.xshape = xshape
        self.num_chains = num_chains
        self.burn_in_steps = burn_in_steps
        self.chain_steps = chain_steps
        self.maxval = maxval

    def step(
        self,
        model: AbstractDiscreteEBM,  # type: ignore[override]
        state: Dict[str, PyTree],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict[str, PyTree]:
        """
        Hamming proposal.

        Args:
            - model: the EBM to sample from
            - inputs: the current state
            - key: the key to use

        Returns:
            - contains the "position" and "energy" (log prob)
        """
        old_position = state["position"]
        key, subkey = jax.random.split(key)
        if self.maxval is None:
            raise ValueError("maxval must be set for this sampler!")
        new_position = random_categorical_hamming_distance(
            subkey, old_position, self.hamming_distance, self.maxval
        )

        new_log_prob = model.energy_function(new_position, **kwargs)
        old_log_prob = state["energy"]
        acceptance = jnp.minimum(1.0, jnp.exp(-new_log_prob + old_log_prob))

        key, subkey = jax.random.split(key)
        next_state = jax.lax.cond(
            jax.random.uniform(subkey) >= acceptance,
            lambda x: {"energy": old_log_prob, "position": old_position},
            lambda x: {"energy": new_log_prob, "position": new_position},
            None,
        )
        return next_state


class CRBMGibbsSampler(AbstractDiscreteSampler, strict=True):
    """
    CRBM gibbs sampling.

    Note: does not currently support PyTree inputs, only arrays.
    """

    burn_in_steps: int
    chain_steps: int
    maxval: Union[int, None]
    xshape: Union[Tuple[int], PyTree, None]
    num_chains: Union[int, None]

    def __init__(
        self,
        xshape: Union[Tuple[int], None],
        burn_in_steps: int,
        chain_steps: int,
        num_chains: int,
    ):
        """Initialize member variables."""
        self.xshape = xshape if xshape is not None else False
        self.num_chains = num_chains
        self.burn_in_steps = burn_in_steps
        self.chain_steps = chain_steps
        self.maxval = None

    def step(
        self,
        model: CategoricalRBM,  # type: ignore[override]
        state: Dict[str, Int[Array, "numhid"]],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict:
        """
        Gibbs proposal.

        Args:
            - model: the CRBM to sample from
            - inputs: the current state
            - key: the key to use

        Returns:
            - contains the "position" (visible) and "energy" (which is associated with visible
                and hidden nodes) and "h"idden nodes
        """
        h_given_v = state["position"]["h"]
        key, subkey = jax.random.split(key)
        v_given_h = model.sample_v_given_h(h_given_v, subkey)
        key, subkey = jax.random.split(key)
        h_given_v = model.sample_h_given_v(v_given_h, subkey)
        return {
            "position": {"v": v_given_h, "h": h_given_v},
            "energy": model.energy_function((v_given_h, h_given_v), **kwargs),
        }

    def run_chain(
        self,
        model: CategoricalRBM,  # type: ignore[override]
        state: Union[Int[Array, "numvis maxcat"], None],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict:
        """
        Gibbs chain.

        Args:
            - model: the CRBM to sample from
            - inputs: the current state
            - key: the key to use

        Returns:
            - contains the "position" (visible) and "energy" (which is associated with visible
                and hidden nodes)
        """
        if state is None:

            clip_probs = lambda x, y: jnp.where(jnp.arange(len(x)) >= y, 0, x)
            logits = jax.vmap(clip_probs)(
                jnp.ones((model.num_visible, model.num_visible)), model.structure
            )
            key, subkey = jax.random.split(key)
            _init = jax.random.categorical(
                subkey,
                logits=jnp.log(logits),
                shape=(model.num_visible,),
            )
            state = jax.nn.one_hot(_init, num_classes=model.max_categories)
        key, subkey = jax.random.split(key, 2)
        keys = jax.random.split(key, self.burn_in_steps + self.chain_steps)
        h_given_v = model.sample_h_given_v(state, subkey)

        one_step = scan_wrapper(lambda state, key: self.step(model, state, key))

        _, states = jax.lax.scan(
            one_step,
            {
                "position": {"v": state, "h": h_given_v},
                "energy": model.energy_function((state, h_given_v), **kwargs),
            },
            keys,
        )

        return jax.tree_map(
            lambda x: jax.tree_map(lambda y: y[self.burn_in_steps :], x), states
        )
