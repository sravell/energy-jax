"""Adaptive step size tooling."""

from typing import Any, Union, Tuple, Optional, Dict
import jax
from jax import numpy as jnp
from jaxtyping import PyTree, Float, Array, PRNGKeyArray
import equinox as eqx  # type: ignore[import]
from .sampler import AbstractSampler
from ..ebms.ebm import AbstractModel
from ..utils import scan_wrapper
from abc import abstractmethod


class AbstractAdaptationState(eqx.Module, strict=True):
    """
    State manager for stateful adaptive operations.

    Since these operations carry over/effect multiple steps in a chain in a stateful way,
    i.e. we want information from the previous step to effect the next step, we
    create a state manager.

    Attributes:
        - current_step_size: the current step size (before the next round of updates)
    """

    current_step_size: eqx.AbstractVar[Float[Array, ""]]


class ErrorAdaptationState(AbstractAdaptationState, strict=True):
    """
    State manager for error based langevin adaptation.

    Attributes:
        - prev_error_ratio: the previous iterations error ratio (defaults to negative one
            which is equivalent to None, but more convenient for certain jax operations)
    """

    current_step_size: Float[Array, ""]
    prev_error_ratio: Float[Array, ""]

    def __init__(
        self,
        current_step_size: Float[Array, ""],
        prev_error_ratio: Optional[Float[Array, ""]] = None,
    ) -> None:
        """Initialize member variables."""
        self.current_step_size = current_step_size
        if prev_error_ratio is None:
            self.prev_error_ratio = jnp.array(-1.0)
        else:
            self.prev_error_ratio = prev_error_ratio


class AbstractAdaptiveSampler(AbstractSampler, strict=True):
    """
    Base class for adaptive samplers.

    This just defines a step function that depends on a state manager, in addition to a
    run chains function that scans over the states with the state manager.

    Attributes:
        - return_full_chain: whether or no to return the full chain rather than just
                the last sample
    """

    return_full_chain: eqx.AbstractVar[bool]

    @abstractmethod
    def step(
        self,
        model: AbstractModel,
        state: Union[PyTree[Float[Array, "..."]], Float[Array, "xshape"]],
        key: PRNGKeyArray,
        adaptive_state: Optional[AbstractAdaptationState] = None,
        **kwargs: Any,
    ) -> Tuple[PyTree, AbstractAdaptationState]:
        """Take a single adaptive step."""
        raise NotImplementedError

    @abstractmethod
    def _get_keys(self, key: PRNGKeyArray) -> PRNGKeyArray:
        """Get the keys used for the chain."""
        raise NotImplementedError

    @abstractmethod
    def _init_adaptive(self) -> AbstractAdaptationState:
        """Initialize adaptive state."""
        raise NotImplementedError

    def run_chain(
        self,
        model: AbstractModel,
        state: Union[PyTree[Float[Array, "..."]], Float[Array, "..."], None],
        key: PRNGKeyArray,
        adaptive_state: Optional[AbstractAdaptationState] = None,
        **kwargs: Any,
    ) -> Dict[str, Union[PyTree[Float[Array, "..."]], Float[Array, "..."]]]:
        """
        Run one chain of Adaptive MCMC.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the random key to use
            - adaptive_state: the adaptive state to start the chain from

        Returns:
            - the results of one chain
        """
        if adaptive_state is None:
            adaptive_state = self._init_adaptive()

        if state is None:
            key, subkey = jax.random.split(key, 2)
            initial_state = self._random_initial_state(subkey)
        else:
            initial_state = state

        keys = self._get_keys(key)

        one_step = scan_wrapper(
            lambda state, key: self.step(model, state[0], key, state[1], **kwargs),
            self.return_full_chain,
        )
        initial_state_adaptive = (
            initial_state,
            adaptive_state,
        )

        if self.return_full_chain:
            final_tuple, all_states = jax.lax.scan(
                one_step, initial_state_adaptive, keys
            )
            return {"position": all_states[0], "energy": None}

        final_tuple, _ = jax.lax.scan(one_step, initial_state_adaptive, keys)
        final_state = final_tuple[0]

        energy = eqx.filter_jit(model.energy_function)(final_state, **kwargs)
        return {"position": final_state, "energy": energy}
