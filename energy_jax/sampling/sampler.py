"""Base sampler class."""

from typing import Tuple, Dict, Union, Any
import jax
from jax import numpy as jnp
import equinox as eqx  # type: ignore[import]
from jaxtyping import Int, Num, Array, PyTree, PRNGKeyArray, Float
from abc import abstractmethod
from ..ebms.ebm import AbstractEBM, AbstractDiscreteEBM
from ..utils import random_categorical_pytree, scan_wrapper, random_uniform_pytree

# TODO: refine typing
InputStateType = Union[None, PyTree, Num[Array, "xshape"], Num[Array, "chains xshape"]]


class AbstractSampler(eqx.Module, strict=True):
    """
    Base class for samplers.

    Both of the attributes are optional. If you wish to use run chains without ever using
    default initialization, i.e. you will always provide your own initialization, then
    xshape can be set to None. If you only intend to use run_chain, and not sample_chains,
    then you can set num_chains to None.

    Attributes:
        - xshape: the shape of the input to the EBM (the shape of the samples), or a pytree
            of the shape that pytree inputs would be (the values don't matter). Note that
            currently tuple PyTrees are not supported as this is how the sampler identifies
            that the array is desired over a pytree.
            TODO: add a specific pytree vs. array flag?
        - num_chains: the number of chains to vectorize or distribute and run
    """

    xshape: eqx.AbstractVar[Union[Tuple[int], PyTree, None]]
    num_chains: eqx.AbstractVar[Union[int, None]]

    @abstractmethod
    def step(
        self, model: AbstractEBM, state: PyTree, key: PRNGKeyArray, **kwargs: Any
    ) -> PyTree:
        """Take a single step of the chain."""
        raise NotImplementedError

    @abstractmethod
    def run_chain(
        self,
        model: AbstractEBM,
        state: Union[PyTree, None],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict[str, PyTree]:
        """Run a full chain of MCMC."""
        raise NotImplementedError

    @abstractmethod
    def _random_initial_state(
        self, subkey: PRNGKeyArray
    ) -> Union[PyTree[Float[Array, "..."]], Float[Array, "..."]]:
        """
        Generate a random discrete initial state.

        Args:
            - subkey: the random key to use

        Returns:
            - the random initial state
        """
        raise NotImplementedError

    def sample_chains(
        self,
        model: AbstractEBM,
        input_state: InputStateType,
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict[str, PyTree]:
        """
        Run multiple vectorized chains of MCMC.

        Note: Users are strongly encouraged to JIT this function or the encompassing
        function that contains this call.

        Note: any keyword arguments must be vmap-able! Additionally, any keyword arguments
        must be pre-filtered.

        Note: if you don't specify xshape and use this function, you must provide a
        (num_chains, xshape) array as input.

        If you are using PyTree inputs of your own, you must provide
        the input state over each chain, which means your pytree must have
        leaf shape (num_chains, shape)!

        Args:
            - model: the EBM to sample from
            - input_state: the current state
            - key: the random key to use

        Returns:
            - the results of all chains
        """
        if self.num_chains is None:
            raise ValueError(
                "You cannot call sample_chains without specifying the number of chains!"
            )
        if self.xshape is None and input_state is None:
            raise ValueError(
                "If you don't provide an xshape you must provide the inputs to the sampler!"
            )

        keys = jax.random.split(key, self.num_chains)

        def inner_step(
            input_state: InputStateType, key: PRNGKeyArray, **kwargs: Any
        ) -> Dict[str, PyTree]:
            return self.run_chain(model, input_state, key, **kwargs)

        if input_state is None or input_state.shape == self.xshape:
            return jax.vmap(inner_step, in_axes=(None, 0))(
                input_state, keys, **kwargs
            )  # filter_vmap can't filter kwargs, so this is to get around it
        return jax.vmap(inner_step, in_axes=(0, 0))(input_state, keys, **kwargs)

    def sample_chains_parallel(
        self,
        model: AbstractEBM,
        input_state: InputStateType,
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict[str, PyTree]:
        """
        Run multi-host distributed chains of MCMC.

        Users are strongly encouraged to JIT this function or the encompassing
        function that contains this call.

        Note: any keyword arguments must be pmap-able!

        Args:
            - model: the EBM to sample from
            - input_state: the current state
            - key: the random key to use

        Returns:
            - the results of the chains
        """
        if self.num_chains is None:
            raise ValueError(
                "You cannot call sample_chains without specifying the number of chains!"
            )
        if not isinstance(self.xshape, tuple):
            raise ValueError(
                "Currently distributed sampling only supports arrays and you must specify their shape!"
            )
        # The number of processes (hosts) to distribute.
        num_processes = jax.device_count()
        quotient = self.num_chains // num_processes
        remainder = self.num_chains % num_processes
        num_pad = num_processes - remainder

        # The nested pmap and vmap can't handle kwargs well, throwing TypeError.
        # This binding skill can resolve the error.
        _wrapped_run_chain = lambda m, i, k: self.run_chain(m, i, k, **kwargs)

        # Adds padding keys for pmap() requirement: shape must be dividable by
        #   the number of processes. `num_processes - remainder` are padded.
        # So, the total number of data to be distributed is:
        #   #chains + #pads
        #   = quotient * #processes + remainder + #pads
        #   = quotient * #processes + remainder + #processes - remainder
        #   = (quotient + 1) * #processes
        keys = jax.random.split(key, num_processes * (quotient + 1))
        # Reshapes the random keys to be distributed among processes.
        # Each JAX PRNGkey has shape of (2,), so the last dimension 2 is added.
        reshaped_keys = jnp.reshape(keys, (num_processes, quotient + 1, 2))  # type: ignore

        # This helper function removes the padded results, which gives
        # `num_chains` chains.
        def _unpad(padded_data: Array) -> Array:
            if not isinstance(self.xshape, tuple):
                raise ValueError(
                    "Currently distributed sampling only supports arrays and you must specify their shape!"
                )
            return padded_data.reshape(-1, *self.xshape)[: self.num_chains]

        if input_state is None or input_state.shape == self.xshape:
            samples = jax.pmap(
                jax.vmap(_wrapped_run_chain, in_axes=(None, None, 0)),
                in_axes=(None, None, 0),
            )(model, input_state, reshaped_keys, **kwargs)
            samples.update({"position": _unpad(samples["position"])})
            return samples

        # `input_state` has its own shape not compatible yet.
        # Pads `input_state` to be compatible.
        ndim = jnp.ndim(input_state)
        # Adds pads to the batch dimension (0), after the last element (1).
        # So pad_width location is [0, 1].
        pad_widths = [[0, num_pad]] + [[0, 0]] * (ndim - 1)
        # Just copies the last input state values.
        input_state = jnp.pad(input_state, pad_widths, "edge")
        # Reshapes `input_states` to be distributed among processes.
        reshaped_input_state = jnp.reshape(
            input_state, (num_processes, quotient + 1, *self.xshape)
        )
        samples = jax.pmap(
            jax.vmap(_wrapped_run_chain, in_axes=(None, 0, 0)),
            in_axes=(None, 0, 0),
        )(model, reshaped_input_state, reshaped_keys)
        samples.update({"position": _unpad(samples["position"])})
        return samples


class AbstractDiscreteSampler(AbstractSampler, strict=True):
    """
    Base discrete sampler class.

    Attributes:
        - burn_in_steps: the number of steps to burn in (i.e. samples to discard)
        - chain_steps: the number of steps to take after burn in (so total steps = burn_in + chain)
        - maxval: the maximum categorical value
    """

    burn_in_steps: eqx.AbstractVar[int]
    chain_steps: eqx.AbstractVar[int]
    maxval: eqx.AbstractVar[Union[int, None]]

    @abstractmethod
    def step(
        self,
        model: AbstractDiscreteEBM,  # type: ignore[override]
        state: Dict[str, PyTree],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict:
        """
        Take a single step of the chain.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the random key to use

        Returns:
            - the next state
        """
        raise NotImplementedError("step not implemented.")

    # TODO: make all this shape stuff better, like VBT
    def _random_initial_state(
        self, subkey: PRNGKeyArray
    ) -> Union[PyTree[Float[Array, "..."]], Float[Array, "..."]]:
        """
        Generate a random discrete initial state.

        Args:
            - subkey: the random key to use

        Returns:
            - the random initial state
        """
        if self.xshape is None or self.maxval is None:
            raise ValueError(
                "If you don't provide an xshape and maxval you must provide the input to the sampler!"
            )
        elif isinstance(self.xshape, tuple):
            return jax.random.randint(
                subkey, minval=0, maxval=self.maxval, shape=self.xshape
            )
        return random_categorical_pytree(subkey, self.xshape, self.maxval)


class AbstractDiscreteRunChain(AbstractDiscreteSampler, strict=True):
    """With run chains."""

    def run_chain(
        self,
        model: AbstractDiscreteEBM,  # type: ignore[override]
        state: Union[PyTree, Int[Array, "xshape"], None],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict:
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
            lambda state, key: self.step(model, state, key, **kwargs)
        )

        keys = jax.random.split(key, self.burn_in_steps + self.chain_steps)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return jax.tree_util.tree_map(lambda x: x[self.burn_in_steps :], states)


class AbstractContinuousSampler(AbstractSampler, strict=True):
    """
    Base class for continuous sampling.

    Note that minval and maxval are not used for clipping, purely for initialization.

    Attributes:
        - minval: the minimum value of the xshape array used in initialization
            (does not clip/clamp generated values)
        - maxval: the maximum value of the xshape array used in initialization
            (does not clip/clamp generated values)
    """

    minval: eqx.AbstractVar[Union[float, None]]
    maxval: eqx.AbstractVar[Union[float, None]]

    def _random_initial_state(
        self, subkey: PRNGKeyArray
    ) -> Union[PyTree[Float[Array, "..."]], Float[Array, "..."]]:
        """
        Generate a random initial state.

        Args:
            - subkey: the random key to use

        Returns:
            - the random initial state
        """
        if self.xshape is None or self.maxval is None or self.minval is None:
            raise ValueError(
                "If you don't provide an xshape and maxval and minval you must provide the input to the sampler!"
            )
        elif isinstance(self.xshape, tuple):
            return jax.random.uniform(
                subkey, minval=self.minval, maxval=self.maxval, shape=self.xshape
            )
        return random_uniform_pytree(subkey, self.xshape, self.minval, self.maxval)
