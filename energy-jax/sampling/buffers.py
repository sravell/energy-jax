"""Persisent tooling for EBMs."""

from typing import Tuple
import equinox as eqx  # type: ignore[import]
from jax import numpy as jnp
import jax
from jaxtyping import Float, Array, Int, Num, PRNGKeyArray
from abc import abstractmethod


class AbstractReplayBuffer(eqx.Module, strict=True):
    """
    Base class for replay buffers.

    Currently only supports arrays!

    TODO: Support PyTrees.
    """

    buffer: eqx.AbstractVar[jnp.ndarray]
    buffer_size: eqx.AbstractVar[int]

    @abstractmethod
    def sample(self, key: PRNGKeyArray) -> Num[Array, "..."]:
        """
        Sample from the replay buffer.

        Args:
            - key: the random key to use when picking inputs to the sampler

        Returns:
            - the dictionary that the sampler returns
        """
        raise NotImplementedError("sample is not implemented.")


class DiscreteReplayBuffer(AbstractReplayBuffer, strict=True):
    """
    Replay buffer class for quasi-persistent updates of discrete EBMs.

    Use the `update_buffer` function to add to it.

    Attributes:
        - n_new: the number of new random samples to draw each time sampling is done
        - n_old: the number of old samples to use each time sampling is done
        - xshape: the shape of the sampled distribution
        - maxval: the maximum possible (exclusive) value of the distribution's samples
    """

    n_new: Tuple[int]
    n_old: Tuple[int]
    xshape: Tuple
    maxval: int
    buffer: jnp.ndarray
    buffer_size: int

    def __init__(
        self,
        buffer_shape: Tuple[int],
        maxval: int,
        num_chains: int,
        ratio_new: float,
        key: PRNGKeyArray,
    ) -> None:
        """
        Initialize member variables.

        Args:
            - buffer_shape: the shape of the complete buffer
            - maxval: the maxmal integer value of possible states (exclusive)
            - num_chains: the number of chains to run
            - ration_new: what ratio to draw the new samples as (e.g. 0.05)
            - key: the random key to use for initialization

        Returns:
            - None
        """
        self.buffer = jax.random.randint(
            key, shape=buffer_shape, minval=0, maxval=maxval
        )
        self.maxval = maxval
        self.n_new = (int(num_chains * ratio_new),)
        self.n_old = (int(num_chains - self.n_new[0]),)
        self.buffer_size = buffer_shape[0]
        self.xshape = buffer_shape[1:]

    def sample(self, key: PRNGKeyArray) -> Int[Array, "num_chains xshape"]:
        """
        Sample from the replay buffer.

        Args:
            - key: the random key to use when picking inputs to the sampler

        Returns:
            - the new samples
        """
        key, subkey = jax.random.split(key, 2)
        new_samples = jax.random.randint(
            subkey,
            minval=0,
            maxval=self.maxval,
            shape=self.n_new + self.xshape,
        )
        key, subkey = jax.random.split(key, 2)
        old_samples = jax.random.choice(subkey, self.buffer, shape=self.n_old)
        input_samples = jnp.concatenate((new_samples, old_samples), axis=0)
        # key, subkey = jax.random.split(key, 2)
        # gen_samples = self.sampler.sample_chains(model, input_samples, subkey, **kwargs)
        # gen_samples = jax.tree_util.tree_map(
        #    lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), gen_samples
        # )  # reshape from [num_chains, chain_length, *xshape] ->
        # [num_chains * chain length, *xshape]
        return input_samples


class ContinuousReplayBuffer(AbstractReplayBuffer, strict=True):
    """
    Replay buffer class for quasi-persistent updates of continuous EBMs.

    Use the `update_buffer` function to add to it.

    Attributes:
        - n_new: the number of new random samples to draw each time sampling is done
        - n_old: the number of old samples to use each time sampling is done
        - xshape: the shape of the sampled distribution
        - minval: the minimum value of possible samples
        - maxval: the maximum value of possible samples
    """

    n_new: Tuple[int]
    n_old: Tuple[int]
    xshape: Tuple
    minval: int
    maxval: int
    buffer: jnp.ndarray
    buffer_size: int

    def __init__(
        self,
        buffer_shape: Tuple[int],
        minval: int,
        maxval: int,
        num_chains: int,
        ratio_new: float,
        key: PRNGKeyArray,
    ) -> None:
        """
        Initialize member variables.

        Args:
            - buffer_shape: the shape of the complete buffer
            - minval: the minimum value of possible samples
            - maxval: the maximum value of possible samples
            - num_chains: the number of chains to sample from
            - ration_new: what ratio to draw the new samples as (e.g. 0.05)
            - key: the random key to use for initialization

        Returns:
            - None
        """
        self.buffer = jax.random.uniform(
            key, shape=buffer_shape, minval=minval, maxval=maxval
        )
        self.minval = minval
        self.maxval = maxval
        self.n_new = (int(num_chains * ratio_new),)
        self.n_old = (int(num_chains - self.n_new[0]),)
        self.buffer_size = buffer_shape[0]
        self.xshape = buffer_shape[1:]

    def sample(self, key: PRNGKeyArray) -> Float[Array, "num_chains xshape"]:
        """
        Sample from the replay buffer.

        Args:
            - key: the random key to use when picking inputs to the sampler

        Returns:
            - the new samples
        """
        key, subkey = jax.random.split(key, 2)
        new_samples = jax.random.uniform(
            subkey,
            minval=self.minval,
            maxval=self.maxval,
            shape=self.n_new + self.xshape,
        )
        key, subkey = jax.random.split(key, 2)
        old_samples = jax.random.choice(subkey, self.buffer, shape=self.n_old)
        input_samples = jnp.concatenate((new_samples, old_samples), axis=0)
        return input_samples


def update_buffer(
    buffer: AbstractReplayBuffer, new_examples: Float[Array, "xshape"]
) -> AbstractReplayBuffer:
    """
    Update the replay buffer.

    Args:
        - buffer: the old replay buffer
        - new_examples: an array of the new examples to add to the buffer

    Returns:
        - a new replay buffer with the updated array
    """
    new_array = jnp.concatenate((new_examples, buffer.buffer), axis=0)[
        : buffer.buffer_size
    ]
    return eqx.tree_at(lambda b: b.buffer, buffer, new_array)
