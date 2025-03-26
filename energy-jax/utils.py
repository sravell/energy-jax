"""EBM utility functions."""

from functools import wraps
from typing import Optional, Union, Any, Callable
from jax import numpy as jnp
from jax import lax
import jax
from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray


def kronecker_delta(i: Int[Array, "..."], j: Int[Array, "..."]) -> Int[Array, "..."]:
    """
    Compute the Kronecker delta function.

    Args:
        - i: value one
        - j: value 2

    Returns:
        - 1s where i == j and 0s otherwise, same shape as i
    """
    return jnp.where(i == j, 1, 0)


def stable_softmax(
    inputs: Float[Array, "..."],
    axis: Optional[Union[int, tuple[int, ...], None]] = -1,
    where: Optional[Any] = None,
    initial: Optional[Any] = None,
) -> tuple[Float[Array, "..."], Float[Array, ""]]:
    r"""Softmax function.

    Computes the function which rescales elements to the range :math:`[0, 1]`
    such that the elements along :code:`axis` sum to :math:`1`.

    .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Adapted from jax, license avaible https://github.com/google/jax/blob/main/LICENSE

    Args:
        - inputs (DeviceArray): input array
        - axis (optional, int or tuple): the axis or axes along which the softmax s
            hould be computed. The softmax output summed across these dimensions
            should sum to :math:`1`. Either an integer or a tuple of integers.
        - where (optional, sequence): Elements to include in the :code:`softmax`.
        - initial (float): The minimum value used to shift the input array. Must be present
            when :code:`where` is not None.

    Returns:
        - (DeviceArray): the softmax'ed array
        - (float): the deominator/partition function of the softmax
    """
    x_max = jnp.max(inputs, axis, where=where, initial=initial, keepdims=True)
    unnormalized = jnp.exp(inputs - lax.stop_gradient(x_max))
    denominator = jnp.sum(unnormalized, axis, where=where, keepdims=True)
    return unnormalized / denominator, jnp.squeeze(denominator)


def int_to_bin(
    samples: Int[Array, "samples"], dim: int
) -> Int[Array, "samples qubits"]:
    """
    Convert integers to binary.

    In its current form, this function is NOT usable with jax.grad().

    Args:
        - samples (DeviceArray): an array of categorical samples
        - dim (int): the total qubits in the system

    Returns:
        - (DeviceArray): a (samples, qubits) array that represents the counts for each samples
    """
    return jnp.flip(samples[:, None] & (1 << jnp.arange(dim)) > 0, axis=-1).astype(
        "int8"
    )


def get_domain(structure: Int[Array, "nodes"]) -> Int[Array, "combinations"]:
    """
    Generate the domain corresponding to a discrete node structure.

    The domain of a function is the set of viable inputs. This is conceptually equivalent
    to the set of basis states.

    Args:
        - structure (DeviceArray): the list of dims for each node, e.g. [2, 2, 2] for qubits

    Returns:
        - (DeviceArray): the combinations that make up the entire domain
    """
    n_qudits = len(structure)
    jnp_structure = jnp.array(structure)
    size = jnp.prod(jnp_structure)
    x = jnp.tile(jnp.expand_dims(jnp.arange(size), axis=-1), (1, n_qudits))
    base = jnp.concatenate([jnp.array([1]), jnp.cumprod(jnp_structure[::-1])])[
        n_qudits - 1 :: -1
    ]
    y = jnp.tile(jnp.expand_dims(base, axis=0), (size, 1))
    weights = jnp.expand_dims(jnp_structure, axis=0)
    y_mod = jnp.tile(weights, (size, 1))
    return (x // y) % y_mod


def scan_wrapper(func: Callable, log_all: bool = True) -> Callable:
    """
    Generate a function wrapper that returns two of the function.

    This is useful since `jax.lax.scan` requires two return values, so we can
    either duplicate our return value for logging, or have the second return
    value be None. This is important for memory constraints, as if you have a
    large change jnp.stack(func_eval) over the whole chain may result in OOM.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html for more.

    Args:
        - func: a function to wraper
        - log_all: whether the second returned argument should be None
            or not

    Returns:
        - the wrapped, double return function
    """
    if log_all:

        @wraps(func)
        def double_return(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            func_eval = func(*args, **kwargs)
            return func_eval, func_eval

    else:

        @wraps(func)
        def double_return(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            func_eval = func(*args, **kwargs)
            return func_eval, None

    return double_return


def softargmax(inputs: Float[Array, "... args"], beta: float) -> Float[Array, "..."]:
    """
    Differentiable approximation to argmax.

    Adapted from TF implementation:
    https://stackoverflow.com/questions/46926809/getting-around-tf-argmax-which-is-not-differentiable

    Args:
        - inputs: the x to compute argmax over the last axis
        - beta: the inverse temperature, ideally beta -> jnp.inf, but this causes
            numerical instabilities as beta increases

    Returns:
        - the argmax-ed over array
    """
    return jnp.sum(
        jax.nn.softmax(inputs * beta) * jnp.arange(0, inputs.shape[-1]), axis=-1
    )


def random_uniform_pytree(
    key: PRNGKeyArray,
    pytree: PyTree[jax.ShapeDtypeStruct],
    minval: float = -1.0,
    maxval: float = 1.0,
) -> PyTree[Array]:
    """
    Generate a random uniform pytree.

    Args:
        - key: the random key to be used for the sample
        - pytree: a pytree of the desired shape
        - maxval: the maximum value to generate

    Returns:
        - a random uniform pytree of the same shape as the input
    """
    treedef = jax.tree_util.tree_structure(pytree)
    keys = jax.tree_util.tree_unflatten(
        treedef, jax.random.split(key, treedef.num_leaves)
    )
    return jax.tree_util.tree_map(
        lambda a, b: jax.random.uniform(
            key=b, shape=a.shape, dtype=a.dtype, minval=minval, maxval=maxval
        ),
        pytree,
        keys,
    )


def random_categorical_pytree(
    key: PRNGKeyArray, pytree: PyTree[jax.ShapeDtypeStruct], maxval: int = 2
) -> PyTree[Array]:
    """
    Generate a random categorical pytree.

    Args:
        - key: the random key to be used for the sample
        - position: a pytree of the desired shape
        - maxval: the maximum value to generate

    Returns:
        - a random categorical pytree of the same shape as the input
    """
    treedef = jax.tree_util.tree_structure(pytree)
    keys = jax.tree_util.tree_unflatten(
        treedef, jax.random.split(key, treedef.num_leaves)
    )
    return jax.tree_util.tree_map(
        lambda a, b: jax.random.randint(
            key=b, shape=a.shape, dtype=a.dtype, minval=0, maxval=maxval
        ),
        pytree,
        keys,
    )
