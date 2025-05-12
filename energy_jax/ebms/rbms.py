"""Restricted Boltzmann Machines."""

from typing import Optional, Dict, Any, Union, Tuple
from jax import numpy as jnp
import jax
from jaxtyping import Float, Array, Int, PRNGKeyArray
import equinox as eqx  # type: ignore[import]
from .ebm import AbstractDiscreteEBM
from ..utils import get_domain, stable_softmax


ParamTypeCRBM = Dict[str, Float[Array, "..."]]


def get_random_crbm_params(
    key: PRNGKeyArray,
    num_visible: int,
    num_hidden: int,
    max_dim: int = 2,
    sigma: float = 0.01,
) -> ParamTypeCRBM:
    """
    Generate random parameters for Categorical RBM.

    Args:
        - key (PRNGKey): the key to use to generate the random initialization
        - num_visible (int): the number of visible nodes
        - num_hidden (int): the number of hidden nodes
        - max_dim (int): the maximum dimension of any given visible node
        - sigma: the std of initialization

    Returns:
        - (dict): a dictionary with the "W" weights, "b" visible biases, and "c"
            hidden biases
    """
    mean = 0
    params = {}
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    params["W"] = mean + sigma * jax.random.normal(
        subkey1, (max_dim, num_hidden, num_visible)
    )
    params["b"] = mean + sigma * jax.random.normal(subkey2, (max_dim, num_visible))
    params["c"] = mean + sigma * jax.random.normal(subkey3, (num_hidden,))
    return params


# TODO: RBM sticks out too much, lets bring it in to the fold in EBM and Sampler space
class CategoricalRBM(AbstractDiscreteEBM, strict=True):
    """
    Categorical Restricted Boltzmann Machine (RBM) with Bernoulli hidden variables.

    Adapted from: https://github.com/mhdadk/rbm-dbn/tree/main
    Original paper: https://www.cs.toronto.edu/~hinton/absps/netflix.pdf

    The bitstings, along with all functions that rely on their usage, must be limited in
    size (probably to <2**15 Hilbert space).

    Visible inputs should be one hot encoded unless otherwise specified (see type hints
    for shaping of visible inputs).

    Attributes:
        - hid_dim (int): the number of hidden nodes
        - hid_structure (DeviceArray): the dimensions of each hidden node
        - max_categories (int): the maximum number of categories
        - rbm_bitstrings (tuple): all possible visible and hidden combinations
        - theta: the parameter pytree
    """

    hid_dim: int
    hid_structure: jnp.ndarray
    theta: ParamTypeCRBM
    num_visible: int
    structure: Int[Array, "dims"]
    hilbert_space: int
    bitstrings: Optional[Int[Array, "hilbert_space dims"]]
    hidden_bitstrings: Optional[Int[Array, "hilbert_space dims"]]
    max_categories: int

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        theta: Optional[ParamTypeCRBM] = None,
        structure: Optional[Int[Array, "dims"]] = None,
        generate_bitstrings: bool = False,
    ) -> None:
        """
        Initialize member variable.

        Args:
            - num_visible (int): number of visible nodes
            - num_hidden (int): the number of hidden nodes
            - theta (optional, dict): the parameters of the CRBM
            - structure (optional, list): the structure of the visible nodes
        """
        self.num_visible = num_visible
        theta = (
            get_random_crbm_params(
                jax.random.PRNGKey(0),
                num_visible,
                num_hidden,
                2 if structure is None else int(jnp.max(structure)),
            )
            if theta is None
            else theta
        )  # TODO: make key
        self.theta = theta
        if structure is not None:
            self.structure = structure
        else:
            self.structure = jnp.array([2] * num_visible)
        self.hilbert_space = int(jnp.prod(self.structure))
        self.max_categories = int(jnp.max(self.structure))
        self.hid_dim = num_hidden
        self.hid_structure = jnp.array([2] * num_hidden)
        if generate_bitstrings:
            self.bitstrings, self.hidden_bitstrings = (
                jax.nn.one_hot(get_domain(self.structure), self.max_categories),
                get_domain(self.hid_structure),
            )
        else:
            self.bitstrings = None
            self.hidden_bitstrings = None

    def probability_vector(
        self,
        return_par: Optional[bool] = False,
    ) -> Union[Tuple[Float[Array, "..."], Float[Array, ""]], Float[Array, "..."]]:
        """
        Compute the probabilities over the visible nodes.

        CAUTION: Do not use for large systems.

        Args:
            - return_par (optional, bool): whether or not to return the partition function

        Returns:
            - (DeviceArray): the probability vector
            - (optional, float): the partition function
        """
        exp = lambda x, y: jnp.exp(-1 * self.energy_function_two(x, y))
        inner_fn = lambda x: jnp.mean(
            jax.vmap(exp, in_axes=(None, 0))(x, self.hidden_bitstrings)
        )
        vec = jax.vmap(inner_fn)(self.bitstrings)
        prob_vector, par = stable_softmax(vec)

        return (prob_vector, par) if return_par else prob_vector

    def compute_ph_given_v(
        self, visible: Float[Array, "numvis maxcat"]
    ) -> Float[Array, "numhid"]:
        """
        Compute p(h|v).

        Args:
            - visible (DeviceArray): the visible nodes

        Returns:
            - (DeviceArray): the probabilities of the hidden nodes
        """
        # visible = [vis, cats]
        visible = visible.T * 2 - 1
        prod = jax.vmap(jnp.matmul)(self.theta["W"], visible)  # [cats, hidden]
        log_ph_given_v = jnp.sum(prod, axis=0) + self.theta["c"]
        ph_given_v = jax.nn.sigmoid(log_ph_given_v)
        return ph_given_v

    def sample_h_given_v(
        self,
        visible: Float[Array, "numvis maxcat"],
        key: PRNGKeyArray,
    ) -> Int[Array, "numhid"]:
        """
        Sample x ~ p(h|v).

        Args:
            - visible (DeviceArray): the visible nodes
            - key (PRNGKey): the key to use to sample
            - return_probs (optional, bool): whether or not to also return the
                exact probabilities

        Returns:
            - (DeviceArray): a sample from the hidden nodes
            - (optional, DeviceArray): the probabilities of the hidden nodes
        """
        ph_given_v = self.compute_ph_given_v(visible)
        h_given_v = jax.random.bernoulli(key, p=ph_given_v).astype("int8")
        return h_given_v

    def compute_pv_given_h(
        self, hidden: Int[Array, "numhid"]
    ) -> Float[Array, "numvis maxcat"]:
        """
        Compute p(v|h).

        Args:
            - hidden (DeviceArray): the hidden nodes

        Returns:
            - (DeviceArray): the probabilities of the visible nodes
        """

        def f(
            x: Float[Array, "numhid numvis"],
            y: Int[Array, "numhid"],
            z: Float[Array, "numvis"],
        ) -> Float[Array, "numvis"]:
            return jnp.matmul(x.T, y) + z

        # h = [hid]
        # W = [max, hid, vis]
        log_pv_given_h = jax.vmap(f, in_axes=(0, None, 0))(
            self.theta["W"], hidden, self.theta["b"]
        )  # [max, vis]
        pv_given_h = jax.nn.softmax(log_pv_given_h, axis=0)
        return pv_given_h.T

    def sample_v_given_h(
        self,
        hidden: Int[Array, "numhid"],
        key: PRNGKeyArray,
    ) -> Int[Array, "numvis maxcat"]:
        """
        Sample x ~ p(v|h).

        Args:
            - hidden (DeviceArray): the hidden nodes
            - key (PRNGKey): the key to use to sample
            - return_probs (optional, bool): whether or not to also return the
                exact probabilities

        Returns:
            - (DeviceArray): a sample from the visible nodes
            - (optional, DeviceArray): the probabilities of the visible nodes
        """
        pv_given_h = self.compute_pv_given_h(hidden)  # [vis, max]
        categories = jax.random.categorical(key, logits=jnp.log(pv_given_h))
        categories = jnp.clip(
            0, categories, self.structure - 1
        )  # clip to specific dimensions TODO: fix like in qrox
        v_given_h = jax.nn.one_hot(categories, self.max_categories)
        return v_given_h

    def energy_function_two(
        self,
        visible: Int[Array, "numvis maxcat"],
        hidden: Int[Array, "numhid"],
    ) -> Float[Array, ""]:
        """
        Alias of the energy function that allows multiple inputs.

        Visible inputs should be one hot encoded.

        Args:
            - v (DeviceArray): the visible node values
            - h (DeviceArray): the hidden node values

        Returns:
            - (float): the energy of these values
        """
        return self.energy_function((visible, hidden))

    def energy_function(
        self,
        x: Union[
            Tuple[Int[Array, "numvis maxcat"], Int[Array, "numhid"]],
            Dict[str, Int[Array, "..."]],
        ],
        **kwargs: Any,
    ) -> Float[Array, ""]:
        r"""
        Compute of the energy function.

        E = -(\sum_h \sum_cat W * v * h + \sum_cat W * b + h * b)

        Args:
            - x (tuple): the visible node values and the hidden node values

        Returns:
            - (float): the energy of these values
        """
        if isinstance(x, dict):
            visible, hidden = x["v"], x["h"]
        else:
            visible, hidden = x
        visible = visible.T * 2 - 1  # convert {0, 1} -> {-1, 1}
        first_term_inner = jax.vmap(jnp.matmul)(self.theta["W"], visible)  # [cat, hid]
        first_term_outer = jax.vmap(jnp.dot, in_axes=(None, 0))(
            hidden, first_term_inner
        )  # [cat]
        first_term = jnp.sum(first_term_outer)

        second_term_inner = jax.vmap(jnp.dot)(visible, self.theta["b"])
        second_term = jnp.sum(second_term_inner)

        third_term = jnp.dot(hidden, self.theta["c"])

        return -(first_term + second_term + third_term)

    def param_count(self) -> int:
        """Compute number of trainable parameters in EBM."""
        return sum(
            x.size
            for x in jax.tree_util.tree_leaves(
                eqx.filter(self.theta, eqx.is_inexact_array)
            )
        )
