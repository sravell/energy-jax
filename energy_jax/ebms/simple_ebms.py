"""Class containing classic (non-neural network) based EBMs."""

from typing import Dict, Optional, Any
from jax import numpy as jnp
from jax.tree_util import tree_leaves
import equinox as eqx
import networkx as nx  # type: ignore[import]
from jaxtyping import Float, Array, Num, Int
from ..utils import kronecker_delta, get_domain
from .ebm import AbstractDiscreteProbability, AbstractEBM


ParamType = Dict[str, Float[Array, "params"]]  # TODO: refine?


class PottsEBM(AbstractDiscreteProbability, strict=True):
    """
    Potts energy based model.

    Ising like model for multidimensional nodes.

    Attributes:
        - adj: the adjacency matrix
        - e_fn_type: which type of energy function to use
        - theta: the parameter pytree
    """

    adj: jnp.ndarray
    e_fn_type: str
    theta: ParamType
    structure: Int[Array, "dims"]
    hilbert_space: int
    bitstrings: Optional[Int[Array, "hilbert_space dims"]]
    max_categories: int

    def __init__(
        self,
        graph: nx.Graph,
        theta: ParamType,
        structure: Optional[Int[Array, "dims"]] = None,
        e_fn_type: Optional[str] = None,
        generate_bitstrings: bool = False,
    ) -> None:
        """
        Initialize member variables.

        Args:
            - graph: the graph of the Potts model
            - theta: the parameters of the EBM
            - structure: the dimensions of each node
            - e_fn_type: the method of computing the energy

        Returns:
            - None
        """
        dims = len(graph.nodes)
        if structure is None:
            self.structure = jnp.array([2] * dims)
        else:
            self.structure = structure
        if generate_bitstrings:
            self.bitstrings = get_domain(self.structure)
        else:
            self.bitstrings = None
        self.hilbert_space = int(jnp.prod(self.structure))
        self.max_categories = int(jnp.max(self.structure))

        e_fn_type = "vector" if e_fn_type is None else e_fn_type
        self.adj = jnp.array(nx.adjacency_matrix(graph).todense())
        self.e_fn_type = e_fn_type
        self.theta = theta

    def energy_function(
        self, x: Num[Array, "nodes"], **kwargs: Any
    ) -> Float[Array, ""]:
        """
        Compute the energy of the state given the parameters.

        Args:
            - x: the input array

        Returns:
            - the energy of x given theta
        """
        if self.e_fn_type == "vector":
            return self.cos_energy_function(x)
        if self.e_fn_type == "kron":
            return self.kron_energy_function(x)
        raise ValueError("Invalid energy function type.")

    def cos_energy_function(self, x: Int[Array, "nodes"]) -> Float[Array, ""]:
        r"""
        Cosine energy function.

        $$ E = \sum_{i, j} J_{i, j} cos(2 \pi s_i / q - 2 \pi s_j / q) + \sum_i h_i s_i $$

        x = [0, 1, 2, 4, 2, 1] (i.e. any integer less than max(structure))

        Args:
            - x: state to evaluate energy of

        Returns:
            - the cosine energy of the state
        """
        edge_weights = jnp.triu(self.adj * self.theta["edges"], k=1)
        edge_terms = jnp.triu(
            eqx.filter_vmap(
                lambda x, y: jnp.cos(
                    2 * jnp.pi * x / self.max_categories
                    - 2 * jnp.pi * y / self.max_categories
                ),
                in_axes=(0, None),
            )(x, x),
            k=1,
        )
        two_body_term = jnp.sum(eqx.filter_vmap(jnp.dot)(edge_weights, edge_terms))
        one_body_term = jnp.dot(self.theta["nodes"], x)
        return two_body_term + one_body_term

    def kron_energy_function(self, x: Int[Array, "nodes"]) -> Float[Array, ""]:
        r"""
        Cosine energy function.

        $$ E = \sum_{i, j} J_{i, j} \delta (s_i, s_j) + \sum_i h_i s_i $$

        x = [0, 1, 2, 4, 2, 1] (i.e. any integer less than max(structure))

        Args:
            - theta: parameters of the EBM
            - x: state to evaluate energy of

        Returns:
            - the cosine energy of the state
        """
        edge_weights = jnp.triu(self.adj * self.theta["edges"], k=1)
        edge_terms = jnp.triu(
            eqx.filter_vmap(kronecker_delta, in_axes=(0, None))(x, x), k=1
        )
        two_body_term = jnp.sum(eqx.filter_vmap(jnp.dot)(edge_weights, edge_terms))
        one_body_term = jnp.dot(self.theta["nodes"], x)
        return two_body_term + one_body_term

    def param_count(self) -> int:
        """Compute number of trainable parameters in EBM."""
        return sum(
            x.size for x in tree_leaves(eqx.filter(self.theta, eqx.is_inexact_array))
        )


class BoltzmannEBM(AbstractDiscreteProbability, strict=True):
    """
    A Boltzmann machine over a specific graph.

    Boltzmann machines are defined over binary variables, and as such,
    this only formally supports qubits. However, it has been used for qudit based systems
    to some success, but these results are not theoretically supported.

    Attributes:
        - adj: the adjacency matrix
        - theta: the parameter pytree
    """

    adj: jnp.ndarray
    theta: ParamType
    structure: Int[Array, "dims"]
    hilbert_space: int
    bitstrings: Optional[Int[Array, "hilbert_space dims"]]
    max_categories: int

    def __init__(
        self,
        graph: nx.Graph,
        theta: ParamType,
        structure: Optional[Int[Array, "dims"]] = None,
        generate_bitstrings: bool = False,
    ) -> None:
        """
        Initialize member variables.

        Args:
            - graph: the graph the EBM operates on
            - weighted_ad_fn (callable): a function to compute the weighted adjacency matrix
            - theta (optional, DeviceArray): the initial parameters of the EBM

        Returns:
            - None
        """
        dims = len(graph.nodes)
        self.adj = jnp.array(nx.adjacency_matrix(graph).todense())
        dims = len(graph.nodes)
        if structure is None:
            self.structure = jnp.array([2] * dims)
        else:
            self.structure = structure
        if generate_bitstrings:
            self.bitstrings = get_domain(self.structure)
        else:
            self.bitstrings = None
        self.hilbert_space = int(jnp.prod(self.structure))
        self.max_categories = int(jnp.max(self.structure))
        self.theta = theta

    def energy_function(
        self, x: Num[Array, "nodes"], **kwargs: Any
    ) -> Float[Array, ""]:
        """
        Compute the energy for a specific set of parameters.

        Specifically, given edge params w_ij and node params theta_i, this
        returns a function that computes E(x_i) = sum_{i < j} w_ij x_i x_j + sum_i theta_i s_i

        Args:
            - theta (dict): dictionary of "nodes" and "edges" parameters. The edge
                params are in a sparse form and in the same order as the edges in the graph
            - x (DeviceArray): the inputs to calculate the energy for

        Returns:
            - (float): the energy of the EBM
        """
        x = 2 * x - 1
        return jnp.inner(
            x, (self.adj * jnp.triu(self.theta["edges"], k=1)) @ x
        ) + jnp.inner(self.theta["nodes"], x)

    def param_count(self) -> int:
        """Compute number of trainable parameters in EBM."""
        return sum(
            x.size for x in tree_leaves(eqx.filter(self.theta, eqx.is_inexact_array))
        )


class ContinuousBoltzmannEBM(AbstractEBM, strict=True):
    """
    A continuous version of the Boltzmann EBM.

    This is essentially the same concept, just with flexibility in the inputs, allowing
    for continuous parameters to be based through a tanh.

    Attributes:
        - adj: the adjacency matrix
        - theta: the parameters of the EBM
    """

    theta: dict
    adj: jnp.ndarray

    def __init__(self, graph: nx.Graph, theta: ParamType) -> None:
        """
        Initialize the member variables.

        Args:
            - theta: dictionary of "nodes" and "edges" parameters. The edge
                params are in a sparse form and in the same order as the edges in the graph
            - graph: the graph to use

        Returns:
            - Non
        """
        self.adj = jnp.array(nx.adjacency_matrix(graph).todense())
        self.theta = theta

    def energy_function(
        self, x: Float[Array, "nodes"], **kwargs: Any
    ) -> Float[Array, ""]:
        r"""
        Compute the energy for a specific set of parameters.

        Specifically, given edge params w_ij and node params theta_i, this
        returns a function that computes

        $$ E(x_i) = sum_{i < j} w_ij tanh(x_i) tanh(x_j) + sum_i theta_i tanh(x_i) $$


        Args:
            - x (DeviceArray): the inputs to calculate the energy for

        Returns:
            - (float): the energy of the EBM
        """
        x = jnp.tanh(x)
        return jnp.inner(
            x, (self.adj * jnp.triu(self.theta["edges"], k=1)) @ x
        ) + jnp.inner(self.theta["nodes"], x)

    def param_count(self) -> int:
        """Compute number of trainable parameters in EBM."""
        return sum(
            x.size for x in tree_leaves(eqx.filter(self.theta, eqx.is_inexact_array))
        )
