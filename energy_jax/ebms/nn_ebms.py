"""Neural network energy functions."""

from typing import Optional, Any
from jaxtyping import Float, Array, Int
from jax.tree_util import tree_leaves
from jax import numpy as jnp
import equinox as eqx  # type: ignore[import]
from .ebm import AbstractEBM, AbstractDiscreteProbability
from ..utils import get_domain


class ContinuousNNEBM(AbstractEBM, strict=True):
    """
    NN EBM for continuous state spaces.

    Attributes:
        - nn: the neural network to use for the energy function
    """

    nn: eqx.Module

    def __init__(self, nn: eqx.Module) -> None:
        """
        Initialize member variables.

        Args:
            - nn: the neural network
            - dims: the input dimensions

        Returns:
            - None
        """
        self.nn = nn

    def energy_function(
        self, x: Float[Array, "dims"], **kwargs: Any
    ) -> Float[Array, ""]:
        """
        Forward pass of the EBM neural network.

        Args:
            - x: the input to the nn

        Returns:
            - the energy of the state
        """
        if not callable(self.nn):
            raise TypeError("nn must be callable!")
        return jnp.squeeze(self.nn(x, **kwargs))

    def param_count(self) -> int:
        """Compute number of trainable parameters in EBM."""
        return sum(
            x.size for x in tree_leaves(eqx.filter(self.nn, eqx.is_inexact_array))
        )


class DiscreteNNEBM(AbstractDiscreteProbability, strict=True):
    """
    NN EBM for discrete state spaces.

    Attributes:
        - nn: the NN to use for the energy function
    """

    nn: eqx.Module
    structure: Optional[Int[Array, "dims"]]
    hilbert_space: Optional[int]
    bitstrings: Optional[Int[Array, "hilbert_space dims"]]
    max_categories: Optional[int]

    def __init__(
        self,
        nn: eqx.Module,
        structure: Optional[Int[Array, "dims"]] = None,
        generate_bitstrings: bool = False,
    ) -> None:
        """
        Initialize member variables.

        Args:
            - nn: the neural network
            - dims: the input dimensions
            - structure: the structure of the inputs

        Returns:
            - None
        """
        self.structure = structure
        if structure is None:
            self.structure = structure
            self.hilbert_space = structure
            self.max_categories = structure
        else:
            self.structure = structure
            self.hilbert_space = int(jnp.prod(self.structure))
            self.max_categories = int(jnp.max(self.structure))
        if generate_bitstrings and self.structure is not None:
            self.bitstrings = get_domain(self.structure)
        else:
            self.bitstrings = None
        self.nn = nn

    def energy_function(
        self, x: Float[Array, "dims"], **kwargs: Any
    ) -> Float[Array, ""]:
        """
        Forward pass of the EBM neural network.

        Args:
            - x: the input to the nn

        Returns:
            - the energy of the state
        """
        if not callable(self.nn):
            raise TypeError("nn must be callable!")
        return jnp.squeeze(self.nn(x, **kwargs))

    def param_count(self) -> int:
        """Compute number of trainable parameters in EBM."""
        return sum(
            x.size for x in tree_leaves(eqx.filter(self.nn, eqx.is_inexact_array))
        )
