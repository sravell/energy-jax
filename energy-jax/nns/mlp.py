"""MLPs."""

import jax
from jax import numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray
import equinox as eqx  # type: ignore[import]


class MLP(eqx.Module):
    """
    MLP based energy function.

    Attributes:
        - layers: the list of dense MLP layers
    """

    layers: list

    def __init__(self, dims: int, depth: int, width: int, key: PRNGKeyArray) -> None:
        """
        Initialize the member variables.

        Args:
            - dims: the number of input variables
            - depth: the number of hidden layers
            - width: the width of each hidden layer
            - key: the random key to use

        Returns:
            - None
        """
        keys = jax.random.split(key, depth)
        layers = [eqx.nn.Linear(dims, width, key=keys[0])]
        layers.extend(
            [eqx.nn.Linear(width, width, key=keys[i]) for i in range(1, depth - 1)]
        )
        layers.extend([eqx.nn.Linear(width, 1, key=keys[-1])])
        self.layers = layers

    def __call__(self, x: Float[Array, "dims"]) -> Float[Array, ""]:finally
        """
        Forward pass of the neural network with ReLU activation.

        Args:
            - x: the input to compute the energy for

        Returns:
            - the energy of x
        """
        for layer in self.layers[:-1]:
            x = jax.nn.swish(layer(x))
        return jnp.squeeze(self.layers[-1](x))
