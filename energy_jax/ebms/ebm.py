"""General EBM class."""

from typing import Optional, Union, Callable, Tuple, Any
from jaxtyping import Array, Float, Num, Int, PyTree

from jax import numpy as jnp
from jax import tree_util as jtu
from jax import linen as nn

import jax


from abc import abstractmethod
from ..utils import stable_softmax

_EPSILON = 1e-8


class AbstractEBM(nn.Module):
    """Base Energy Based Model."""

    @abstractmethod
    def energy_function(self, x: Any, **kwargs: Any) -> Float[Array, ""]:
        """Energy function corresponding to the EBM."""
        raise NotImplementedError

    def score(self, x: Any, **kwargs: Any) -> Any:
        """Score (- grad(E)(x)) of the EBM."""
        return jax.tree_map(
            lambda i: -1 * i, jax.grad(self.energy_function)(x, **kwargs)
        )

    def exponential(self, x: Any, **kwargs: Any) -> Float[Array, ""]:
        """Exponentiated energy function."""
        return jnp.exp(-1 * self.energy_function(x, **kwargs))

    @abstractmethod
    def param_count(self) -> int:
        """Compute number of trainable parameters in EBM."""
        raise NotImplementedError


class AbstractDiscreteEBM(AbstractEBM):
    """Discrete state energy-based models."""

    structure: Optional[Int[Array, "dims"]]
    hilbert_space: Optional[int]
    bitstrings: Optional[PyTree[Int[Array, "hilbert_space dims"]]]
    max_categories: Optional[int]

    def probability_vector(
        self,
        return_par: Optional[bool] = False,
    ) -> Union[Tuple[Float[Array, "hspace"], Float[Array, ""]], Float[Array, "hspace"]]:
        """Compute probability vector and partition function."""
        vec = jax.vmap(lambda x: -self.energy_function(x))(self.bitstrings)
        prob_vector, par = stable_softmax(vec)
        return (prob_vector, par) if return_par else prob_vector

    def entropy(self) -> Float[Array, ""]:
        """Compute Shannon entropy."""
        vec = self.probability_vector()
        if isinstance(vec, tuple):
            raise TypeError("Cannot compute entropy of a Tuple")
        return jnp.sum(jax.scipy.special.entr(vec + _EPSILON))

    def expectation_value(
        self,
        function: Callable,
        fn_vals: Optional[Num[Array, "num length"]] = None,
        return_par: Optional[bool] = False,
    ) -> Union[Tuple[Float[Array, ""], Float[Array, ""]], Float[Array, ""]]:
        """Compute expectation value of a function."""
        vec, par = self.probability_vector(return_par=True)
        fn_vals = self.bitstrings if fn_vals is None else fn_vals
        val = jnp.inner(vec, jax.vmap(function)(fn_vals)).sum()
        return (val, par) if return_par else val
