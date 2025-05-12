"""Custom types."""

from .models.factor_graphs import VariableNode
from jaxtyping import PyTree, Shaped, ArrayLike

State = PyTree[Shaped[ArrayLike, "?state"], "State"]
VariableMapping = dict[VariableNode, bool]
VariableStates = PyTree[State, "VS"]
