"""Defining classes for BP on factor graphs with EBMs."""

from typing import Union, Optional, Set
from energy_jax.ebms import ebm
import equinox as eqx


class AbstractNode(eqx.Module, strict=True):
    """Abstract node class."""

    name: eqx.AbstractVar[str]

    def __hash__(self) -> int:
        """Hash the object."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Test equality of nodes."""
        if not isinstance(other, AbstractNode):
            raise NotImplementedError
        return self.name == other.name

    def __gt__(self, other: object) -> bool:
        """Test greater than of nodes."""
        if not isinstance(other, AbstractNode):
            raise NotImplementedError
        return self.name > other.name  # type: ignore

    def __lt__(self, other: object) -> bool:
        """Test less than of nodes."""
        if not isinstance(other, AbstractNode):
            raise NotImplementedError
        return self.name < other.name  # type: ignore


class VariableNode(AbstractNode, strict=True):
    """
    A class for variable nodes in a factor graph.

    Attributes:
        name: The id of the variable node.
    """

    name: str


# TODO: docstrings
class FactorNode(AbstractNode, strict=True):
    """
    A class for factor nodes in a factor graph.

    Note: the order of the connected variables is the order that will be used
    to concatenate information (if array_inputs is True) when sampling from
    this FactorNode.

    Attributes:
        connected_variables: The list of connected variable nodes.
        factor: The energy-based model associated with the factor node.
        name: The ID of the factor node.
        array_inputs: whether the inputs to the EBM are solely of an array type or of a
            dictionary type
    """

    name: str
    connected_variables: list[VariableNode]
    factor: ebm.AbstractEBM
    array_inputs: bool  # TODO: do we need this?
    mapping: Union[dict[VariableNode, str], dict[VariableNode, VariableNode]]

    def __init__(
        self,
        connected_variables: list[VariableNode],
        factor: ebm.AbstractEBM,
        name: str,
        array_inputs: bool = True,
        mapping: Optional[dict[VariableNode, str]] = None,
    ) -> None:
        """Initialize a factor node in a factor graph.

        Note: the `array_inputs` is used to determine whether the input to the
        factor's EBM is all arrays (and as such can simply be concatenated when doing
        gibbs sampling) or of a dictionary (pytree) type. If the factor has multiple variables
        and these variables are not of trivially concatenatable shapes (e.g. (10, 2), (3, 4, 5)),
        this must be set to false. Additionally, any sampler used with this factor must match this
        approach. Specifically, if it is array only, the sampler must be compatible with the
        concatenated shape of all variables. If `array_inputs` is false, the sampler must be
        compatible with dictionary based inputs. Note that this may require some consideration
        on the users part, since some samplers, e.g. HMC, require parameters dependent on the input
        space and as such the user must know this over the full concatenated space or over the full
        flattened pytree space. Not all samplers require this though.

        Args:
            connected_variables: The list of connected variable nodes.
            factor: The energy-based model associated with the factor node.
            name: The name of the factor node.
            array_inputs: whether the input to the factor EBM are only arrays. Defaults to True.
        """
        self.name = name
        self.connected_variables = connected_variables
        self.factor = factor
        self.array_inputs = array_inputs
        if mapping is None:
            self.mapping = {var: var for var in connected_variables}
        else:
            self.mapping = mapping


class FactorGraph(eqx.Module, strict=True):
    """
    A class for factor graphs with energy-based models on the factors.

    Attributes:
        variable_incidence_dict: A dictionary of the variables and their adjacent factor nodes.
    """

    factors: list[FactorNode]
    variable_incidence_dict: dict[VariableNode, list[int]]

    def __init__(self, factor_list: list[FactorNode]) -> None:
        """
        Initialize a factor graph.

        Args:
            factor_list: A list of factor nodes.
        """
        if any([not isinstance(i, FactorNode) for i in factor_list]):
            raise ValueError("Ill formated factor list, must be all factor nodes")
        self.factors = factor_list
        self.variable_incidence_dict = self._get_variable_incidence_dict()

        if len(set(factor_list)) != len(factor_list):
            raise ValueError("Duplicate factors in factor graph.")

        names = [factor.name for factor in factor_list]
        if len(set(names)) != len(names):
            raise ValueError("Same name used twice for different factors.")

    @property
    def variables(self) -> list[VariableNode]:
        """Var."""
        return list(self.variable_incidence_dict.keys())

    @property
    def factor_incidence_dict(self) -> dict[FactorNode, list[VariableNode]]:
        """Create a dictionary of the factors and their adjacent variables."""
        return {f: sorted(f.connected_variables) for f in self.factors}

    def _get_variable_incidence_dict(self) -> dict[VariableNode, list[int]]:
        """Create a dictionary of the variables and their adjacent factors."""
        inc_dict: dict[VariableNode, Set[FactorNode]] = {}
        for factor in self.factors:
            for variable in factor.connected_variables:
                inc_dict[variable] = inc_dict.get(variable, set())
                inc_dict[variable].add(factor)

        ret_dict = {}
        for i in inc_dict:
            ret_dict[i] = [self.factors.index(j) for j in sorted(list(inc_dict[i]))]
        return ret_dict

    def get_incident_factors(self, var: VariableNode) -> list[FactorNode]:
        """Grab the incident factors to a variable."""
        return [self.factors[i] for i in self.variable_incidence_dict[var]]

    def pprint(self) -> None:
        """Pretty print the factor graph showing its factors, variables, and connections."""
        print("Factor Graph Structure")
        print("======================")
        print("Variables:")
        for variable in sorted(self.variables, key=lambda v: v.name):
            print(f"  Variable Node '{variable.name}'")

        print("\nFactors:")
        for factor in sorted(self.factors, key=lambda f: f.name):
            connected_vars = ", ".join(
                [str(v.name) for v in factor.connected_variables]
            )
            print(
                f"  Factor Node '{factor.name}': Connected Variables=[{connected_vars}], Array Inputs={factor.array_inputs}, Mapping={factor.mapping}"
            )

        print("\nConnections:")
        for factor in sorted(self.factors, key=lambda f: f.name):
            for variable in factor.connected_variables:
                print(
                    f"  Factor Node '{factor.name}' <-> Variable Node '{variable.name}'"
                )
