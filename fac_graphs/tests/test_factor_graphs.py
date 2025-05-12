import unittest
from fac_graphs import FactorGraph, FactorNode, VariableNode
from energy_jax import nns
from energy_jax.ebms.nn_ebms import DiscreteNNEBM
import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class TestVariableNode(unittest.TestCase):
    def test_initialization(self):
        # Test default initialization
        node = VariableNode("test_node")
        self.assertEqual(node.name, "test_node")


class TestFactorNode(unittest.TestCase):
    def setUp(self):
        self.var1 = VariableNode("var1")
        self.var2 = VariableNode("var2")
        self.ebm = None  # Assuming EBM has a default constructor

    def test_initialization(self):
        # Test basic initialization
        factor_node = FactorNode([self.var1, self.var2], self.ebm, "factor")
        self.assertEqual(factor_node.name, "factor")
        self.assertEqual(factor_node.connected_variables, [self.var1, self.var2])
        self.assertEqual(factor_node.factor, self.ebm)


class TestFactorGraph(unittest.TestCase):
    def setUp(self):
        self.var1 = VariableNode("var1")
        self.var2 = VariableNode("var2")
        self.var3 = VariableNode("var3")
        self.var4 = VariableNode("var4")
        self.ebm = None  # Assuming EBM has a default constructor
        self.factor1 = FactorNode([self.var1, self.var2], self.ebm, "factor1")
        self.factor2 = FactorNode([self.var1, self.var3], self.ebm, "factor2")
        self.factor3 = FactorNode([self.var1, self.var4], self.ebm, "factor3")
        self.graph = FactorGraph([self.factor1, self.factor2])

    def test_initialization(self):
        self.assertListEqual(self.graph.factors, [self.factor1, self.factor2])
        self.assertListEqual(
            sorted(self.graph.variables), sorted([self.var1, self.var2, self.var3])
        )

    def test_factor_incidence_dict(self):
        expected_dict = {
            self.factor1: sorted([self.var1, self.var2]),
            self.factor2: sorted([self.var1, self.var3]),
        }
        self.assertDictEqual(self.graph.factor_incidence_dict, expected_dict)

    def test_variable_incidence_dict(self):
        expected_dict = {
            self.var1: [self.factor1, self.factor2],
            self.var2: [self.factor1],
            self.var3: [self.factor2],
        }
        for i in expected_dict:
            self.assertSetEqual(
                set(self.graph.get_incident_factors(i)), set(expected_dict[i])
            )


class TestFactorGraphTraining(unittest.TestCase):
    def setUp(self):
        key = jax.random.PRNGKey(0)
        self.var1 = VariableNode("var1")
        self.var2 = VariableNode("var2")
        self.var3 = VariableNode("var3")
        self.ebm1 = DiscreteNNEBM(
            nns.MLP(dims=4, depth=3, width=10, key=key),
            structure=jnp.array([2, 2, 2, 2]),
            generate_bitstrings=True,
        )
        self.ebm2 = DiscreteNNEBM(
            nns.MLP(dims=4, depth=3, width=10, key=key),
            structure=jnp.array([2, 2, 2, 2]),
            generate_bitstrings=True,
        )
        self.factor1 = FactorNode([self.var1, self.var2], self.ebm1, "factor1")
        self.factor2 = FactorNode([self.var1, self.var3], self.ebm2, "factor2")
        self.graph = FactorGraph([self.factor1, self.factor2])

        def test(FG):
            return jnp.array([1.0])

        self.test = test

        def loss(FG, x):
            energy = 0
            for factor in FG.factors:
                energy += factor.factor.energy_function(x)
            return energy

        self.loss = loss

    def test_pprint(self):
        self.graph.pprint()
        self.assertTrue(True)

    def test_jitability_factor_graph(self):
        graph = FactorGraph([self.factor1, self.factor2])
        out = eqx.filter_jit(self.test)(graph)
        self.assertEqual(out, jnp.array([1.0]))

    def test_training_factor_graph(self):
        x = jnp.ones((4,))
        graph = FactorGraph([self.factor1, self.factor2])
        loss_value_grad = eqx.filter_value_and_grad(self.loss)

        optimizer = optax.adam(learning_rate=0.1)
        opt_state = optimizer.init(eqx.filter(graph, eqx.is_array))

        # @eqx.filter_jit
        def step(FG, x, opt_state):
            _, grads = loss_value_grad(FG, x)
            updates, opt_state = optimizer.update(grads, opt_state)
            FG = eqx.apply_updates(FG, updates)
            return FG, opt_state

        for i in range(40):
            graph, opt_state = step(graph, x, opt_state)

        # probability of 1111 state must be 1. This corresponds to index -1
        # in probability_vector
        energies = eqx.filter_vmap(self.factor1.factor.energy_function)(
            self.factor1.factor.bitstrings
        )
        probs = jax.nn.softmax(-energies)
        self.assertEqual(probs[-1], 1.0)
