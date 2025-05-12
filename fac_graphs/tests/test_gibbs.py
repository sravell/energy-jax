import unittest
import jax
import jax.numpy as jnp
from fac_graphs import GibbsSampler
from fac_graphs.inference import gibbs_sampling
from energy_jax.ebms import ebm
from energy_jax.sampling import discrete, sampler
import equinox as eqx
from typing import Any


class debm(ebm.AbstractEBM):
    prob: jnp.ndarray

    def __init__(self, x):
        self.prob = x

    def energy_function(self, x):
        return jnp.squeeze(-1 * jnp.sum(self.prob * x))

    def param_count(self):
        return 0


class wrapper(sampler.AbstractSampler):
    _inner: sampler.AbstractDiscreteSampler
    xshape: Any = None
    _inner: Any = None
    num_chains: Any = None

    def run_chain(self, model, state, key, **kwargs):
        return jax.tree_util.tree_map(
            lambda x: x.squeeze(axis=0),
            self._inner.run_chain(model, state, key, **kwargs),
        )

    def step():
        pass

    def _random_initial_state():
        pass


class TestGibbs(unittest.TestCase):
    def test_full(self):  # TODO: split this up into individual test
        var1 = fac_graphs.VariableNode("var1")
        var2 = fac_graphs.VariableNode("var2")
        var3 = fac_graphs.VariableNode("var3")
        ebm1 = debm(jnp.array([-100, 900.0, -100.0, -100.0, 900.0]))
        ebm2 = debm(jnp.array([-100, 900.0, -900.0, 900.0, -900.0]))
        factor1 = fac_graphs.FactorNode([var1, var2], ebm1, "factor1")
        factor2 = fac_graphs.FactorNode([var1, var3], ebm2, "factor2")
        graph = fac_graphs.FactorGraph([factor1, factor2])

        key = jax.random.PRNGKey(42)

        samp2 = wrapper(discrete.DiscreteUniformMH((2,), 100, 1, 2))
        samp3 = wrapper(discrete.DiscreteUniformMH((3,), 100, 1, 2))

        def _init(evidence: bool = False, traversal_order: list = None):
            return (
                {
                    var1: jnp.zeros(2).astype("int32"),
                    var2: jnp.zeros(3).astype("int32"),
                    var3: jnp.zeros(3).astype("int32"),
                },
                {var1: samp2, var2: samp3, var3: samp3},
                {var1: evidence, var2: evidence, var3: evidence},
                traversal_order,
            )

        state1, sampl1, ev1, trav1 = _init()
        state2, sampl2, ev2, trav2 = _init()

        g1 = GibbsSampler(sampl1, None, None, ev1, trav1)
        g2 = GibbsSampler(sampl2, None, None, ev2, trav2)

        # Jit compatibility tests
        self.assertTrue(
            jnp.allclose(
                g1.sample_single_variable(graph, var1, state1, key),
                eqx.filter_jit(g2.sample_single_variable)(graph, var1, state2, key),
            )
        )

        state1, sampl1, ev1, trav1 = _init()
        state2, sampl2, ev2, trav2 = _init()

        g1 = GibbsSampler(sampl1, None, None, ev1, trav1)
        g2 = GibbsSampler(sampl2, None, None, ev2, trav2)

        vars1sorted = sorted(k for k in state1.keys() if not ev1[k])
        vars2sorted = sorted(k for k in state2.keys() if not ev2[k])

        step1 = g1.step(graph, state1, vars1sorted, key)
        step2 = eqx.filter_jit(g2.step)(graph, state2, vars2sorted, key)

        for i in step1:
            self.assertTrue(jnp.allclose(step1[i], step2[i]))

        state1, sampl1, ev1, trav1 = _init()
        state2, sampl2, ev2, trav2 = _init()

        g1 = GibbsSampler(sampl1, None, 3, ev1, trav1)
        g2 = GibbsSampler(sampl2, None, 3, ev2, trav2)

        jit_gibbs = eqx.filter_jit(g2.run_chain)

        step1 = g1.run_chain(graph, state1, key)
        step2 = jit_gibbs(graph, state2, key)

        for i in step1:
            self.assertTrue(jnp.allclose(step1[i], step2[i]))

        # Correctness tests

        state1, sampl1, ev1, trav1 = _init()

        g1 = GibbsSampler(sampl1, None, 20, ev1, trav1)

        base1 = jnp.array([0.0, 1.0])
        base2 = jnp.array([0.0, 0.0, 1.0])
        base3 = jnp.array([0.0, 1.0, 0.0])

        results = jit_gibbs(graph, state1, key)

        self.assertTrue(jnp.allclose(base1, results[var1]))
        self.assertTrue(jnp.allclose(base2, results[var2]))
        self.assertTrue(jnp.allclose(base3, results[var3]))

        # Checking evidence True does not change sampling

        state1, sampl1, ev1, trav1 = _init(True)
        g1 = GibbsSampler(sampl1, None, 20, ev1, trav1)
        step1_evidence = g1.run_chain(graph, state1, key)

        self.assertTrue(jnp.allclose(jnp.zeros(2), step1_evidence[var1]))
        self.assertTrue(jnp.allclose(jnp.zeros(3), step1_evidence[var2]))
        self.assertTrue(jnp.allclose(jnp.zeros(3), step1_evidence[var3]))

        # Correctness tests with var1 evidence
        state1, sampl1, ev1, trav1 = _init()
        ev1[var1] = True
        g1 = GibbsSampler(sampl1, None, 20, ev1, trav1)

        step1 = g1.run_chain(graph, state1, key)
        base1evidence = jnp.array([0.0, 0.0])

        results = eqx.filter_jit(g1.run_chain)(graph, state1, key)

        self.assertTrue(jnp.allclose(base1evidence, step1[var1]))  # var1 is unchanged
        self.assertTrue(jnp.allclose(base1evidence, results[var1]))
        self.assertTrue(jnp.allclose(base2, results[var2]))
        self.assertTrue(jnp.allclose(base3, results[var3]))

        state1, sampl1, ev1, trav1 = _init()
        ev1[var1] = True
        g1 = GibbsSampler(sampl1, None, 20, None, trav1)

        step1 = g1.run_chain(graph, state1, key, ev1)
        base1evidence = jnp.array([0.0, 0.0])

        results = eqx.filter_jit(g1.run_chain)(graph, state1, key, ev1)

        self.assertTrue(jnp.allclose(base1evidence, step1[var1]))  # var1 is unchanged
        self.assertTrue(jnp.allclose(base1evidence, results[var1]))
        self.assertTrue(jnp.allclose(base2, results[var2]))
        self.assertTrue(jnp.allclose(base3, results[var3]))

        # Correctness tests with var1 evidence and traversal order
        state1, sampl1, ev1, trav1 = _init(False, [var2, var1, var3])
        ev1[var1] = True
        g1 = GibbsSampler(sampl1, None, 20, ev1, trav1)

        results = eqx.filter_jit(g1.run_chain)(graph, state1, key)

        self.assertTrue(jnp.allclose(base1evidence, results[var1]))
        self.assertTrue(jnp.allclose(base2, results[var2]))
        self.assertTrue(jnp.allclose(base3, results[var3]))
