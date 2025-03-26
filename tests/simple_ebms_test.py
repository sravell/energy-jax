"""Unit testing."""

from absl.testing import absltest
from jax import numpy as jnp
import jax
import networkx as nx
import equinox as eqx
from energax.ebms import simple_ebms
from energax.sampling.discrete import DiscreteUniformMH, DiscreteHammingMH
from energax.sampling.continuous import (
    LangevinSampler,
    HMCSampler,
    ErrorAdaptiveLangevinSampler,
    OverdampedLangevinSDESampler,
)
from diffrax import Heun

_EPSILON = 1e-4


class BoltzEBMTests(absltest.TestCase):
    def setUp(self):
        self.nodes, self.edges = [0, 1, 2, 3], [(0, 1), (1, 2), (2, 0), (0, 3)]
        self.ones = {
            "nodes": jnp.array([1.0] * len(self.nodes)),
            "edges": jnp.ones((len(self.nodes), len(self.nodes))),
        }
        self.spins = jnp.array([1.0] * len(self.nodes))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)
        self.zeros = {
            "nodes": jnp.array([0.0] * len(self.nodes)),
            "edges": jnp.zeros((len(self.nodes), len(self.nodes))),
        }

        self.nodest, self.edgest = [0, 1, 2], [(0, 1), (1, 2), (2, 0)]
        self.onest = {
            "nodes": jnp.array([1.0] * len(self.nodest)),
            "edges": jnp.ones((len(self.nodest), len(self.nodest))),
        }
        self.spinst = jnp.array([1.0] * len(self.nodest))
        self.tri_graph = nx.Graph()
        self.tri_graph.add_nodes_from(self.nodest)
        self.tri_graph.add_edges_from(self.edgest)

        self.zerost = {
            "nodes": jnp.array([0.0] * len(self.nodest)),
            "edges": jnp.zeros((len(self.nodest), len(self.nodest))),
        }

    def test_eq_known(self):
        """
        Test energy compared to know solution.

        E = (self.edges + self.nodes)

        self.nodes = 4 * (1 * 1) = 4
        self.edges = 4 * (1 * 1 * 1) = 4

        E = (4 + 4) = 8
        """
        boltz_ebm = simple_ebms.BoltzmannEBM(self.graph, theta=self.ones)
        e_fn = boltz_ebm.energy_function(self.spins)
        self.assertEqual(boltz_ebm.param_count(), 20)
        self.assertAlmostEqual(e_fn, 8)

        jit_e = eqx.filter_jit(boltz_ebm.energy_function)
        e_fn = jit_e(self.spins)
        self.assertAlmostEqual(e_fn, 8)

        """
        E = (self.edges + self.nodes)

        self.nodes = 3 * (1 * 1) = 3
        self.edges = 3 * (1 * 1 * 1) = 3

        E = (3 + 3) = 6
        """

        boltz_ebm = simple_ebms.BoltzmannEBM(self.tri_graph, theta=self.onest)
        jit_e = eqx.filter_jit(boltz_ebm.energy_function)
        e_fn = jit_e(self.spinst)
        self.assertAlmostEqual(e_fn, 6)

    def test_prob(self):
        boltz_ebm = simple_ebms.BoltzmannEBM(
            self.tri_graph, theta=self.zerost, generate_bitstrings=True
        )
        probs = boltz_ebm.probability_vector()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)
        self.assertTrue(jnp.allclose(probs, 1 / 2 ** len(self.tri_graph.nodes)))

        boltz_ebm = simple_ebms.BoltzmannEBM(
            self.tri_graph, theta=self.onest, generate_bitstrings=True
        )
        probs = boltz_ebm.probability_vector()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)

        boltz_ebm = simple_ebms.BoltzmannEBM(
            self.tri_graph, theta=self.zerost, generate_bitstrings=True
        )
        jit_p = eqx.filter_jit(boltz_ebm.probability_vector)
        probs = jit_p()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)
        self.assertTrue(jnp.allclose(probs, 1 / 2 ** len(self.tri_graph.nodes)))

        boltz_ebm = simple_ebms.BoltzmannEBM(
            self.graph, theta=self.zeros, generate_bitstrings=True
        )
        jit_p = eqx.filter_jit(boltz_ebm.probability_vector)
        probs = jit_p()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)
        self.assertTrue(jnp.allclose(probs, 1 / 2 ** len(self.graph.nodes)))

    def test_exp(self):
        def fn(x):
            return jnp.array(0.5)

        boltz_ebm = simple_ebms.BoltzmannEBM(
            self.tri_graph, theta=self.zerost, generate_bitstrings=True
        )
        exp_val = boltz_ebm.expectation_value(fn)
        self.assertAlmostEqual(exp_val, 0.5)

        jit_e = eqx.filter_jit(boltz_ebm.expectation_value)
        exp_val = jit_e(fn)
        self.assertAlmostEqual(exp_val, 0.5)

        def fn(x):
            return len(x)

        exp_val = jit_e(fn)
        self.assertAlmostEqual(exp_val, 3.0)

        boltz_ebm = simple_ebms.BoltzmannEBM(
            self.graph, theta=self.zeros, generate_bitstrings=True
        )
        jit_e = eqx.filter_jit(boltz_ebm.expectation_value)
        exp_val = jit_e(fn)
        self.assertAlmostEqual(exp_val, 4.0)

    def test_sample(self):
        params = self.onest.copy()  # skew distribution away from uniform
        params["edges"] = params["edges"].at[0].set(0.9)
        params["nodes"] = params["nodes"].at[0].set(0.09)
        params["nodes"] = params["nodes"].at[1].set(10.0)
        boltz_ebm = simple_ebms.BoltzmannEBM(
            self.tri_graph, theta=params, generate_bitstrings=True
        )
        max_ = boltz_ebm.bitstrings[jnp.argmax(boltz_ebm.probability_vector())]

        def str_mode(arr):
            vals = {}
            for i in arr:
                j = str(i.astype(max_.dtype))
                if j in vals:
                    vals[j] += 1
                else:
                    vals[j] = 0
            max_val = -jnp.inf
            max_ind = None
            for i in vals:
                if max_val < vals[i]:
                    max_val = vals[i]
                    max_ind = i
            return max_ind

        key = jax.random.PRNGKey(42)
        sampler = DiscreteUniformMH((3,), 10, 1000, 1)
        x = sampler.run_chain(boltz_ebm, None, key)["position"]
        self.assertTrue(str(max_) == str_mode(x))


class PottsEBMTest(absltest.TestCase):
    def setUp(self):
        self.nodes, self.edges = [0, 1, 2, 3], [(0, 1), (1, 2), (2, 0), (0, 3)]
        self.ones = {
            "nodes": jnp.array([1.0] * len(self.nodes)),
            "edges": jnp.ones((len(self.nodes), len(self.nodes))),
        }
        self.spins = jnp.array([1.0] * len(self.nodes))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)
        self.zeros = {
            "nodes": jnp.array([0.0] * len(self.nodes)),
            "edges": jnp.zeros((len(self.nodes), len(self.nodes))),
        }

        self.nodest, self.edgest = [0, 1, 2], [(0, 1), (1, 2), (2, 0)]
        self.onest = {
            "nodes": jnp.array([1.0] * len(self.nodest)),
            "edges": jnp.ones((len(self.nodest), len(self.nodest))),
        }
        self.spinst = jnp.array([1.0] * len(self.nodest))
        self.tri_graph = nx.Graph()
        self.tri_graph.add_nodes_from(self.nodest)
        self.tri_graph.add_edges_from(self.edgest)

        self.zerost = {
            "nodes": jnp.array([0.0] * len(self.nodest)),
            "edges": jnp.zeros((len(self.nodest), len(self.nodest))),
        }

    def test_eq_known(self):
        """
        Test energy compared to know solution.

        E = (self.edges + self.nodes)

        self.nodes = 4 * (1 * 1) = 4
        self.edges = 4 * (1 * 1 * 1) = 4

        E = (4 + 4) = 8
        """
        potts_ebm = simple_ebms.PottsEBM(self.graph, theta=self.ones)
        self.assertEqual(potts_ebm.param_count(), 20)
        e_fn = potts_ebm.energy_function(self.spins)
        self.assertAlmostEqual(e_fn, 8)

        jit_e = eqx.filter_jit(potts_ebm.energy_function)
        e_fn = jit_e(self.spins)
        self.assertAlmostEqual(e_fn, 8)

        """
        E = (self.edges + self.nodes)

        self.nodes = 3 * (1 * 1) = 3
        self.edges = 3 * 1 * delta(1, 1) = 3

        E = (3 + 3) = 6
        """
        potts_ebm = simple_ebms.PottsEBM(
            self.tri_graph, theta=self.onest, e_fn_type="kron"
        )
        jit_e = eqx.filter_jit(potts_ebm.energy_function)
        e_fn = jit_e(self.spinst)
        self.assertAlmostEqual(e_fn, 6)

    def test_prob(self):
        potts_ebm = simple_ebms.PottsEBM(
            self.tri_graph, theta=self.zerost, generate_bitstrings=True
        )
        probs = potts_ebm.probability_vector()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)
        self.assertTrue(jnp.allclose(probs, 1 / 2 ** len(self.tri_graph.nodes)))

        potts_ebm = simple_ebms.PottsEBM(
            self.tri_graph, theta=self.onest, generate_bitstrings=True
        )
        probs = potts_ebm.probability_vector()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)

        potts_ebm = simple_ebms.PottsEBM(
            self.tri_graph, theta=self.zerost, generate_bitstrings=True
        )
        jit_p = eqx.filter_jit(potts_ebm.probability_vector)
        probs = jit_p()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)
        self.assertTrue(jnp.allclose(probs, 1 / 2 ** len(self.tri_graph.nodes)))

        potts_ebm = simple_ebms.PottsEBM(
            self.graph, theta=self.zeros, generate_bitstrings=True
        )
        jit_p = eqx.filter_jit(potts_ebm.probability_vector)
        probs = jit_p()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)
        self.assertTrue(jnp.allclose(probs, 1 / 2 ** len(self.graph.nodes)))

    def test_exp(self):
        def fn(x):
            return 0.5

        potts_ebm = simple_ebms.PottsEBM(
            self.tri_graph, theta=self.zerost, generate_bitstrings=True
        )
        exp_val = potts_ebm.expectation_value(fn)
        self.assertAlmostEqual(exp_val, 0.5)

        jit_e = eqx.filter_jit(potts_ebm.expectation_value)
        exp_val = jit_e(fn)
        self.assertAlmostEqual(exp_val, 0.5)

        def fn(x):
            return len(x)

        exp_val = jit_e(fn)
        self.assertAlmostEqual(exp_val, 3.0)

        potts_ebm = simple_ebms.PottsEBM(
            self.graph, theta=self.zeros, generate_bitstrings=True
        )
        jit_e = eqx.filter_jit(potts_ebm.expectation_value)
        exp_val = jit_e(fn)
        self.assertAlmostEqual(exp_val, 4.0)

    def test_sample(self):
        params = self.onest.copy()  # skew distribution away from uniform
        params["edges"] = params["edges"].at[0].set(0.9)
        params["nodes"] = params["nodes"].at[0].set(0.09)
        params["nodes"] = params["nodes"].at[1].set(10.0)
        potts_ebm = simple_ebms.PottsEBM(
            self.tri_graph, theta=params, generate_bitstrings=True
        )
        max_ = potts_ebm.bitstrings[jnp.argmax(potts_ebm.probability_vector())]

        def str_mode(arr):
            vals = {}
            for i in arr:
                j = str(i.astype(max_.dtype))
                if j in vals:
                    vals[j] += 1
                else:
                    vals[j] = 0
            max_val = -jnp.inf
            max_ind = None
            for i in vals:
                if max_val < vals[i]:
                    max_val = vals[i]
                    max_ind = i
            return max_ind

        key = jax.random.PRNGKey(42)
        sampler = DiscreteUniformMH((3,), 10, 500, 2)
        x = sampler.sample_chains(potts_ebm, None, key)["position"]
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        self.assertTrue(str(max_) == str_mode(x))


class ContinuousBoltzEBMTests(absltest.TestCase):
    def setUp(self):
        self.nodes, self.edges = [0, 1, 2, 3], [(0, 1), (1, 2), (2, 0), (0, 3)]
        self.ones = {
            "nodes": jnp.array([1.0] * len(self.nodes)),
            "edges": jnp.ones((len(self.nodes), len(self.nodes))),
        }
        self.spins = jnp.array([1.0] * len(self.nodes))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)
        self.zeros = {
            "nodes": jnp.array([0.0] * len(self.nodes)),
            "edges": jnp.zeros((len(self.nodes), len(self.nodes))),
        }

        self.nodest, self.edgest = [0, 1, 2], [(0, 1), (1, 2), (2, 0)]
        self.onest = {
            "nodes": jnp.array([1.0] * len(self.nodest)),
            "edges": jnp.ones((len(self.nodest), len(self.nodest))),
        }
        self.spinst = jnp.array([1.0] * len(self.nodest))
        self.tri_graph = nx.Graph()
        self.tri_graph.add_nodes_from(self.nodest)
        self.tri_graph.add_edges_from(self.edgest)

        self.zerost = {
            "nodes": jnp.array([0.0] * len(self.nodest)),
            "edges": jnp.zeros((len(self.nodest), len(self.nodest))),
        }

    def test_eq_known(self):
        """
        Test energy compared to know solution.

        E = (self.edges + self.nodes)

        self.nodes = 4 * (1 * 1) = 4
        self.edges = 4 * (1 * 1 * 1) = 4

        E = (4 + 4) = 8
        """
        boltz_ebm = simple_ebms.ContinuousBoltzmannEBM(self.graph, self.ones)
        self.assertEqual(boltz_ebm.param_count(), 20)
        e_fn = boltz_ebm.energy_function(1e10 * self.spins)  # big because tanh
        self.assertAlmostEqual(e_fn, 8)

        """
        E = (self.edges + self.nodes)

        self.nodes = 3 * (1 * 1) = 3
        self.edges = 3 * (1 * 1 * 1) = 3

        E = (3 + 3) = 6
        """

        boltz_ebm = simple_ebms.ContinuousBoltzmannEBM(self.tri_graph, self.onest)
        e_fn = boltz_ebm.energy_function(1e10 * self.spinst)
        self.assertAlmostEqual(e_fn, 6)

    def test_sample_langevin(self):
        boltz_ebm = simple_ebms.ContinuousBoltzmannEBM(self.tri_graph, self.onest)
        sampler = LangevinSampler((3,), 1, -1.0, 1.0, 1.0, 0.01, 100)
        sample = sampler.sample_chains(boltz_ebm, None, jax.random.PRNGKey(42))[
            "position"
        ][0]
        energy = boltz_ebm.energy_function(sample)
        true_min_energy = -2  # ideal state is 2 -1 and one 1,
        # so we have self.edges of (-1, -1, 1) + self.nodes (-1, -1, 1)
        print(energy, sample)
        mse = (energy - true_min_energy) ** 2
        self.assertTrue(mse <= _EPSILON)

        boltz_ebm = simple_ebms.ContinuousBoltzmannEBM(self.tri_graph, self.onest)
        sampler = LangevinSampler(
            (3,), 1, -1.0, 1.0, 0.1, 0.01, 800, metropolis_adjustment=True
        )
        sample = sampler.sample_chains(boltz_ebm, None, jax.random.PRNGKey(42))[
            "position"
        ][0]
        energy = boltz_ebm.energy_function(sample)
        true_min_energy = -2  # ideal state is 2 -1 and one 1,
        # so we have self.edges of (-1, -1, 1) + self.nodes (-1, -1, 1)
        print(energy, sample)
        mse = (energy - true_min_energy) ** 2
        self.assertTrue(mse <= _EPSILON)

        boltz_ebm = simple_ebms.ContinuousBoltzmannEBM(self.tri_graph, self.onest)
        sampler = ErrorAdaptiveLangevinSampler(
            (3,), 1, -1.0, 1.0, 0.1, jnp.sqrt(0.1 * 2), 800, 1e-1, 1e-3, 1e-4, 1e-4
        )
        sample = sampler.sample_chains(boltz_ebm, None, jax.random.PRNGKey(42))[
            "position"
        ][0]
        energy = boltz_ebm.energy_function(sample)
        true_min_energy = -2  # ideal state is 2 -1 and one 1,
        # so we have self.edges of (-1, -1, 1) + self.nodes (-1, -1, 1)
        print(energy, sample)
        mse = (energy - true_min_energy) ** 2
        self.assertTrue(mse <= _EPSILON)

    def test_sample_hmc(self):
        boltz_ebm = simple_ebms.ContinuousBoltzmannEBM(self.tri_graph, self.onest)
        sampler = HMCSampler(
            (3,), 1, -1.0, 1.0, 100, 0.5, 3, jnp.ones_like(self.onest["nodes"])
        )
        sample = sampler.sample_chains(boltz_ebm, None, jax.random.PRNGKey(42))[
            "position"
        ][0]
        energy = boltz_ebm.energy_function(sample)
        true_min_energy = -2  # ideal state is 2 -1 and one 1,
        # so we have self.edges of (-1, -1, 1) + self.nodes (-1, -1, 1)
        print(energy, sample)
        mse = (energy - true_min_energy) ** 2
        self.assertTrue(mse <= _EPSILON)

    def test_sample_sde(self):
        boltz_ebm = simple_ebms.ContinuousBoltzmannEBM(self.tri_graph, self.onest)
        sampler = OverdampedLangevinSDESampler(
            (3,), 1, -1.0, 1.0, 1.0, 0.01, 10, 2_000, Heun(), 0.0, 100.0, 150.0, (3,)
        )
        sample = jnp.mean(
            sampler.sample_chains(boltz_ebm, None, jax.random.PRNGKey(42))["position"][
                0
            ],
            axis=0,
        )
        energy = boltz_ebm.energy_function(sample)
        true_min_energy = -2  # ideal state is 2 -1 and one 1,
        # so we have self.edges of (-1, -1, 1) + self.nodes (-1, -1, 1)
        print(energy, sample)
        mse = (energy - true_min_energy) ** 2
        self.assertTrue(mse <= _EPSILON)


if __name__ == "__main__":
    absltest.main()
