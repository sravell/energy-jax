"""RBMs testing."""

from absl.testing import absltest
from jax import numpy as jnp
import jax
import equinox as eqx
from energax.ebms import rbms
from energax.sampling.discrete import CRBMGibbsSampler


class CRBMTests(absltest.TestCase):
    def setUp(self):
        self.max_dim = 3
        self.structure = jnp.array([3, 2, 3, 2])
        self.total_dim = int(jnp.prod(self.structure))
        self.hid = 5
        self.vis = 4
        self.ones = {
            "W": jnp.ones((self.max_dim, self.hid, self.vis)),
            "b": jnp.ones((self.max_dim, self.vis)),
            "c": jnp.ones((self.hid,)),
        }
        self.zeros = {
            "W": jnp.zeros((self.max_dim, self.hid, self.vis)),
            "b": jnp.zeros((self.max_dim, self.vis)),
            "c": jnp.zeros((self.hid,)),
        }
        self.hid_node = jnp.ones((self.hid,))
        self.vis_node = jax.nn.one_hot(jnp.zeros((self.vis,)), self.max_dim)

    def test_eq_known(self):
        """
        Test energy compared to know solution.

        Keep in mind, it implicitly maps [0, 0, 1] -> [-1, -1, 1]. So
        actual input is [[1, -1, -1], [1, -1, -1], [1, -1, -1], [1, -1, -1]] (4, 3)

        E = -1 * ((-4 * 5) + (-4) + (5)) = 19
        """
        crbm = rbms.CategoricalRBM(
            self.vis, self.hid, theta=self.ones, structure=self.structure
        )
        self.assertEqual(crbm.param_count(), 77)
        e_fn = crbm.energy_function((self.vis_node, self.hid_node))
        self.assertAlmostEqual(e_fn, 19)

        jit_e = eqx.filter_jit(crbm.energy_function)
        e_fn = jit_e((self.vis_node, self.hid_node))
        self.assertAlmostEqual(e_fn, 19)

    def test_prob(self):
        crbm = rbms.CategoricalRBM(
            self.vis,
            self.hid,
            theta=self.zeros,
            structure=self.structure,
            generate_bitstrings=True,
        )
        probs = crbm.probability_vector()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)
        self.assertTrue(jnp.allclose(probs, 1 / len(probs)))

        jit_p = eqx.filter_jit(crbm.probability_vector)
        probs = jit_p()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)
        self.assertTrue(jnp.allclose(probs, 1 / len(probs)))

        crbm = rbms.CategoricalRBM(
            self.vis,
            self.hid,
            theta=self.ones,
            structure=self.structure,
            generate_bitstrings=True,
        )
        probs = crbm.probability_vector()
        self.assertAlmostEqual(1.0, jnp.sum(probs), delta=1e-5)

    def test_exp(self):

        def fn(x):
            return jnp.array(0.5)

        crbm = rbms.CategoricalRBM(
            self.vis,
            self.hid,
            theta=self.zeros,
            structure=self.structure,
            generate_bitstrings=True,
        )
        exp_val = crbm.expectation_value(fn, jnp.array([1] * self.total_dim))
        self.assertAlmostEqual(exp_val, 0.5, delta=1e-5)

        jit_e = eqx.filter_jit(crbm.expectation_value)
        exp_val = jit_e(fn, jnp.array([[1] * 3] * self.total_dim))
        self.assertAlmostEqual(exp_val, 0.5, delta=1e-5)

        def fn(x):
            return len(x)

        exp_val = jit_e(fn, jnp.array([[1] * 3] * self.total_dim))
        self.assertAlmostEqual(exp_val, 3.0, delta=1e-5)

    def test_sample(self):
        key = jax.random.PRNGKey(42)
        params = rbms.get_random_crbm_params(key, self.vis, self.hid, self.max_dim)
        params["b"] = (
            params["b"].at[0].multiply(100)
        )  # create a more skewed distribution
        crbm = rbms.CategoricalRBM(
            self.vis,
            self.hid,
            theta=params,
            structure=self.structure,
            generate_bitstrings=True,
        )
        max_ = crbm.bitstrings[jnp.argmax(crbm.probability_vector())]

        def str_mode(arr):
            """
            Gets the most common value (mode) and returns it as a string.
            """
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

        sampler = CRBMGibbsSampler(None, 1, 2000, 1)

        key, subkey = jax.random.split(key)
        x = sampler.sample_chains(
            crbm,
            jnp.expand_dims(
                jax.nn.one_hot(jnp.array([0] * self.vis), self.max_dim), axis=0
            ),
            subkey,
        )["position"]["v"]
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        print(x.shape, max_, str_mode(x))
        self.assertTrue(str(max_) == str_mode(x))
        # this checks to make sure that the sampler's state mode is actually the most probably state

        key, subkey = jax.random.split(key)
        x = sampler.sample_chains(
            crbm,
            None,
            subkey,
        )["position"][
            "v"
        ]  # test random initialization
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        print(x.shape, max_, str_mode(x))
        self.assertTrue(str(max_) == str_mode(x))

        # Test multiple chains
        sampler = CRBMGibbsSampler(None, 10, 500, 2)
        key, subkey = jax.random.split(key)
        x = sampler.sample_chains(
            crbm,
            jnp.array(
                [
                    jax.nn.one_hot(jnp.array([1] * self.vis), self.max_dim),
                    jax.nn.one_hot(jnp.array([0] * self.vis), self.max_dim),
                ]
            ),
            subkey,
        )["position"]["v"]
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        self.assertTrue(str(max_) == str_mode(x))


if __name__ == "__main__":
    absltest.main()
