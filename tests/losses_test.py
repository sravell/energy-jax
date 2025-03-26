"""Unit testing."""

from absl.testing import absltest
from jax import numpy as jnp
import jax
from energax import losses
from energax.ebms.nn_ebms import ContinuousNNEBM
from energax import nns
import equinox as eqx


class CustomScore(ContinuousNNEBM):
    def score(self, x, sigma=None, **kwargs):
        return -1 * eqx.filter_grad(self.energy_function)(x, **kwargs)

    def __call__(self, x):
        return self.energy_function(x)


class LossesTest(absltest.TestCase):
    def setUp(self):
        self.ebm = CustomScore(
            nns.MLP(dims=10, depth=4, width=8, key=jax.random.PRNGKey(42))
        )

    def test_cd(self):
        fn = eqx.filter_value_and_grad(losses.contrastive_divergence)
        val, grad = fn(self.ebm, jnp.ones((2, 10)), jnp.zeros((2, 10)))
        self.assertTrue(val.shape == ())
        grads, _ = jax.flatten_util.ravel_pytree(grad)
        self.assertAlmostEqual(jnp.sum(jnp.isnan(grads).astype("int8")), 0)

    def test_softplus(self):
        fn = eqx.filter_value_and_grad(losses.softplus_contrastive_divergence)
        val, grad = fn(self.ebm, jnp.ones((2, 10)), jnp.zeros((2, 10)))
        self.assertTrue(val.shape == ())
        grads, _ = jax.flatten_util.ravel_pytree(grad)
        self.assertAlmostEqual(jnp.sum(jnp.isnan(grads).astype("int8")), 0)

    def test_logsumexp(self):
        fn = eqx.filter_value_and_grad(losses.logsumexp_contrastive_divergence)
        val, grad = fn(self.ebm, jnp.ones((2, 10)), jnp.zeros((2, 10)))
        self.assertTrue(val.shape == ())
        grads, _ = jax.flatten_util.ravel_pytree(grad)
        self.assertAlmostEqual(jnp.sum(jnp.isnan(grads).astype("int8")), 0)

    def test_ssm(self):
        fn = eqx.filter_value_and_grad(losses.ssm)
        key = jax.random.PRNGKey(0)
        val, grad = fn(self.ebm, jnp.ones((2, 10)), key)
        self.assertTrue(val.shape == ())
        grads, _ = jax.flatten_util.ravel_pytree(grad)
        self.assertAlmostEqual(jnp.sum(jnp.isnan(grads).astype("int8")), 0)

    def test_ssm_vr(self):
        fn = eqx.filter_value_and_grad(losses.ssm_vr)
        key = jax.random.PRNGKey(0)
        val, grad = fn(self.ebm, jnp.ones((2, 10)), key)
        self.assertTrue(val.shape == ())
        grads, _ = jax.flatten_util.ravel_pytree(grad)
        self.assertAlmostEqual(jnp.sum(jnp.isnan(grads).astype("int8")), 0)

    def test_dsm(self):
        fn = eqx.filter_value_and_grad(losses.dsm)
        key = jax.random.PRNGKey(0)
        val, grad = fn(self.ebm, jnp.ones((2, 10)), key)
        self.assertTrue(val.shape == ())
        grads, _ = jax.flatten_util.ravel_pytree(grad)
        self.assertAlmostEqual(jnp.sum(jnp.isnan(grads).astype("int8")), 0)

    def test_anneal_dsm(self):
        fn = eqx.filter_value_and_grad(losses.anneal_dsm)
        key = jax.random.PRNGKey(0)
        val, grad = fn(self.ebm, jnp.ones((2, 10)), jnp.array([0.1, 0.05, 0.01]), key)
        self.assertTrue(val.shape == ())
        grads, _ = jax.flatten_util.ravel_pytree(grad)
        self.assertAlmostEqual(jnp.sum(jnp.isnan(grads).astype("int8")), 0)


if __name__ == "__main__":
    absltest.main()
