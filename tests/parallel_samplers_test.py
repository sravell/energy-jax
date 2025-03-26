"""Test distributed version of samplers written in pmap."""

from absl.testing import absltest
import chex

# This function should be called before any other jax functions.
# It makes a fake environment with 4 host CPU devices, and
# we can test pmap() on a single CPU device.
_NUM_CPU_DEVICES = 4
chex.set_n_cpu_devices(_NUM_CPU_DEVICES)

import jax
from jax import numpy as jnp
import equinox as eqx
from energax.sampling import (
    LangevinSampler,
    update_buffer,
    ContinuousReplayBuffer,
)


class Constant(eqx.Module):
    val: jnp.ndarray

    def __init__(self, val):
        self.val = jnp.array(val)

    def __call__(self, x):
        return jnp.squeeze(jnp.mean(x * self.val))

    def energy_function(self, x, keyword=None):
        return self.__call__(x)


class PmappedSamplerTests(absltest.TestCase):
    def test_device_env(self):
        if jax.devices()[0].platform != "cpu":
            self.skipTest("this test is only for multi-core CPUs")
        chex.assert_devices_available(_NUM_CPU_DEVICES, "cpu", backend="cpu")
        detected_devices = jax.device_count()
        self.assertEqual(detected_devices, _NUM_CPU_DEVICES)

    def test_langevin_sample_chains_parallel(self):
        if jax.devices()[0].platform != "cpu":
            self.skipTest("this test is only for multi-core CPUs")
        batch_size = 23
        sampler = LangevinSampler((10, 2), batch_size, -1.0, 1.0, 10.0, 0.005, 20)
        samples = sampler.sample_chains_parallel(
            Constant(1.0), None, jax.random.PRNGKey(0)
        )["position"]
        self.assertTrue(samples.shape == (23, 10, 2))


class PmappedSamplersReplayTests(absltest.TestCase):
    def test_buffer_shapes_continuous(self):
        if jax.devices()[0].platform != "cpu":
            self.skipTest("this test is only for multi-core CPUs")
        key = jax.random.PRNGKey(42)
        batch_size = 23
        sampler = LangevinSampler((10, 2), batch_size, -1.0, 1.0, 10.0, 0.005, 20)

        buffer = ContinuousReplayBuffer((100, 10, 2), 0.0, 1.0, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = sampler.sample_chains_parallel(
            Constant(1.0), samples, jax.random.PRNGKey(0)
        )["position"]
        self.assertTrue(samples.shape == (23, 10, 2))
        new_buffer = update_buffer(buffer, jnp.zeros(shape=(2, 10, 2)))
        self.assertIsInstance(new_buffer, ContinuousReplayBuffer)

    def test_metropolis_adjustment(self):
        if jax.devices()[0].platform != "cpu":
            self.skipTest("this test is only for multi-core CPUs")
        key = jax.random.PRNGKey(42)
        batch_size = 23
        sampler = LangevinSampler(
            (10, 2), batch_size, -1.0, 1.0, 10.0, 0.005, 20, metropolis_adjustment=True
        )

        buffer = ContinuousReplayBuffer((100, 10, 2), 0.0, 1.0, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = sampler.sample_chains_parallel(
            Constant(1.0), samples, jax.random.PRNGKey(0)
        )["position"]
        self.assertTrue(samples.shape == (23, 10, 2))

    def test_kwargs_continuous_keyword(self):
        if jax.devices()[0].platform != "cpu":
            self.skipTest("this test is only for multi-core CPUs")
        key = jax.random.PRNGKey(42)
        batch_size = 23
        sampler = LangevinSampler((10, 2), batch_size, -1.0, 1.0, 10.0, 0.005, 20)

        buffer = ContinuousReplayBuffer((100, 10, 2), 0.0, 1.0, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = sampler.sample_chains_parallel(
            Constant(1.0), samples, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples.shape == (23, 10, 2))

    def test_kwargs_continuous_keyword_and_metropolis_adjustment(self):
        if jax.devices()[0].platform != "cpu":
            self.skipTest("this test is only for multi-core CPUs")
        key = jax.random.PRNGKey(42)
        batch_size = 23
        sampler = LangevinSampler(
            (10, 2), batch_size, -1.0, 1.0, 10.0, 0.005, 20, metropolis_adjustment=True
        )

        buffer = ContinuousReplayBuffer((100, 10, 2), 0.0, 1.0, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = sampler.sample_chains_parallel(
            Constant(1.0), samples, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples.shape == (23, 10, 2))


if __name__ == "__main__":
    absltest.main()
