"""Most samplers are test implicitly with the EBMs, here is for the other stuff."""

from absl.testing import absltest
from jax import numpy as jnp
import jax
from energax.ebms import ebm
from energax.sampling import (
    LangevinSampler,
    HMCSampler,
    ContinuousReplayBuffer,
    DiscreteReplayBuffer,
    DiscreteUniformMH,
    update_buffer,
    GibbsWithGradient,
    AnnealedLangevinSampler,
    ErrorAdaptiveLangevinSampler,
    OverdampedLangevinSDESampler,
    UnderdampedLangevinSDESampler,
)
from diffrax import Heun, Ralston, PIDController
import equinox as eqx


class Constant(ebm.AbstractEBM):
    val: jnp.ndarray

    def __init__(self, val):
        super().__init__()
        self.val = jnp.array(val)

    def __call__(self, x):
        if isinstance(x, jnp.ndarray):
            return jnp.squeeze(jnp.mean(x**2 * self.val))
        else:
            return jnp.squeeze(jnp.mean(x["a"] ** 2 * self.val))

    def energy_function(self, x, keyword=None, **kwargs):
        return self.__call__(x)

    def param_count(self) -> int:
        raise NotImplementedError


class ConstantWithKwargs(Constant):
    def __call__(self, x, test_argument):
        x += test_argument
        return super().__call__(x)

    def energy_function(self, x, keyword=None, **kwargs):
        return self.__call__(x, **kwargs)


class SDETests(absltest.TestCase):
    def test_langevin_sde(self):
        batch_size = 23
        n_samples = 13
        sampler = OverdampedLangevinSDESampler(
            (10, 2),
            batch_size,
            -1.0,
            1.0,
            0.1,
            0.1,
            n_samples,
            1_000,
            Heun(),
            0.0,
            1.0,
            10.0,
            (10, 2),
        )
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), None, jax.random.PRNGKey(0)
        )["position"]
        self.assertTrue(samples.shape == (batch_size, n_samples, 10, 2))

    def test_shape_retained_after_sampling(self):
        batch_size = 23
        sampler = OverdampedLangevinSDESampler(
            (1, 10, 2),
            batch_size,
            -1.0,
            1.0,
            0.1,
            0.1,
            -1,  # Special handling in sampler to squeeze output when n_samples=-1
            1_000,
            Heun(),
            0.0,
            1.0,
            10.0,
            (1, 10, 2),
        )
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), None, jax.random.PRNGKey(0)
        )["position"]
        self.assertTrue(samples.shape == (batch_size, 1, 10, 2))

    def test_shape_retained_after_sampling_multiple_steps(self):
        batch_size = 23
        sampler = OverdampedLangevinSDESampler(
            (1, 10, 2),
            batch_size,
            -1.0,
            1.0,
            0.1,
            0.1,
            20,
            1_000,
            Heun(),
            0.0,
            1.0,
            10.0,
            (1, 10, 2),
        )
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), None, jax.random.PRNGKey(0)
        )["position"]
        self.assertTrue(samples.shape == (batch_size, 20, 1, 10, 2))

    def test_hamiltonian_and_langevin_sde_with_kwargs(self):
        # Tests that our sampler can pass in keyword args to the model.
        batch_size = 23
        n_samples = 13
        sampler_args = [
            (10, 2),
            batch_size,
            -1.0,
            1.0,
            0.1,
            0.1,
            n_samples,
            1_000,
            Heun(),
            0.0,
            1.0,
            10.0,
            (10, 2),
            0.1,
        ]
        underdamped_sampler = UnderdampedLangevinSDESampler(*sampler_args)
        overdamped_sampler = OverdampedLangevinSDESampler(*sampler_args)
        underdamped_samples = eqx.filter_jit(underdamped_sampler.sample_chains)(
            ConstantWithKwargs(1.0),
            None,
            jax.random.PRNGKey(0),
            keyword=jnp.ones(batch_size),
            test_argument=jnp.ones(batch_size),
        )["position"]
        overdamped_samples = eqx.filter_jit(overdamped_sampler.sample_chains)(
            ConstantWithKwargs(1.0),
            None,
            jax.random.PRNGKey(0),
            keyword=jnp.ones(batch_size),
            test_argument=jnp.ones(batch_size),
        )["position"]
        self.assertTrue(underdamped_samples.shape == (batch_size, n_samples, 10, 2))
        self.assertTrue(overdamped_samples.shape == (batch_size, n_samples, 10, 2))

    def test_langevin_sde_pytree(self):
        batch_size = 23
        n_samples = 13
        in_tree = {
            "a": jax.ShapeDtypeStruct((10, 2), "float32"),
            "b": jax.ShapeDtypeStruct((1, 2, 3, 4), "float32"),
        }
        in_tree_vals = {
            "a": jnp.ones((10, 2)).astype("float32"),
            "b": jnp.ones((1, 2, 3, 4)).astype("float32"),
        }
        sampler = OverdampedLangevinSDESampler(
            in_tree_vals,
            batch_size,
            -1.0,
            1.0,
            0.1,
            0.1,
            n_samples,
            1_000,
            Heun(),
            0.0,
            1.0,
            10.0,
            in_tree,
        )

        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), None, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples["a"].shape == (batch_size, n_samples, 10, 2))
        self.assertTrue(samples["b"].shape == (batch_size, n_samples, 1, 2, 3, 4))

        sampler = OverdampedLangevinSDESampler(
            in_tree_vals,
            batch_size,
            -1.0,
            1.0,
            0.1,
            0.1,
            n_samples,
            1_000,
            Ralston(),
            0.0,
            1.0,
            10.0,
            in_tree,
            adaptive_controller=PIDController(1e-3, 1e-6),
        )

        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), None, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples["a"].shape == (batch_size, n_samples, 10, 2))
        self.assertTrue(samples["b"].shape == (batch_size, n_samples, 1, 2, 3, 4))

    def test_hamiltonian_sde(self):
        batch_size = 23
        n_samples = 13
        sampler = UnderdampedLangevinSDESampler(
            (10, 2),
            batch_size,
            -1.0,
            1.0,
            0.1,
            0.1,
            n_samples,
            1_000,
            Heun(),
            0.0,
            1.0,
            10.0,
            (10, 2),
            0.1,
        )
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), None, jax.random.PRNGKey(0)
        )["position"]
        self.assertTrue(samples.shape == (batch_size, n_samples, 10, 2))

    def test_hamiltonian_sde_pytree(self):
        batch_size = 23
        n_samples = 13
        in_tree = {
            "a": jax.ShapeDtypeStruct((10, 2), "float32"),
            "b": jax.ShapeDtypeStruct((1, 2, 3, 4), "float32"),
        }
        in_tree_vals = {
            "a": jnp.ones((10, 2)).astype("float32"),
            "b": jnp.ones((1, 2, 3, 4)).astype("float32"),
        }
        sampler = UnderdampedLangevinSDESampler(
            in_tree_vals,
            batch_size,
            -1.0,
            1.0,
            0.1,
            0.1,
            n_samples,
            1_000,
            Heun(),
            0.0,
            1.0,
            10.0,
            in_tree,
            0.1,
        )
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), None, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples["a"].shape == (batch_size, n_samples, 10, 2))
        self.assertTrue(samples["b"].shape == (batch_size, n_samples, 1, 2, 3, 4))


class ReplayTests(absltest.TestCase):
    def test_buffer_shapes_continuous(self):
        key = jax.random.PRNGKey(42)
        batch_size = 23
        sampler = LangevinSampler((10, 2), batch_size, -1.0, 1.0, 10.0, 0.005, 20)

        buffer = ContinuousReplayBuffer((100, 10, 2), 0.0, 1.0, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), samples, jax.random.PRNGKey(0)
        )["position"]
        self.assertTrue(samples.shape == (23, 10, 2))
        new_buffer = update_buffer(buffer, jnp.zeros(shape=(2, 10, 2)))
        self.assertIsInstance(new_buffer, ContinuousReplayBuffer)

    def test_kwargs_continuous(self):
        key = jax.random.PRNGKey(42)
        batch_size = 23
        sampler = LangevinSampler((10, 2), batch_size, -1.0, 1.0, 10.0, 0.005, 20)

        buffer = ContinuousReplayBuffer((100, 10, 2), 0.0, 1.0, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), samples, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples.shape == (23, 10, 2))

        key = jax.random.PRNGKey(42)
        batch_size = 23
        sampler = HMCSampler((10,), batch_size, -1.0, 1.0, 10, 0.5, 3, jnp.ones((10,)))

        buffer = ContinuousReplayBuffer((100, 10), 0.0, 1.0, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), samples, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples.shape == (23, 10))

        sampler = LangevinSampler(
            (10, 2), batch_size, -1.0, 1.0, 10.0, 0.005, 20, metropolis_adjustment=True
        )

        buffer = ContinuousReplayBuffer((100, 10, 2), 0.0, 1.0, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), samples, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples.shape == (23, 10, 2))

        sampler = ErrorAdaptiveLangevinSampler(
            (10, 2), batch_size, -1.0, 1.0, 10.0, 0.005, 20, 1e-5, 1e-8, 1e-7, 1e-7
        )
        buffer = ContinuousReplayBuffer((100, 10, 2), 0.0, 1.0, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), samples, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples.shape == (23, 10, 2))

    def test_anneal_langevin(self):
        batch_size = 23
        sampler = AnnealedLangevinSampler(
            (10, 2), batch_size, -1.0, 1.0, jnp.array([0.1, 0.01]), 5, 1e-5
        )
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), None, jax.random.PRNGKey(0)
        )["position"]
        self.assertTrue(samples.shape == (23, 10, 2))

    def test_return_all_states(self):
        batch_size = 23
        num_langevin_steps = 5
        num_noise_levels = 2

        # LangevinSampler
        sampler = LangevinSampler(
            (10, 2),
            batch_size,
            -1.0,
            1.0,
            10.0,
            0.005,
            num_langevin_steps,
            return_full_chain=True,
        )
        samples = sampler.sample_chains(
            Constant(1.0), None, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        print
        self.assertTrue(samples.shape == (batch_size, num_langevin_steps, 10, 2))

        # AnnealedLangevinSampler - returns all steps for every noise level (num_chains, noise_levels, num_steps, input_size)
        sampler = AnnealedLangevinSampler(
            (10, 2),
            batch_size,
            -1.0,
            1.0,
            jnp.array([0.1, 0.01]),
            num_langevin_steps,
            1e-5,
            return_full_chain=True,
        )
        samples = sampler.sample_chains(Constant(1.0), None, jax.random.PRNGKey(0))[
            "position"
        ]
        self.assertTrue(
            samples.shape == (batch_size, num_noise_levels, num_langevin_steps, 10, 2)
        )

        # HMCSampler
        sampler = HMCSampler(
            (10,),
            batch_size,
            -1.0,
            1.0,
            num_langevin_steps,
            0.5,
            3,
            jnp.ones((10,)),
            return_full_chain=True,
        )
        samples = sampler.sample_chains(Constant(1.0), None, jax.random.PRNGKey(0))[
            "position"
        ]
        self.assertTrue(samples.shape == (batch_size, num_langevin_steps, 10))

        # Error Adaptive Langevin Sampler
        sampler = ErrorAdaptiveLangevinSampler(
            (10, 2),
            batch_size,
            -1.0,
            1.0,
            10.0,
            0.005,
            num_langevin_steps,
            1e-5,
            1e-8,
            1e-7,
            1e-7,
            return_full_chain=True,
        )
        buffer = ContinuousReplayBuffer(
            (100, 10, 2), 0.0, 1.0, batch_size, 0.7, jax.random.PRNGKey(0)
        )
        samples = buffer.sample(jax.random.PRNGKey(0))
        samples = sampler.sample_chains(
            Constant(1.0),
            samples,
            jax.random.PRNGKey(0),
            keyword=jnp.ones((batch_size,)),
        )["position"]
        self.assertTrue(samples.shape == (batch_size, num_langevin_steps, 10, 2))

    def test_buffer_update_continuous(self):
        key = jax.random.PRNGKey(42)
        batch_size = 23

        buffer = ContinuousReplayBuffer((100, 10), 0.0, 1.0, batch_size, 0.7, key)
        samples = buffer.sample(key)
        self.assertTrue(samples.shape == (23, 10))
        new_buffer = update_buffer(buffer, jnp.zeros(shape=(2, 10)))
        self.assertIsInstance(new_buffer, ContinuousReplayBuffer)

    def test_buffer_shapes_discrete(self):
        key = jax.random.PRNGKey(42)
        batch_size = 23
        chain_length = 4
        sampler = DiscreteUniformMH((10,), 10, chain_length, batch_size)

        buffer = DiscreteReplayBuffer((100, 10), 2, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), samples, jax.random.PRNGKey(0)
        )["position"]
        self.assertTrue(samples.shape == (batch_size, chain_length, 10))

    def test_pytree_inputs(self):
        batch_size = 23
        in_tree = {"a": jnp.ones((10, 2)), "b": jnp.ones((1, 2, 3, 4))}
        sampler = LangevinSampler(in_tree, batch_size, -1.0, 1.0, 10.0, 0.005, 20)

        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), None, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples["a"].shape == (23, 10, 2))
        self.assertTrue(samples["b"].shape == (23, 1, 2, 3, 4))

        chain_length = 4
        in_tree = {
            "a": jnp.ones((10, 2)).astype("int8"),
            "b": jnp.ones((1, 2, 3, 4)).astype("int8"),
        }
        sampler = DiscreteUniformMH(in_tree, 10, chain_length, batch_size)

        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), None, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        self.assertTrue(samples["a"].shape == (23, chain_length, 10, 2))
        self.assertTrue(samples["b"].shape == (23, chain_length, 1, 2, 3, 4))

    def test_kwargs_discrete(self):
        key = jax.random.PRNGKey(42)
        batch_size = 23
        chain_length = 4
        sampler = DiscreteUniformMH((10,), 10, chain_length, batch_size)

        buffer = DiscreteReplayBuffer((100, 10), 2, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0), samples, jax.random.PRNGKey(0), keyword=jnp.ones(batch_size)
        )["position"]
        # jax.tree_util.tree_map(
        #    lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), samples
        # )  # reshape from [num_chains, chain_length, *xshape] -> [num_chains * chain length, *xshape]
        self.assertTrue(samples.shape == (batch_size, chain_length, 10))

        sampler = GibbsWithGradient((10,), 10, chain_length, batch_size)

        buffer = DiscreteReplayBuffer((100, 10), 2, batch_size, 0.7, key)
        samples = buffer.sample(key)
        samples = eqx.filter_jit(sampler.sample_chains)(
            Constant(1.0),
            samples,
            jax.random.PRNGKey(0),
            temperature=0.5 * jnp.ones(batch_size),
        )["position"]
        self.assertTrue(samples.shape == (batch_size, chain_length, 10))

    def test_buffer_update_discrete(self):
        key = jax.random.PRNGKey(42)
        batch_size = 23

        buffer = DiscreteReplayBuffer((100, 10), 2, batch_size, 0.7, key)
        new_buffer = update_buffer(buffer, jnp.zeros(shape=(2, 10)))
        self.assertIsInstance(new_buffer, DiscreteReplayBuffer)


if __name__ == "__main__":
    absltest.main()
