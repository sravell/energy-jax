"""EBMs testing."""

# TODO: remove absl test
from absl.testing import absltest
import jax
from jax import numpy as jnp
from energax.ebms import ebm
from energax.sampling import continuous


class FunctionalEBMTests(absltest.TestCase):
    def test_sampling(self):
        def f(params, state):
            return jnp.mean(params * state)

        init_params = jnp.zeros(10)
        init_state = jnp.ones((4, 10))

        debm = ebm.FunctionalEBM(f, init_params)

        sampler = continuous.LangevinSampler(
            xshape=None,
            num_chains=4,
            minval=None,
            maxval=None,
            stepsize=0.001,
            sigma=0.005,
            num_langevin_steps=10,
            sample_clip=None,
            grad_clip=None,
            metropolis_adjustment=True,
        )
        samples = sampler.sample_chains(debm, init_state, jax.random.PRNGKey(0))[
            "position"
        ]
        self.assertTupleEqual((4, 10), samples.shape)


if __name__ == "__main__":
    absltest.main()
