import itertools
import unittest
import numpy as np
from jax import numpy as jnp
import jax
from energax import utils


def _get_domain(structure):
    return jnp.array(list(itertools.product(*[range(s) for s in structure])))


class UtilTests(unittest.TestCase):
    def test_softmax(self):
        array = jnp.array([-1.0, 0.0, 1.0, 100.0, 1000.0])
        energax_sm, _ = utils.stable_softmax(array)
        jax_sm = jax.nn.softmax(array)
        self.assertTrue(jnp.allclose(energax_sm, jax_sm))

        jit_sm = jax.jit(utils.stable_softmax)
        energax_sm, _ = jit_sm(array)
        self.assertTrue(jnp.allclose(energax_sm, jax_sm))

    def test_get_domain(self):
        for i in range(5):
            structure = [x + 2 for x in range(i + 2)]
            with self.subTest(f"qudits with structure={structure}"):
                truth_result = _get_domain(structure)
                expected_result = utils.get_domain(structure)
                np.testing.assert_array_equal(expected_result, truth_result)
