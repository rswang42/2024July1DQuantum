"""Test Rotate basis"""

import os
import unittest
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np

from VMC.utils import WFAnsatz
from VMC.utils import MLPFlow


class TestOrthogonal(unittest.TestCase):
    """Test wavefunction orthogonality"""

    def setUp(self) -> None:
        key = jax.random.PRNGKey(42)
        self.model_flow = MLPFlow(out_dims=1, mlp_width=3, mlp_depth=3)
        key, subkey = jax.random.split(key)
        x_dummy = jax.random.normal(subkey, dtype=jnp.float64)  # Dummy input data
        key, subkey = jax.random.split(key)
        self.params = self.model_flow.init(subkey, x_dummy)
        self.wf_ansatz_obj = WFAnsatz(flow=self.model_flow)

    def tearDown(self) -> None:
        pass

    def test_orthogonal(self) -> None:
        """Test orthogonality"""
        xmin = -10
        xmax = 10
        Nmesh = 2000
        xmesh = np.linspace(xmin, xmax, Nmesh, dtype=np.float64)
        state_indices = np.arange(5)
        wf_vmapped = jax.vmap(self.wf_ansatz_obj.wf_ansatz, in_axes=(None, 0, None))
        for i, j in combinations(state_indices, r=2):
            wfi = wf_vmapped(self.params, xmesh, i)
            wfj = wf_vmapped(self.params, xmesh, j)
            inner = jnp.sum(wfi * wfj)
            self.assertAlmostEqual(inner, 0)


if __name__ == "__main__":
    unittest.main()
