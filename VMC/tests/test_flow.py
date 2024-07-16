"""Test Flow"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from VMC.utils import MLPFlow


class TestFlow(unittest.TestCase):
    """Test Flow"""

    def setUp(self) -> None:
        self.x_shape = 3
        key = jax.random.PRNGKey(42)
        self.model_flow = MLPFlow(out_dims=self.x_shape, mlp_width=3, mlp_depth=3)
        key, subkey = jax.random.split(key)
        x_dummy = jax.random.normal(subkey, shape=(self.x_shape,))  # Dummy input data
        key, subkey = jax.random.split(key)
        self.params = self.model_flow.init(subkey, x_dummy)

    def tearDown(self) -> None:
        pass

    def test_seperality(self) -> None:
        """Test flowed coordinates only depend on input of itself"""
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        # init_x = jax.random.normal(subkey,shape=(x_shape,))
        init_x = jnp.zeros(self.x_shape)
        init_z = self.model_flow.apply(self.params, init_x)

        # change x[0]
        x_changed = init_x.at[0].set(init_x[0] + 0.5)
        z_after_x_changed = self.model_flow.apply(self.params, x_changed)

        expect_unchange_indices = np.array([1, 2])
        print(f"init_x={init_x}\nx_changed={x_changed}")
        np.testing.assert_array_almost_equal(
            z_after_x_changed[expect_unchange_indices], init_z[expect_unchange_indices]
        )


if __name__ == "__main__":
    unittest.main()
