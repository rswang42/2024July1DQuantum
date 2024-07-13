"""Support Functionalities"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as flax_nn
import optax

jax.config.update("jax_enable_x64", True)


def buildH(
    Vpot: callable,
    xmesh: np.ndarray,
    Nmesh: int,
    h: float | int,
):
    """Build Hamiltonian on a discrete mesh"""
    Vx = np.array([Vpot(x) for x in xmesh])
    H = np.diag(Vx)

    for i in range(Nmesh):
        H[i, i] += 1.0 / (h * h)

    for i in range(Nmesh - 1):
        H[i, i + 1] += -0.5 / (h * h)
        H[i + 1, i] += -0.5 / (h * h)

    return H


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def hermite(n: int, x: float) -> jax.Array:
    """The hermite polynominals"""
    h0 = 1.0 / jnp.pi ** (1 / 4)
    h1 = jnp.sqrt(2.0) * x / jnp.pi ** (1 / 4)

    def body_fun(i, val):
        valm2, valm1 = val
        return valm1, jnp.sqrt(2.0 / i) * x * valm1 - jnp.sqrt((i - 1) / i) * valm2

    _, hn = jax.lax.fori_loop(2, n + 1, body_fun, (h0, h1))

    return jax.lax.cond(n > 0, lambda: hn, lambda: h0)


@hermite.defjvp
def hermite_jvp(n: int, primals, tangents):
    (x,) = primals
    (dx,) = tangents
    hn = hermite(n, x)
    dhn = jnp.sqrt(2 * n) * hermite((n - 1) * (n > 0), x) * dx
    primals_out, tangents_out = hn, dhn
    return primals_out, tangents_out


def wf_base(x: float | jax.Array, n: int, m=1) -> jax.Array:
    """The wave function ansatz (Gaussian)
    NOTE: 1D!

    Args:
        x: the 1D coordinate of the (single) particle.
        n: the excitation quantum number
        m: the particle mass(a.u.)

        NOTE: n=0 for GS!

    Returns:
        psi: the probability amplitude at x.
    """
    psi = (m ** (1 / 4)) * jnp.exp(-0.5 * m * (x**2)) * hermite(n, jnp.sqrt(m) * x)
    return psi


class WFAnsatz:
    """The flow wave function ansatz"""

    def __init__(
        self,
        flow: flax_nn.Module,
    ) -> None:
        self.flow = flow

    def log_wf_ansatz(
        self,
        params: jax.Array | np.ndarray,
        x: float | jax.Array,
    ) -> jax.Array:
        """The flow transformed log wavefunction

        Args:
            params: the flow parameter
            x: the coordinate before flow

        Returns:
            log_amplitude: the log wavefunction (with real
                and imagine part)
        """
        raise NotImplementedError
        z = self.flow.apply(params, x)
        log_phi = wf_base(z)

        flow_flatten = lambda x: self.flow.apply(params, x)
        jac = jax.jacfwd(flow_flatten)(x)
        logjacdet = jnp.log(abs(jac))

        return jnp.stack([log_phi.real + 0.5 * logjacdet, log_phi.imag])

    def wf_ansatz(
        self,
        params: jax.Array | np.ndarray | dict,
        x: float | jax.Array,
        n: int,
    ) -> jax.Array:
        """The flow transformed log wavefunction

        Args:
            params: the flow parameter
            x: the coordinate before flow
            n: the excitation quantum number

            NOTE: n=0 for GS!

        Returns:
            amplitude: the wavefunction
        """
        z = self.flow.apply(params, x)[0]
        phi = wf_base(z, n)

        def _flow_func(x):
            return self.flow.apply(params, x)[0]

        jac = jax.jacfwd(_flow_func)(x)
        jacdet = jnp.sqrt(jnp.abs(jac))
        amplitude = phi * jacdet
        return amplitude


class EnergyEstimator:
    """Energy Estimator"""

    def __init__(self, wf_ansatz: callable) -> None:
        self.wf_ansatz = wf_ansatz

    def local_kinetic_energy(
        self,
        params: jax.Array | np.ndarray | dict,
        xs: jax.Array,
        state_indices: np.ndarray,
    ) -> jax.Array:
        """Local Kinetic Energy Estimator
        NOTE: per batch implementation.

        Args:
            params: the flow parameter
            xs: (num_of_orbs,) the 1D coordinate of the particle(s).
                The xs are in corresponding order as in state_indices.
            state_indices: (num_of_orbs,) the array containing
                the excitations in each bath, for example,
                [0,1,2] represents there are totally 3 number
                of orbitals: ground state: 0
                            first excitation state: 1
                            second excitation state: 2

        Returns:
            local_kinetic: the local kinetic energy.
        """
        single_laplacian_func = jax.grad(jax.grad(self.wf_ansatz, argnums=1), argnums=1)
        wf_vmapped = jax.vmap(self.wf_ansatz, in_axes=(None, 0, 0))
        laplacians = jax.vmap(single_laplacian_func, in_axes=(None, 0, 0))(
            params, xs, state_indices
        )
        local_kinetics = -0.5 * laplacians / wf_vmapped(params, xs, state_indices)
        return local_kinetics

    @staticmethod
    def local_potential_energy(
        xs: jax.Array,
    ) -> jax.Array:
        """Local Potential Energy Estimator

        Args:
            xs: (num_of_orbs,) the 1D coordinate of the particle(s).
                The xs are in corresponding order as in state_indices.
        Returns:
            local_potential: the local potential energy.
        """
        local_potentials = 3 * xs**4 + xs**3 / 2 - 3 * xs**2
        return local_potentials

    def local_energy(
        self,
        params: jax.Array | np.ndarray | dict,
        xs: jax.Array,
        state_indices: np.ndarray,
    ) -> jax.Array:
        """Local Energy Estimator

        Args:
            params: the flow parameter
            xs: (num_of_orbs,) the 1D coordinate of the particle(s).
                The xs are in corresponding order as in state_indices.
            state_indices: (num_of_orbs,) the array containing
                the excitations in each bath, for example,
                [0,1,2] represents there are totally 3 number
                of orbitals: ground state: 0
                            first excitation state: 1
                            second excitation state: 2

        Returns:
            local_energy: the local energy.
        """
        kin_energy = self.local_kinetic_energy(params, xs, state_indices)
        pot_energy = self.local_potential_energy(xs)
        local_energy = kin_energy + pot_energy
        return local_energy


class Metropolis:
    """The Metropolis Algorithm
    iterating the positions of MCMC walkers according
    to the Metropolis algorithm.
    """

    def __init__(self, wf_ansatz: callable) -> None:
        self.wf_ansatz = wf_ansatz

    def oneshot_sample(
        self,
        xs: jax.Array,
        state_indices: np.ndarray,
        probability: jax.Array,
        params: jax.Array | np.ndarray | dict,
        step_size: float,
        key: jax.random.PRNGKey,
    ) -> list[
        float | jax.Array,
        float,
        int,
    ]:
        """The one-step Metropolis update

        Args:
            xs: (num_of_orbs,) the 1D coordinate of the particle(s).
                The xs are in corresponding order as in state_indices.
            state_indices: (num_of_orbs,) the array containing
                the excitations in each bath, for example,
                [0,1,2] represents there are totally 3 number
                of orbitals: ground state: 0
                            first excitation state: 1
                            second excitation state: 2
            probability:(num_of_orbs,) the probability in current particle coordiante
                xs and
                with corresponding state_indices.
            params: the flow parameter
            step_size: the step size of each sample step. e.g.
                x_new = x + step_size * jax.random.normal(subkey, shape=pos.shape)
            key: the jax PRNG key.

        Returns:
            xs_new: the updated xs
            probability_new: the updated probability
            accetp_count: the number of updates performed per walker.
        """
        key, subkey = jax.random.split(key)
        wf_vmapped = jax.vmap(self.wf_ansatz, in_axes=(None, 0, 0))

        xs_new = xs + step_size * jax.random.normal(subkey, shape=xs.shape)
        probability_new = wf_vmapped(params, xs_new, state_indices) ** 2

        # Metropolis
        key, subkey = jax.random.split(key)
        cond = (
            jax.random.uniform(subkey, shape=probability.shape)
            < probability_new / probability
        )
        probability_new = jnp.where(cond, probability_new, probability)
        xs_new = jnp.where(cond, xs_new, xs)

        # Count accepted proposals
        accept_count = jnp.sum(cond)

        return xs_new, probability_new, accept_count


def mcmc(
    steps: int,
    num_substeps: int,
    metropolis_sampler_batched: callable,
    key: jax.random.PRNGKey,
    xs_batched: jax.Array,
    state_indices: np.ndarray,
    params: dict,
    probability_batched: jax.Array,
    mc_step_size: float,
    pmove: float,
) -> list[jax.random.PRNGKey, jax.Array, jax.Array, float, float]:
    """The batched mcmc function for #steps sampling.
    NOTE: this is a jax foriloop implementation

    Args:
        steps: the steps to perform
        num_substeps: the mcmc substep number.
        metropolis_sampler_batched: the batched metropolis sampler.
        key: the jax.PRNGkey
        xs_batched: (num_of_batch,num_of_orbs,) the batched 1D coordinate
            of the particle(s).
            The xs are in corresponding order as in state_indices.
        state_indices: (num_of_orbs,)the array containing
            the excitations in each bath, for example,
            [0,1,2] represents there are totally 3 number
            of orbitals: ground state: 0
                        first excitation state: 1
                        second excitation state: 2
        params: the parameters of network
        probability_batched: (num_of_batch,num_of_orbs,)
            the batched probability for each state (wavefunction**2)
        mc_step_size: last mcmc moving step size.
        pmove: the portion of moved particles in last mcmc step.

    Returns:
        key: the jax.PRNGkey
        xs_batched: (num_of_batch,num_of_orbs,) the batched 1D
            coordinate of the particle(s).
            The xs are in corresponding order as in state_indices.
        probability_batched: (num_of_batch,num_of_orbs,)
            the batched probability for each state (wavefunction**2)
        mc_step_size: updated mcmc moving step size.
        pmove: the portion of moved particles in current mcmc step.
    """

    @jax.jit
    def _body_func(i, val):
        """MCMC Body function"""
        key, xs_batched, probability_batched, mc_step_size, pmove = val
        batch_size = xs_batched.shape[0]
        key, batch_keys = key_batch_split(key, batch_size)
        xs_batched, probability_batched, accept_count = metropolis_sampler_batched(
            xs_batched,
            state_indices,
            probability_batched,
            params,
            mc_step_size,
            batch_keys,
        )
        pmove = np.sum(accept_count) / (
            num_substeps * batch_size * state_indices.shape[0]
        )
        mc_step_size = jnp.where(
            pmove > 0.5,
            mc_step_size * 1.1,
            mc_step_size * 0.9,
        )
        return key, xs_batched, probability_batched, mc_step_size, pmove

    mcmc_init_val = (key, xs_batched, probability_batched, mc_step_size, 0)
    key, xs_batched, probability_batched, mc_step_size, pmove = jax.lax.fori_loop(
        0, steps, _body_func, mcmc_init_val
    )
    return key, xs_batched, probability_batched, mc_step_size, pmove


def init_batched_x(
    key: jax.random.PRNGKey,
    batch_size: int,
    num_of_orbs: int,
    init_width: float = 1.0,
) -> jax.Array:
    """Init batched x

    Args:
        key: the jax PRNG key
        batch_size: the number of walkers to creat.
        num_of_orbs: the total number of orbitals.
        init_width: scale of initial Gaussian distribution

    Returns:
        init_x: (batch_size,) the initialized (batched) x.
    """
    init_x = init_width * jax.random.normal(
        key, shape=(batch_size, num_of_orbs), dtype=jnp.float64
    )
    return init_x


def key_batch_split(key: jax.random.PRNGKey, batch_size: int):
    """Like jax.random.split, but returns one subkey per batch element.

    Args:
        key: jax.PRNGkey
        batch_size: the batch size.

    Returns:
        key: jax.PRNGkey, shape (2,)
        batch_keys: jax.PRNGkeys, shape (batch_size, 2)
    """
    key, *batch_keys = jax.random.split(key, num=batch_size + 1)
    return key, jnp.asarray(batch_keys)


def make_loss(
    wf_ansatz: callable, local_energy_estimator: callable, state_indices: np.ndarray
):
    """Copied from Ferminet.

    Creates the loss function, including custom gradients.
    NOTE: batched implementation!

    Args:
    wf_ansatz: function, signature (params, x, n), which evaluates wavefunction
        at a single MCMC configuration given the network parameters.
    local_energy_estimator: signature (params, xs, state_indices), here
        xs refers to the x coordinates in one single batch, xs has shape (num_of_orb,)
    state_indices: (num_of_orbs,)the array containing
        the excitations in each bath, for example,
        [0,1,2] represents there are totally 3 number
        of orbitals: ground state: 0
                    first excitation state: 1
                    second excitation state: 2
    Returns:
        total_energy: function which evaluates the both the loss as the mean of the
        local_energies, and the local energies. With custom gradients, accounting
        for derivatives in the probability distribution.
    """
    batch_local_energy = jax.vmap(
        local_energy_estimator, in_axes=(None, 0, None), out_axes=0
    )
    # FermiNet is defined in terms of the logarithm of the network, which is
    # better for numerical stability and also makes some expressions neater.
    log_wf = lambda params, x, n: jnp.log(jnp.abs(wf_ansatz(params, x, n)))
    # vmapped wavefunction: signature (params, xs, state_indices)
    log_wf_vmapped = jax.vmap(log_wf, in_axes=(None, 0, 0))
    # batch_wf: signature (params, batched_xs, state_indices)
    batch_wf = jax.vmap(log_wf_vmapped, in_axes=(None, 0, None), out_axes=0)

    @jax.custom_jvp
    def total_energy(params, batched_xs) -> tuple[float, np.ndarray]:
        """Total energy of an ensemble (batched)

        Args:
            params: the network parameters
            batched_xs: (num_of_batch, num_of_orbs,)
                the batched 1D coordinate of the particle(s).
                The xs are in corresponding order as in state_indices.

        Returns:
            loss: the total loss of the system, meaned over batch.
            e_l: (batch,num_of_orbs,) the local energy of each state.
        """
        e_l = batch_local_energy(params, batched_xs, state_indices)
        loss = jnp.mean(jnp.sum(e_l, axis=-1))
        return loss, e_l

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        """Custom Jacobian-vector product for unbiased local energy gradients."""
        params, batched_xs = primals
        loss, local_energy = total_energy(params, batched_xs)
        diff = local_energy - loss

        def _batch_wf(params, batched_xs):
            return batch_wf(params, batched_xs, state_indices)

        psi_primal, psi_tangent = jax.jvp(_batch_wf, primals, tangents)
        primals_out = loss, local_energy
        tangents_out = (
            jnp.mean(2 * jnp.sum(psi_tangent * diff, axis=-1)),
            local_energy,
        )

        return primals_out, tangents_out

    return total_energy


class Update:
    """Update functions"""

    def __init__(
        self,
        mcmc_steps: int,
        state_indices: np.ndarray,
        batch_size: int,
        num_substeps: int,
        metropolis_sampler_batched: callable,
        loss_and_grad: callable,
        optimizer: optax.GradientTransformation,
        acc_steps: int,
    ) -> None:
        self.mcmc_steps = mcmc_steps
        self.state_indices = state_indices
        self.batch_size = batch_size
        self.num_substeps = num_substeps
        self.metropolis_sampler_batched = metropolis_sampler_batched
        self.loss_and_grad = loss_and_grad
        self.optimizer = optimizer
        self.acc_steps = acc_steps

    def update(
        self,
        key: jax.random.PRNGKey,
        xs_batched: jax.Array,
        probability_batched: jax.Array,
        mc_step_size: float,
        params: jax.Array | np.ndarray | dict,
        opt_state: optax.OptState,
    ) -> tuple[float, float, jax.Array, jax.Array, dict, optax.OptState, float, float]:
        """Single update
        NOTE: call WITHOUT vmap!

        Args:
            key: the jax.PRNGkey
            xs_batched: (num_of_batch,num_of_orbs,) the batched 1D coordinate
                of the particle(s).
                The xs are in corresponding order as in state_indices.
            probability_batched: (num_of_batch,num_of_orbs,)
                the batched probability for each state (wavefunction**2)
            mc_step_size: the initial mcmc moving step size.
            params: the flow parameters
            opt_state: the optimizer state.

        Returns:
            loss: the meaned energy as loss
            energy_std: the energy standard deviation.
            xs_batched: the coordinate
            probability_batched: the batched probability (wavefunction**2)
            params: the flow parameters
            opt_state: the optimizer state.
            pmove: the portion of moved coordinates.
            mc_step_size: the mcmc moving step size.
        """

        def _mcmc_body_func(i, val):
            """MCMC Body function"""
            key, xs_batched, probability_batched, mc_step_size, pmove = val
            key, batch_keys = key_batch_split(key, self.batch_size)
            (
                xs_batched,
                probability_batched,
                accept_count,
            ) = self.metropolis_sampler_batched(
                xs_batched,
                self.state_indices,
                probability_batched,
                params,
                mc_step_size,
                batch_keys,
            )
            pmove = np.sum(accept_count) / (
                self.num_substeps * self.batch_size * self.state_indices.shape[0]
            )
            mc_step_size = jnp.where(
                pmove > 0.5,
                mc_step_size * 1.1,
                mc_step_size * 0.9,
            )
            return key, xs_batched, probability_batched, mc_step_size, pmove

        def _acc_body_func(i, val):
            """Gradient Accumulation Body Function"""
            (
                loss,
                energies,
                grad,
                key,
                xs_batched,
                probability_batched,
                mc_step_size,
                pmove,
                params,
            ) = val
            mcmc_init_val = (key, xs_batched, probability_batched, mc_step_size, 0)
            (
                key,
                xs_batched,
                probability_batched,
                mc_step_size,
                pmove,
            ) = jax.lax.fori_loop(0, self.mcmc_steps, _mcmc_body_func, mcmc_init_val)
            (loss, energies), gradients = self.loss_and_grad(params, xs_batched)
            grad = jax.tree.map(jnp.mean, gradients)
            return (
                loss,
                energies,
                grad,
                key,
                xs_batched,
                probability_batched,
                mc_step_size,
                pmove,
                params,
            )

        # Only for initialization of val in jax.lax.foriloop!
        (loss, energies), gradients = self.loss_and_grad(params, xs_batched)
        grad = jax.tree.map(jnp.mean, gradients)
        acc_init_val = (
            loss,
            energies,
            grad,
            key,
            xs_batched,
            probability_batched,
            mc_step_size,
            0,
            params,
        )

        (
            loss,
            energies,
            grad,
            key,
            xs_batched,
            probability_batched,
            mc_step_size,
            pmove,
            params,
        ) = jax.lax.fori_loop(0, self.acc_steps, _acc_body_func, acc_init_val)

        updates, opt_state = self.optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        energy_std = jnp.sqrt(
            ((jnp.sum(energies, axis=-1) ** 2).mean() - loss**2)
            / (self.acc_steps * self.batch_size)
        )
        return (
            loss,
            energy_std,
            xs_batched,
            probability_batched,
            params,
            opt_state,
            pmove,
            mc_step_size,
        )


class MLPFlow(flax_nn.Module):
    """A simple MLP flow"""

    out_dims: int

    @flax_nn.compact
    def __call__(self, x):
        _init_x = x
        x = x.reshape(
            1,
        )
        x = flax_nn.Dense(3)(x)
        x = flax_nn.sigmoid(x)
        x = flax_nn.Dense(3)(x)
        x = flax_nn.sigmoid(x)
        x = flax_nn.Dense(self.out_dims)(x)
        x = _init_x + x
        _init_x = x
        x = flax_nn.Dense(3)(x)
        x = flax_nn.sigmoid(x)
        x = flax_nn.Dense(3)(x)
        x = flax_nn.sigmoid(x)
        x = flax_nn.Dense(self.out_dims)(x)
        x = _init_x + x
        _init_x = x
        x = flax_nn.Dense(3)(x)
        x = flax_nn.sigmoid(x)
        x = flax_nn.Dense(3)(x)
        x = flax_nn.sigmoid(x)
        x = flax_nn.Dense(self.out_dims)(x)
        return _init_x + x
