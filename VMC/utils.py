"""Support Functionalities"""

from functools import partial
import time
import os

import jax.ad_checkpoint
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as flax_nn
import optax
import tqdm.notebook as tqdm

jax.config.update("jax_enable_x64", True)


def buildH(
    Vpot: callable,
    xmesh: np.ndarray,
    Nmesh: int,
    h: float | int,
):
    """Build Hamiltonian on a discrete mesh"""
    Vx = Vpot(xmesh)
    Vx = jnp.diag(Vx, 0)
    H_kinetic = (
        -2 * jnp.diag(jnp.ones(Nmesh, dtype=jnp.float64), 0)
        + jnp.diag(jnp.ones(Nmesh - 1, dtype=jnp.float64), 1)
        + jnp.diag(jnp.ones(Nmesh - 1, dtype=jnp.float64), -1)
    ) / (2 * h**2)
    H = Vx - H_kinetic

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


def wf_base(
    x: float | jax.Array,
    n: int,
) -> jax.Array:
    """The wave function ansatz (Gaussian)
    NOTE: 1D!

    Args:
        x: the 1D coordinate of the (single) particle.
        n: the excitation quantum number

        NOTE: n=0 for GS!

    Returns:
        psi: the probability amplitude at x.
    """
    psi = jnp.exp(-0.5 * (x**2)) * hermite(n, x)
    return psi


def log_wf_base(
    x: float | jax.Array,
    n: int,
) -> jax.Array:
    """The wave function ansatz (Gaussian)
    NOTE: 1D!

    Args:
        x: the 1D coordinate of the (single) particle.
        n: the excitation quantum number

        NOTE: n=0 for GS!

    Returns:
        log_psi: the log probability amplitude at x.
    """
    log_psi = -0.5 * x**2 + jnp.log(jnp.abs(hermite(n, x)))
    return log_psi


class WFAnsatz:
    """The flow wave function ansatz

    Attributes:
        self.flow: the normalizing flow network
    """

    def __init__(
        self,
        flow: flax_nn.Module,
    ) -> None:
        self.flow = flow

    def log_wf_ansatz(
        self,
        params: jax.Array | np.ndarray,
        x: float | jax.Array,
        n: int,
    ) -> jax.Array:
        """The flow transformed log wavefunction

        Args:
            params: the flow parameter
            x: the coordinate before flow
            n: the excitation quantum number

        Returns:
            log_amplitude: the log wavefunction
                log|psi|
        """
        z = self.flow.apply(params, x)[0]
        log_phi = log_wf_base(z, n)

        def _flow_func(x):
            return self.flow.apply(params, x)[0]

        jac = jax.jacfwd(_flow_func)(x)
        # _,logjacdet = jnp.linalg.slogdet(jac)
        logjacdet = jnp.log(jnp.abs(jac))

        log_amplitude = log_phi + 0.5 * logjacdet
        return log_amplitude

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

    def log_wf_ansatz_fock(
        self,
        params: jax.Array | np.ndarray,
        xs: jax.Array,
        state_indices: jax.Array,
    ) -> jax.Array:
        """The flow transformed log wavefunction
        NOTE: in fock space! Namely that gives
        the wavefunction of different orbitals
        as concatenated array!

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
            log_amplitudes: (num_of_orbs,) the log wavefunction
                log|psi| corresponding to each orbitals
                in state_indices.
        """
        # zs: (num_of_orbs,)
        # prefactors: (num_of_orbs,)
        zs, prefactors = self.flow.apply(params, xs)

        vmapped_log_wf_base = jax.vmap(
            log_wf_base,
            in_axes=(0, 0),
            out_axes=0,
        )
        # logphis: (num_of_orbs,)
        log_phis = vmapped_log_wf_base(zs, state_indices)

        log_prefactors = jnp.log(prefactors)

        def _flow_func(xs):
            new_xs, _ = self.flow.apply(params, xs)
            new_xs = new_xs.reshape(-1)
            return new_xs

        # jac: (num_of_orbs,num_of_orbs)
        jac = jax.jacfwd(_flow_func)(xs)

        # jac_fock: (num_of_orbs,)
        jac_fock = jnp.diagonal(jac)
        logjacdet_fock = jnp.log(jnp.abs(jac_fock))

        log_amplitudes = log_prefactors + log_phis + 0.5 * logjacdet_fock
        return log_amplitudes

    def wf_ansatz_fock(
        self,
        params: jax.Array | np.ndarray | dict,
        xs: jax.Array,
        state_indices: jax.Array,
    ) -> jax.Array:
        """The flow transformed log wavefunction
        NOTE: in fock space! Namely that gives
        the wavefunction of different orbitals
        as concatenated array!

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

            NOTE: n=0 for GS!

        Returns:
            amplitudes: (num_of_orbs,) the wavefunction
                log|psi| corresponding to each orbitals
                in state_indices.
        """
        zs, prefactors = self.flow.apply(params, xs)

        vmapped_wf_base = jax.vmap(
            wf_base,
            in_axes=(0, 0),
            out_axes=0,
        )
        phis = vmapped_wf_base(zs, state_indices)

        def _flow_func(xs):
            new_xs, _ = self.flow.apply(params, xs)
            new_xs = new_xs.reshape(-1)
            return new_xs

        jac = jax.jacfwd(_flow_func)(xs)

        jac_fock = jnp.diagonal(jac)
        jacdet_fock = jnp.sqrt(jnp.abs(jac_fock))

        amplitudes = prefactors * phis * jacdet_fock
        return amplitudes


class EnergyEstimator:
    """Energy Estimator

    Attributes:
        self.wf_ansatz_fock: the callable wave function ansatz
            in fock space
            signature: (params, xs, state_indices)
        self.log_gomain: True for working in log domain.
    """

    def __init__(
        self,
        wf_ansatz_fock: callable,
        log_domain: bool = True,
    ) -> None:
        self.wf_ansatz_fock = wf_ansatz_fock
        self.log_domain = log_domain

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
            local_kinetic: (num_of_orbs,) the local kinetic energy.
        """

        # NOTE: Previously single state wf ansatze.
        # =====================================================
        # NOTE: only work for one-dimensional system!
        # For higher dimension, use jax.jacrev and jax.jvp

        # single_grad_func = jax.grad(self.wf_ansatz, argnums=1)
        # single_laplacian_func = jax.grad(single_grad_func, argnums=1)
        # wf_vmapped = jax.vmap(self.wf_ansatz, in_axes=(None, 0, 0))
        # grads = jax.vmap(single_grad_func, in_axes=(None, 0, 0))(
        #     params, xs, state_indices
        # )
        # laplacians = jax.vmap(single_laplacian_func, in_axes=(None, 0, 0))(
        #     params, xs, state_indices
        # )
        # =====================================================

        # NOTE: currently only working on one-dimensional x!
        def _wf_fock(xs):
            return self.wf_ansatz_fock(params, xs, state_indices)

        first_order_derivative = jax.jacfwd(_wf_fock)(xs)
        grads = jnp.diagonal(first_order_derivative)

        second_order_derivative = jax.hessian(_wf_fock)(xs)
        main_diagonals = jnp.diagonal(second_order_derivative)
        laplacians = jnp.diagonal(main_diagonals)

        if self.log_domain:
            local_kinetics = -0.5 * (laplacians + (grads**2))
        else:
            local_kinetics = (
                -0.5 * laplacians / self.wf_ansatz_fock(params, xs, state_indices)
            )
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
            local_potential: (num_of_orbs,) the local potential energy.
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
            local_energy: (num_of_orbs,) the local energy.
        """
        kin_energy = self.local_kinetic_energy(params, xs, state_indices)
        pot_energy = self.local_potential_energy(xs)
        local_energy = kin_energy + pot_energy
        return local_energy


class Metropolis:
    """The Metropolis Algorithm
    iterating the positions of MCMC walkers according
    to the Metropolis algorithm.

    Attributes:
        self.wf_ansatz_fock: the callable wave function ansatz
            in fock space
            signature: (params, xs, state_indices)
        self.log_gomain: True for working in log domain.
    """

    def __init__(
        self,
        wf_ansatz_fock: callable,
        log_domain: bool = True,
    ) -> None:
        self.wf_ansatz_fock = wf_ansatz_fock
        self.log_domain = log_domain

    def oneshot_sample(
        self,
        xs: jax.Array,
        state_indices: np.ndarray,
        probability: jax.Array,
        params: jax.Array | np.ndarray | dict,
        step_size: jax.Array,
        key: jax.random.PRNGKey,
    ) -> list[
        jax.Array,
        jax.Array,
        jax.Array,
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
                NOTE: if log_domain == True, then this refers to
                    log probability!
            params: the flow parameter
            step_size: (num_of_orbs,) the step size of each sample step. e.g.
                xs_new = xs + step_size * jax.random.normal(subkey, shape=xs.shape)
            key: the jax PRNG key.

        Returns:
            xs_new: (num_of_orbs,)the updated xs
            probability_new: (num_of_orbs,)the updated probability
            cond: (num_of_orbs,) the accept condition on each orbital,
                for example, for state_indices=[0,1,2]
                after one-shot sample, cond would be
                like [True, False, True]
        """
        key, subkey = jax.random.split(key)

        # vmap wavefunction on different orbitals(states)
        # wf_vmapped = jax.vmap(self.wf_ansatz, in_axes=(None, 0, 0))
        wf_vmapped = self.wf_ansatz_fock
        xs_new = xs + step_size * jax.random.normal(subkey, shape=xs.shape)

        if self.log_domain:
            log_wf_vmapped = wf_vmapped(params, xs_new, state_indices)
            # log probability = log |psi|^2 = 2 Re log |psi|
            probability_new = 2 * log_wf_vmapped
            ratio = probability_new - probability
            # Metropolis
            key, subkey = jax.random.split(key)
            cond = jnp.log(jax.random.uniform(subkey, shape=probability.shape)) < ratio

        else:
            probability_new = wf_vmapped(params, xs_new, state_indices) ** 2
            ratio = probability_new / probability
            # Metropolis
            key, subkey = jax.random.split(key)
            cond = jax.random.uniform(subkey, shape=probability.shape) < ratio
        probability_new = jnp.where(cond, probability_new, probability)
        xs_new = jnp.where(cond, xs_new, xs)

        return xs_new, probability_new, cond


def mcmc(
    steps: int,
    metropolis_sampler_batched: callable,
    key: jax.random.PRNGKey,
    xs_batched: jax.Array,
    state_indices: np.ndarray,
    params: dict,
    probability_batched: jax.Array,
    mc_step_size: jax.Array,
    log_domain: bool,
) -> list[jax.random.PRNGKey, jax.Array, jax.Array, jax.Array, jax.Array]:
    """The batched mcmc function for #steps sampling.
    NOTE: this is a jax foriloop implementation

    Args:
        steps: the steps to perform
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
            the batched probability for each state
                NOTE: if log_domain == True, then this refers to
                    log probability!
        mc_step_size: (num_of_orbs,) last mcmc moving step size.
            NOTE: this is a per orbital property!
        log_domain: True for work in log domain (wavefunction and
            probability)

    Returns:
        key: the jax.PRNGkey
        xs_batched: (num_of_batch,num_of_orbs,) the batched 1D
            coordinate of the particle(s).
            The xs are in corresponding order as in state_indices.
        probability_batched: (num_of_batch,num_of_orbs,)
            the batched probability for each state
                NOTE: if log_domain == True, then this refers to
                    log probability!
        mc_step_size: (num_of_orbs,) updated mcmc moving step size.
            NOTE: this is a per orbital property!
        pmove_per_orb: (num_of_orbs,) the portion of moved particles in last mcmc step.
            NOTE: this is a per orbital property!
    """

    if log_domain:
        print("Note: Working in log domain...")

    batch_size = xs_batched.shape[0]

    @jax.jit
    def _body_func(i, val):
        """MCMC Body function"""
        key, xs_batched, probability_batched, mc_step_size, cond = val
        key, batch_keys = key_batch_split(key, batch_size)
        xs_batched, probability_batched, current_cond = metropolis_sampler_batched(
            xs_batched,
            state_indices,
            probability_batched,
            params,
            mc_step_size,
            batch_keys,
        )
        # cond: (batch,num_of_orbs,)
        cond += current_cond
        return key, xs_batched, probability_batched, mc_step_size, cond

    mcmc_init_val = (
        key,
        xs_batched,
        probability_batched,
        mc_step_size,
        jnp.zeros_like(xs_batched),
    )
    key, xs_batched, probability_batched, mc_step_size, cond = jax.lax.fori_loop(
        0, steps, _body_func, mcmc_init_val
    )

    # pmove_per_orb: (num_of_orb,)
    pmove_per_orb = jnp.mean(cond, axis=0) / steps
    # mc_step_size: (num_of_orb,)
    mc_step_size = jnp.where(
        pmove_per_orb > 0.515,
        mc_step_size * 1.1,
        mc_step_size,
    )
    mc_step_size = jnp.where(
        pmove_per_orb < 0.495,
        mc_step_size * 0.9,
        mc_step_size,
    )
    return key, xs_batched, probability_batched, mc_step_size, pmove_per_orb


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
    wf_ansatz_fock: callable,
    local_energy_estimator: callable,
    state_indices: np.ndarray,
    clip_factor: float | None,
    wf_clip_factor: float | None,
    log_domain=True,
) -> callable:
    """Copied from Ferminet.

    Creates the loss function, including custom gradients.
    NOTE: batched implementation!

    Args:
    wf_ansatz_fock: function, signature (params, xs, state_indices),
        which evaluates wavefunction in fock space
        at a single MCMC configuration given the network parameters.
    local_energy_estimator: signature (params, xs, state_indices), here
        xs refers to the x coordinates in one single batch, xs has shape (num_of_orb,)
    state_indices: (num_of_orbs,)the array containing
        the excitations in each bath, for example,
        [0,1,2] represents there are totally 3 number
        of orbitals: ground state: 0
                    first excitation state: 1
                    second excitation state: 2
    clip_factor: the gradient clipping factor for energy part. None for
        not making energy clipping.
    wf_clip_factor: the gradient clipping factor for wavefunction part. None
        for not making wavefunction clipping.
    log_domain: True for work in log domain (wavefunction and
            probability)

    Returns:
        total_energy: function which evaluates the both the loss as the mean of the
        local_energies, and the local energies. With custom gradients, accounting
        for derivatives in the probability distribution.
    """
    if wf_clip_factor:
        raise NotImplementedError("wavefunction clip not implemented!")

    batch_local_energy = jax.vmap(
        local_energy_estimator, in_axes=(None, 0, None), out_axes=0
    )

    # if log_domain:
    #     log_wf = wf_ansatz
    # else:
    #     log_wf = lambda params, x, n: jnp.log(jnp.abs(wf_ansatz(params, x, n)))
    # # vmapped wavefunction: signature (params, xs, state_indices)
    # log_wf_vmapped = jax.vmap(log_wf, in_axes=(None, 0, 0))
    # # batch_wf: signature (params, batched_xs, state_indices)
    # batch_wf = jax.vmap(log_wf_vmapped, in_axes=(None, 0, None), out_axes=0)

    if log_domain:
        log_wf_fock = wf_ansatz_fock
    else:
        log_wf_fock = lambda params, xs, state_indices: jnp.log(
            jnp.abs(wf_ansatz_fock(params, xs, state_indices))
        )
    batch_wf = jax.vmap(log_wf_fock, in_axes=(None, 0, None), out_axes=0)

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
        # only mean over batch dimension
        energy_expectation_per_orb = jnp.mean(
            local_energy,
            axis=0,
        )

        if clip_factor:
            energy_clip_tv = jnp.mean(
                jnp.abs(local_energy - energy_expectation_per_orb), axis=0
            )
            local_energy = jnp.clip(
                local_energy,
                energy_expectation_per_orb - clip_factor * energy_clip_tv,
                energy_expectation_per_orb + clip_factor * energy_clip_tv,
            )

        diff = local_energy - energy_expectation_per_orb

        def _batch_wf(params, batched_xs):
            return batch_wf(params, batched_xs, state_indices)

        # local_energy: (batch, num_of_orbs)
        # diff: (batch, num_of_orbs)
        psi_primal, psi_tangent = jax.jvp(_batch_wf, primals, tangents)
        primals_out = loss, local_energy
        tangents_out = (
            2 * jnp.sum(jnp.mean(psi_tangent * diff, axis=0)),
            local_energy,
        )

        return primals_out, tangents_out

    return total_energy


class Loss:
    """The original total loss function
    as 'it is' and the custom loss function, custom_loss
    ONLY for Gradient Estimator!

    Attributes:
        self.state_indices:  (num_of_orbs,)the array containing
                the excitations in each bath, for example,
                [0,1,2] represents there are totally 3 number
                of orbitals: ground state: 0
                            first excitation state: 1
                            second excitation state: 2
        self.batch_local_energy: the batched local energy estimator.
        self.log_wf_fock: the log domain wavefunction callable
            in fock space, aka, concatenated on ORBITALS!
        self.batch_wf: the logg domain wavefunction callable
            vmapped one BATCH!
        self.clip_factor: the gradient clipping factor for energy part. None for
            not making energy clipping.
        self.wf_clip_factor: the gradient clipping factor for wavefunction part. None
            for not making wavefunction clipping.
    """

    def __init__(
        self,
        wf_ansatz_fock: callable,
        local_energy_estimator: callable,
        state_indices: np.ndarray,
        clip_factor: float | None,
        wf_clip_factor: float | None,
        log_domain=True,
    ) -> None:
        """Init

        Args:
            wf_ansatz_fock: function, signature (params, xs, state_indices),
                which evaluates wavefunction in fock space
                at a single MCMC configuration given the network parameters.
            local_energy_estimator: signature (params, xs, state_indices), here
                xs refers to the x coordinates in one single batch, xs has shape
                (num_of_orb,)
            state_indices: (num_of_orbs,)the array containing
                the excitations in each bath, for example,
                [0,1,2] represents there are totally 3 number
                of orbitals: ground state: 0
                            first excitation state: 1
                            second excitation state: 2
            clip_factor: the gradient clipping factor for energy part. None for
                not making energy clipping.
            wf_clip_factor: the gradient clipping factor for wavefunction part. None
                for not making wavefunction clipping.
            log_domain: True for work in log domain (wavefunction and
                    probability)
        """
        self.state_indices = state_indices
        self.batch_local_energy = jax.vmap(
            local_energy_estimator, in_axes=(None, 0, None), out_axes=0
        )

        # if log_domain:
        #     log_wf = wf_ansatz
        # else:
        #     log_wf = lambda params, x, n: jnp.log(jnp.abs(wf_ansatz(params, x, n)))
        # # vmapped wavefunction: signature (params, xs, state_indices)
        # self.log_wf_vmapped = jax.vmap(log_wf, in_axes=(None, 0, 0))
        # # batch_wf: signature (params, batched_xs, state_indices)
        # self.batch_wf = jax.vmap(
        #     self.log_wf_vmapped, in_axes=(None, 0, None), out_axes=0
        # )

        if log_domain:
            self.log_wf_fock = wf_ansatz_fock
        else:
            self.log_wf_fock = lambda params, xs, state_indices: jnp.log(
                jnp.abs(wf_ansatz_fock(params, xs, state_indices))
            )
        # batch_wf: signature (params, batched_xs, state_indices)
        self.batch_wf = jax.vmap(self.log_wf_fock, in_axes=(None, 0, None), out_axes=0)

        self.clip_factor = clip_factor
        self.wf_clip_factpr = wf_clip_factor
        if wf_clip_factor:
            raise NotImplementedError(
                "Wavefunction clip factor for Hydrogen Type loss not implemented!"
            )

    def total_energy(
        self, params: dict, batched_xs: jax.Array
    ) -> tuple[float, np.ndarray]:
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
        e_l = self.batch_local_energy(params, batched_xs, self.state_indices)
        loss = jnp.mean(jnp.sum(e_l, axis=-1))
        return loss, e_l

    def custom_loss(self, params: dict, batched_xs: jax.Array) -> jax.Array:
        """The custom loss function
        Then the gradient towards params could be
        accessed by DIRECTLY making grad to this function!
        NOTE: For Gradient estimator ONLY!

        Args:
            params: the network parameters
            batched_xs: (num_of_batch, num_of_orbs,)
                the batched 1D coordinate of the particle(s).
                The xs are in corresponding order as in state_indices.

        Returns:
            custom_loss: the custom defined loss.
        """

        def _batch_wf(params, batched_xs):
            return self.batch_wf(params, batched_xs, self.state_indices)

        # local_energis: (batch, num_of_orbs)
        loss, local_energies = jax.lax.stop_gradient(
            self.total_energy(params, batched_xs)
        )
        # logpsix: (batch, num_of_orbs)
        logpsix = _batch_wf(params, batched_xs)
        energies_batch_average = jnp.mean(local_energies, axis=0)  # (num_of_orbs,)

        if self.clip_factor:
            # For Control Variate and clipping
            # Clipping may be important for nodal area!
            clip_factor = self.clip_factor
            tv = jnp.mean(
                jnp.abs(local_energies - energies_batch_average), axis=0
            )  # (num_of_orbs,)
            local_energies_clipped = jnp.clip(
                local_energies,
                energies_batch_average - clip_factor * tv,
                energies_batch_average + clip_factor * tv,
            )
            custom_loss = 2 * jnp.sum(
                jnp.mean(
                    (logpsix * (local_energies_clipped - energies_batch_average)),
                    axis=0,
                )
            )
        else:
            custom_loss = 2 * jnp.sum(
                jnp.mean((logpsix * (local_energies - energies_batch_average)), axis=0)
            )
        return custom_loss

    def loss_and_grad(
        self,
        params: dict,
        batched_xs: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], dict]:
        """Manually implemented loss_and_grad to
        compatible with previous FermiNet style loss,
        jax.value_and_grad(loss_energy, argnums=0, has_aux=True)

        Args:
            params: the network parameters
            batched_xs: (num_of_batch, num_of_orbs,)
                the batched 1D coordinate of the particle(s).
                The xs are in corresponding order as in state_indices.

        Returns:
            loss: the total loss
            energies: (batch, num_of_orbs) the local energies
            gradients: dict, the gradients to network parameters.
        """
        gradients = jax.grad(self.custom_loss, argnums=0)(params, batched_xs)
        loss, energies = jax.lax.stop_gradient(self.total_energy(params, batched_xs))
        return ((loss, energies), gradients)


class Update:
    """Update functions"""

    def __init__(
        self,
        mcmc_steps: int,
        state_indices: np.ndarray,
        batch_size: int,
        metropolis_sampler_batched: callable,
        loss_and_grad: callable,
        optimizer: optax.GradientTransformation,
        acc_steps: int,
        log_domain: bool = True,
    ) -> None:
        self.mcmc_steps = mcmc_steps
        self.state_indices = state_indices
        self.batch_size = batch_size
        self.metropolis_sampler_batched = metropolis_sampler_batched
        self.loss_and_grad = loss_and_grad
        self.optimizer = optimizer
        self.acc_steps = acc_steps
        self.log_domain = log_domain

    def update(
        self,
        key: jax.random.PRNGKey,
        xs_batched: jax.Array,
        probability_batched: jax.Array,
        mc_step_size: jax.Array,
        params: jax.Array | np.ndarray | dict,
        opt_state: optax.OptState,
    ) -> tuple[
        float | jax.Array,
        float | jax.Array,
        jax.Array,
        jax.Array,
        jax.Array | dict,
        optax.OptState,
        jax.Array,
        jax.Array,
    ]:
        """Single update
        NOTE: call WITHOUT vmap!

        Args:
            key: the jax.PRNGkey
            xs_batched: (num_of_batch,num_of_orbs,) the batched 1D coordinate
                of the particle(s).
                The xs are in corresponding order as in state_indices.
            probability_batched: (num_of_batch,num_of_orbs,)
                the batched probability for each state (wavefunction**2)
            mc_step_size: (num_of_orbs,) last mcmc moving step size.
                NOTE: this is a per orbital property!
            params: the flow parameters
            opt_state: the optimizer state.

        Returns:
            loss: the meaned energy as loss
            energy_std: the energy standard deviation.
            xs_batched: the coordinate
            probability_batched: the batched probability (wavefunction**2)
            params: the flow parameters
            opt_state: the optimizer state.
            pmove_per_orb: (num_of_orbs,) the portion of moved
                particles in last mcmc step.
                NOTE: this is a per orbital property!
            mc_step_size: (num_of_orbs,) updated mcmc moving step size.
                NOTE: this is a per orbital property!

        """

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
                pmove_in,
                params,
            ) = val
            (
                key,
                xs_batched,
                probability_batched,
                mc_step_size,
                pmove_per_orb,
            ) = mcmc(
                steps=self.mcmc_steps,
                metropolis_sampler_batched=self.metropolis_sampler_batched,
                key=key,
                xs_batched=xs_batched,
                state_indices=self.state_indices,
                params=params,
                probability_batched=probability_batched,
                mc_step_size=mc_step_size,
                log_domain=self.log_domain,
            )
            (loss_i, energies_i), gradients_i = self.loss_and_grad(params, xs_batched)
            # grad = jax.tree.map(jnp.mean, gradients)
            grad_i = gradients_i
            (loss, energies, grad) = jax.tree.map(
                lambda acc, i: acc + i,
                (loss, energies, grad),
                (loss_i, energies_i, grad_i),
            )
            return (
                loss,
                energies,
                grad,
                key,
                xs_batched,
                probability_batched,
                mc_step_size,
                pmove_per_orb,
                params,
            )

        # Only for initialization of val in jax.lax.foriloop!
        pmove_dummy = jnp.zeros_like(self.state_indices, dtype=jnp.float64)
        (loss, energies), gradients = self.loss_and_grad(params, xs_batched)
        loss = jnp.float64(0.0)
        energies = jnp.zeros_like(energies)
        gradients = jax.tree.map(jnp.zeros_like, gradients)
        # grad = jax.tree.map(jnp.mean, gradients)
        grad = gradients
        acc_init_val = (
            loss,
            energies,
            grad,
            key,
            xs_batched,
            probability_batched,
            mc_step_size,
            pmove_dummy,
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
            pmove_per_orb,
            params,
        ) = jax.lax.fori_loop(0, self.acc_steps, _acc_body_func, acc_init_val)

        (loss, energies, grad) = jax.tree.map(
            lambda acc: acc / self.acc_steps, (loss, energies, grad)
        )

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
            pmove_per_orb,
            mc_step_size,
        )


class MLPFlow(flax_nn.Module):
    """A simple MLP flow"""

    out_dims: int
    mlp_width: int
    mlp_depth: int

    @flax_nn.compact
    def __call__(self, xs):
        _init_xs = xs
        pre_factors = []
        xs_new = []
        for x in xs:
            for i in range(self.mlp_depth):
                x = x.reshape((1,))
                x = flax_nn.Dense(self.mlp_depth)(x)
                x = flax_nn.sigmoid(x)
                x = flax_nn.Dense(1)(x)
            prefactor = flax_nn.sigmoid(x)
            xs_new.append(x.reshape(-1))
            pre_factors.append(prefactor.reshape(-1))
        xs = jnp.array(xs_new).reshape(_init_xs.shape)
        pre_factors = jnp.array(pre_factors).reshape(_init_xs.shape)
        xs = _init_xs + xs
        return xs, pre_factors


class MLPSingleFlow(flax_nn.Module):
    """A simple MLP flow"""

    out_dims: int
    mlp_width: int
    mlp_depth: int

    def single_state(self, x):
        """Single state flow"""
        x = x.reshape((1,))
        x = flax_nn.Dense(self.mlp_depth)(x)
        x = flax_nn.sigmoid(x)
        x = flax_nn.Dense(1)(x)
        return x.reshape(-1)

    @flax_nn.compact
    def __call__(self, x):
        for i in range(self.mlp_depth):
            _init_x = x
            x = self.single_state(x)
            x = _init_x + x
        return x


def training_kernel(args: dict, savefig: bool = True) -> None:
    """Training Kernel"""

    # Initialize flow
    key = args["key"]
    batch_size = args["batch_size"]
    state_indices = args["state_indices"]
    thermal_step = args["thermal_step"]
    acc_steps = args["acc_steps"]
    mc_steps = args["mc_steps"]
    step_size = args["step_size"]
    init_width = args["init_width"]
    mlp_width = args["mlp_width"]
    mlp_depth = args["mlp_depth"]
    init_learning_rate = args["init_learning_rate"]
    iterations = args["iterations"]
    inference_batch_size = args["inference_batch_size"]
    inference_thermal_step = args["inference_thermal_step"]
    figure_save_path = args["figure_save_path"]
    log_domain = args["log_domain"]
    ferminet_loss = args["ferminet_loss"]
    clip_factor = args["clip_factor"]
    wf_clip_factor = args["wf_clip_factor"]

    print("======================================")
    print(f"Computed States Indices: {state_indices}")
    print("======================================")

    state_indices = np.array(state_indices)
    step_size = step_size * jnp.ones_like(state_indices, dtype=jnp.float64)

    if savefig:
        figure_save_path = os.path.join(
            figure_save_path,
            f"batch_{batch_size}/",
            f"mlpwid_{mlp_width}/",
            f"mlpdep_{mlp_depth}/",
            f"initlr_{init_learning_rate:.4f}/",
            f"mcstp_{mc_steps}/",
            f"thrstp_{thermal_step}/",
            f"acc_{acc_steps}/",
            f"xinitwidth_{init_width}/",
        )

        if clip_factor:
            figure_save_path = os.path.join(figure_save_path, f"clip_{clip_factor}/")
        else:
            figure_save_path = os.path.join(figure_save_path, "NoGradientClip/")

        if log_domain:
            figure_save_path = os.path.join(figure_save_path, "log-domain/")
        else:
            figure_save_path = os.path.join(figure_save_path, "real-domain/")

        if ferminet_loss:
            figure_save_path = os.path.join(figure_save_path, "FermiNetLoss/")
        else:
            figure_save_path = os.path.join(figure_save_path, "HydrogenLoss/")
        if wf_clip_factor:
            raise NotImplementedError("wavefunction clip not implemented!")

        if not os.path.isdir(figure_save_path):
            print(f"Creating Directory at {figure_save_path}")
            os.makedirs(figure_save_path)
        else:
            # raise FileExistsError("Desitination already exists! Please check!")
            pass

    model_flow = MLPFlow(
        out_dims=state_indices.shape[0], mlp_width=mlp_width, mlp_depth=mlp_depth
    )
    key, subkey = jax.random.split(key)
    x_dummy = jnp.zeros(state_indices.shape[0], dtype=jnp.float64)
    key, subkey = jax.random.split(key)
    params = model_flow.init(subkey, x_dummy)
    params = jax.tree.map(
        lambda leaf: 0.5 * jax.random.truncated_normal(subkey, -2.0, 2.0, leaf.shape),
        params,
    )

    # Initial Jacobian
    def _flow_only_return_xs(xs):
        result_xs, _ = model_flow.apply(params, xs)
        return result_xs

    init_jacobian = jax.jacfwd(_flow_only_return_xs)(x_dummy)
    if (jnp.diagonal(jnp.abs(init_jacobian - 1.0)) > 0.2).any():
        raise ValueError(
            "Init Jacobian too far from identity!\n"
            "Get init Jacobian for each state in fock"
            f"space {state_indices}={jnp.diagonal(init_jacobian)}\n"
        )
    print(
        "Init Jacobian for each state in fock space"
        f" {state_indices}= \n{jnp.diagonal(init_jacobian)}"
    )

    # Initialize Wavefunction
    wf_ansatz_obj = WFAnsatz(flow=model_flow)
    if log_domain:
        wf_ansatz_fock = wf_ansatz_obj.log_wf_ansatz_fock
    else:
        wf_ansatz_fock = wf_ansatz_obj.wf_ansatz_fock

    # Plotting wavefunction
    print("Wavefunction (Initialization)")
    xmin = -15
    xmax = 15
    Nmesh = 5000
    xmesh = np.linspace(xmin, xmax, Nmesh, dtype=np.float64)
    mesh_interval = xmesh[1] - xmesh[0]
    plt.figure()

    xmesh_fock = np.tile(xmesh, state_indices.shape[0]).reshape(
        state_indices.shape[0], Nmesh
    )
    wfs_on_mesh = jax.vmap(
        wf_ansatz_fock,
        in_axes=(None, 1, None),
        out_axes=0,
    )(params, xmesh_fock, state_indices)
    for i in state_indices:
        wf_on_mesh = wfs_on_mesh[:, i]
        plt.plot(xmesh, wf_on_mesh, label=f"n={i}")
    plt.xlim([-10, 10])
    if log_domain:
        plt.ylabel("log |psi|")
    else:
        plt.ylabel("psi")
    plt.legend()
    plt.title("Wavefunction (Initialization)")
    if savefig:
        plt.savefig(
            f"{os.path.join(figure_save_path, "WavefunctionInitialization.png")}"
        )
        print("Figure Saved.")
    else:
        plt.show()
    plt.close()

    # Local Energy Estimator
    energy_estimator = EnergyEstimator(
        wf_ansatz_fock=wf_ansatz_fock, log_domain=log_domain
    )

    # Metropolis, thermalization
    key, subkey = jax.random.split(key)
    init_x_batched = init_batched_x(
        key=subkey,
        batch_size=batch_size,
        num_of_orbs=state_indices.shape[0],
        init_width=init_width,
    )
    metropolis = Metropolis(wf_ansatz_fock=wf_ansatz_fock, log_domain=log_domain)
    metropolis_sample = metropolis.oneshot_sample
    metropolis_sample_batched = jax.jit(
        jax.vmap(metropolis_sample, in_axes=(0, None, 0, None, None, 0))
    )
    if log_domain:  # log probabilities
        probability_batched = (
            jax.vmap(wf_ansatz_fock, in_axes=(None, 0, None))(
                params, init_x_batched, state_indices
            )
            * 2
        )
    else:
        probability_batched = (
            jax.vmap(wf_ansatz_fock, in_axes=(None, 0, None))(
                params, init_x_batched, state_indices
            )
            ** 2
        )
    xs = init_x_batched

    print("Thermalization...")
    t0 = time.time()
    key, xs, probability_batched, step_size, pmove_per_orb = mcmc(
        steps=thermal_step,
        metropolis_sampler_batched=metropolis_sample_batched,
        key=key,
        xs_batched=xs,
        state_indices=state_indices,
        params=params,
        probability_batched=probability_batched,
        mc_step_size=step_size,
        log_domain=log_domain,
    )
    t1 = time.time()
    time_cost = t1 - t0
    print(
        f"After Thermalization:\tpmove {pmove_per_orb}\t"
        f"step_size={step_size}\ttime={time_cost:.2f}s",
        flush=True,
    )

    if ferminet_loss:
        # Loss function, FermiNet Style
        loss_energy = make_loss(
            wf_ansatz_fock=wf_ansatz_fock,
            local_energy_estimator=energy_estimator.local_energy,
            state_indices=state_indices,
            clip_factor=clip_factor,
            wf_clip_factor=wf_clip_factor,
            log_domain=log_domain,
        )
        loss_and_grad = jax.jit(
            jax.value_and_grad(loss_energy, argnums=0, has_aux=True)
        )
    else:
        # Loss function, as in Hydrogen
        loss_obj = Loss(
            wf_ansatz_fock=wf_ansatz_fock,
            local_energy_estimator=energy_estimator.local_energy,
            state_indices=state_indices,
            clip_factor=clip_factor,
            wf_clip_factor=wf_clip_factor,
            log_domain=log_domain,
        )
        loss_energy = loss_obj.total_energy
        loss_and_grad = jax.jit(loss_obj.loss_and_grad)

    (loss, energies), gradients = loss_and_grad(params, xs)
    print(f"After MCMC, with initial network, loss={loss:.2f}")

    # Optimizer
    optimizer = optax.adam(init_learning_rate)
    opt_state = optimizer.init(params)

    # Training
    update_obj = Update(
        mcmc_steps=mc_steps,
        state_indices=state_indices,
        batch_size=batch_size,
        metropolis_sampler_batched=metropolis_sample_batched,
        loss_and_grad=loss_and_grad,
        optimizer=optimizer,
        acc_steps=acc_steps,
        log_domain=log_domain,
    )
    update = jax.jit(update_obj.update)
    loss_energy_list = []
    energy_std_list = []
    if savefig:  # not in notebook
        for i in range(iterations):
            t0 = time.time()
            key, subkey = jax.random.split(key)
            (
                loss_energy,
                energy_std,
                xs,
                probability_batched,
                params,
                opt_state,
                pmove_per_orb,
                step_size,
            ) = update(
                subkey,
                xs,
                probability_batched,
                step_size,
                params,
                opt_state,
            )
            t1 = time.time()
            print(
                f"Iter:{i}, LossE={loss_energy:.5f}({energy_std:.5f})"
                f"\t pmove={pmove_per_orb}\tstepsize={step_size}"
                f"\t time={(t1-t0):.3f}s"
            )
            loss_energy_list.append(loss_energy)
            energy_std_list.append(energy_std)
    else:  # in notebook
        for i in tqdm.tqdm(range(iterations)):
            key, subkey = jax.random.split(key)
            (
                loss_energy,
                energy_std,
                xs,
                probability_batched,
                params,
                opt_state,
                pmove_per_orb,
                step_size,
            ) = update(
                subkey,
                xs,
                probability_batched,
                step_size,
                params,
                opt_state,
            )
            loss_energy_list.append(loss_energy)
            energy_std_list.append(energy_std)

    # Finite Differential Method
    # which would be <exact> if our mesh intervals are small enough
    potential_func = energy_estimator.local_potential_energy
    H = buildH(potential_func, xmesh, Nmesh, mesh_interval)
    exact_eigenvalues, exact_eigenvectors = jnp.linalg.eigh(H)

    # Training Curve
    trained_states_energy = exact_eigenvalues[state_indices]
    expectedenergy = np.sum(trained_states_energy)
    plot_step = 10
    fig, axs = plt.subplots(2, 1, figsize=(8, 12))
    axs[0].errorbar(
        range(iterations)[::plot_step],
        loss_energy_list[::plot_step],
        energy_std_list[::plot_step],
        label="loss",
        fmt="o",
        markersize=1,
        capsize=0.4,
        elinewidth=0.3,
        linestyle="none",
    )
    axs[0].plot([0, iterations], [expectedenergy, expectedenergy], label="ref")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss,E")
    axs[0].set_ylim([4 * expectedenergy / 5, 2.5 * expectedenergy])
    axs[0].set_xscale("log")
    axs[1].errorbar(
        range(iterations)[-500::],
        loss_energy_list[-500::],
        energy_std_list[-500::],
        label="loss",
        fmt="o",
        markersize=1,
        capsize=0.4,
        elinewidth=0.3,
        linestyle="none",
    )
    axs[1].plot(
        [iterations - 500, iterations], [expectedenergy, expectedenergy], label="ref"
    )
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss,E")
    axs[1].set_ylim([4 * expectedenergy / 5, 1.5 * expectedenergy])
    axs[1].set_title("Last 500 iters zoom in")
    plt.legend()
    if savefig:
        plt.savefig(f"{os.path.join(figure_save_path, "Training.png")}")
        print("Figure Saved.")
    else:
        plt.show()

    # Plotting Potential
    print("Plotting Potential")
    plt.figure()
    plt.plot(xmesh, potential_func(xmesh), "k-", lw=2, label="Potential")
    plt.xlim([-2, 2])
    plt.ylim([-2, 10])
    plt.legend()
    plt.title("Potential")
    if savefig:
        plt.savefig(f"{os.path.join(figure_save_path, "Potential.png")}")
        print("Figure Saved.")
    else:
        plt.show()
    plt.close()

    # Finally Plotting Probability Density
    print("Probability Density (Trained)")
    plt.figure()
    wfs_on_mesh = jax.vmap(
        wf_ansatz_fock,
        in_axes=(None, 1, None),
        out_axes=0,
    )(params, xmesh_fock, state_indices)
    for i in state_indices:
        wf_on_mesh = wfs_on_mesh[:, i]
        if log_domain:
            normalize_factor = (jnp.exp(wf_on_mesh * 2)).sum() * mesh_interval
            prob_density = jnp.exp(wf_on_mesh * 2) / normalize_factor
        else:
            normalize_factor = (wf_on_mesh**2).sum() * mesh_interval
            prob_density = wf_on_mesh**2 / normalize_factor
        plt.plot(xmesh, prob_density, label=f"VMC-n={i}", lw=2)

        exact_wf_on_mesh = exact_eigenvectors[:, i]
        normalize_factor = (exact_wf_on_mesh**2).sum() * mesh_interval
        exact_prob_density = exact_wf_on_mesh**2 / normalize_factor
        plt.plot(xmesh, exact_prob_density, "-.", label=f"Exact-n={i}", lw=1.75)
    plt.xlim([-2.0, 2.0])
    plt.legend()
    plt.ylabel(r"$\rho$")
    plt.title("Probability Density (Trained and Compare)")
    if savefig:
        plt.savefig(
            f"{os.path.join(figure_save_path, "ProbabilityDensityAfterTraining.png")}"
        )
        print("Figure Saved.")
    else:
        plt.show()
    plt.close()

    # Inference
    print("...Inferencing...")
    inference_init_width = 3.0
    print(f"Inferencing init width = {inference_init_width}")
    key, subkey = jax.random.split(key)
    xs_inference = init_batched_x(
        key=subkey,
        batch_size=inference_batch_size,
        num_of_orbs=state_indices.shape[0],
        init_width=inference_init_width,
    )
    if log_domain:
        probability_batched = (
            jax.vmap(wf_ansatz_fock, in_axes=(None, 0, None))(
                params, xs_inference, state_indices
            )
            * 2
        )
    else:
        probability_batched = (
            jax.vmap(wf_ansatz_fock, in_axes=(None, 0, None))(
                params, xs_inference, state_indices
            )
            ** 2
        )
    print("Thermalization...")
    t0 = time.time()
    key, xs_inference, probability_batched, step_size, pmove_per_orb = mcmc(
        steps=inference_thermal_step,
        metropolis_sampler_batched=metropolis_sample_batched,
        key=key,
        xs_batched=xs_inference,
        state_indices=state_indices,
        params=params,
        probability_batched=probability_batched,
        mc_step_size=step_size,
        log_domain=log_domain,
    )
    t1 = time.time()
    time_cost = t1 - t0
    print(
        f"After Thermalization:\tpmove {pmove_per_orb}\t"
        f"step_size={step_size}\ttime={time_cost:.2f}s",
        flush=True,
    )

    print("Mesuring Energy...")
    (_, energies_batched), _ = loss_and_grad(params, xs_inference)
    energy_levels = jnp.mean(energies_batched, axis=0)
    energy_levels_std = jnp.sqrt(
        (jnp.mean(energies_batched**2, axis=0) - energy_levels**2)
        / (acc_steps * batch_size)
    )

    if savefig:
        filepath = os.path.join(figure_save_path, "EnergyLevels.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 50)
            f.write("\n")
            f.write(f"VMC Result with {state_indices} states:\n")
            for i, energyi, stdi in zip(
                state_indices, energy_levels, energy_levels_std
            ):
                f.write(f"n={i}\tenergy={energyi:.5f}({stdi:.5f})\n")
            f.write("=" * 50)
            f.write("\n")
            f.write(f"Exact Result with {state_indices} states:\n")
            for i, energyi in zip(state_indices, exact_eigenvalues[state_indices]):
                f.write(f"n={i}\tenergy={energyi:.5f}\n")
        print(f"Energy levels written to {filepath}")
    else:
        print("=" * 50, "\n")
        print(f"VMC Result with {state_indices} states:\n")
        for i, energyi, stdi in zip(state_indices, energy_levels, energy_levels_std):
            print(f"n={i}\tenergy={energyi:.5f}({stdi[i]:.5f})\n")
        print("=" * 50, "\n")
        print(f"Exact Result with {state_indices} states:\n")
        for energyi in exact_eigenvalues[state_indices]:
            print(f"n={i}\tenergy={energyi:.5f}\n")

    print("======================================")
    print(f"Computed States Indices: {state_indices}")
    print("======================================")
