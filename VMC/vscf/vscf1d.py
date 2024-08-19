"""1 Dimensional VSCF"""

# %%
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy
import jax
import jax.numpy as jnp

import sys

sys.path.append("../../")
from VMC.utils import wf_base_indices_vmapped
from VMC.utils import buildH
from VMC.utils import EnergyEstimator as VMCEnergyEstimator

# Plotting Settings
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["figure.dpi"] = 600


class IntegrealList:
    """The integral list of the eigenfunctions
    of harmonic oscillators
    NOTE: the |psi>s are the eigenfunctions of the
        harmonic oscillators
    """

    def __init__(self, nlevels: int) -> None:
        """Init integral lists

        Args:
            nlevels: the total number of levels of
                uncoupled harmonic oscillator's
                eigenfunctions to be used in VSCF.
        """
        self.psi_q0_psi = self.get_psi_q0_psi(nlevels)
        self.psi_q1_psi = self.get_psi_q1_psi(nlevels)
        self.psi_q2_psi = self.get_psi_q2_psi(nlevels)
        self.psi_q3_psi = self.get_psi_q3_psi(nlevels)
        self.psi_q4_psi = self.get_psi_q4_psi(nlevels)
        self.psi_kinetic_psi = self.get_psi_kinetic_psi(nlevels)

    def _delta(self, i: int, j: int) -> int | np.ndarray:
        """Self defined Dirac delta function"""
        return np.where(i == j, 1, 0)

    def get_psi_q0_psi(self, nlevels: int):
        """<psi|psi>"""
        return np.eye(nlevels, dtype=np.float64)

    def get_psi_q1_psi(self, nlevels: int):
        """<psi|Q|psi>"""
        psi_q1_psi = np.zeros((nlevels, nlevels))
        for m in range(nlevels):
            for n in range(nlevels):
                psi_q1_psi[m, n] = self._delta(m, n + 1) * np.sqrt(m / 2) + self._delta(
                    m, n - 1
                ) * np.sqrt((m + 1) / 2)
        return np.array(psi_q1_psi, dtype=np.float64)

    def get_psi_q2_psi(self, nlevels: int):
        """<psi|Q^2|psi>"""
        psi_q2_psi = np.zeros((nlevels, nlevels))
        for m in range(nlevels):
            for n in range(nlevels):
                psi_q2_psi[m, n] = (
                    self._delta(m, n + 2) * (1 / 2) * np.sqrt((m - 1) * m)
                    + self._delta(m, n) * (m + 1 / 2)
                    + self._delta(m, n - 2) * (1 / 2) * np.sqrt((m + 1) * (m + 2))
                )
        return np.array(psi_q2_psi, dtype=np.float64)

    def get_psi_q3_psi(self, nlevels: int):
        """<psi|Q^3|psi>"""
        psi_q3_psi = np.zeros((nlevels, nlevels))
        for m in range(nlevels):
            for n in range(nlevels):
                psi_q3_psi[m, n] = (
                    self._delta(m, n + 3)
                    * (1 / (2 * np.sqrt(2)))
                    * np.sqrt((m - 2) * (m - 1) * m)
                    + self._delta(m, n + 1) * (1 / (2 * np.sqrt(2))) * 3 * np.sqrt(m**3)
                    + self._delta(m, n - 1)
                    * (1 / (2 * np.sqrt(2)))
                    * 3
                    * np.sqrt((m + 1) ** 3)
                    + self._delta(m, n - 3)
                    * (1 / (2 * np.sqrt(2)))
                    * np.sqrt((m + 1) * (m + 2) * (m + 3))
                )
        return np.array(psi_q3_psi, dtype=np.float64)

    def get_psi_q4_psi(self, nlevels: int):
        """<psi|Q^4|psi>"""
        psi_q4_psi = np.zeros((nlevels, nlevels))
        for m in range(nlevels):
            for n in range(nlevels):
                psi_q4_psi[m, n] = (
                    self._delta(m, n + 4)
                    * (1 / 4)
                    * np.sqrt((m - 3) * (m - 2) * (m - 1) * m)
                    + self._delta(m, n + 2)
                    * (1 / 2)
                    * (2 * m - 1)
                    * np.sqrt((m - 1) * m)
                    + self._delta(m, n) * (3 / 4) * (2 * m**2 + 2 * m + 1)
                    + self._delta(m, n - 2)
                    * (1 / 2)
                    * (2 * m + 3)
                    * np.sqrt((m + 1) * (m + 2))
                    + self._delta(m, n - 4)
                    * (1 / 4)
                    * np.sqrt((m + 1) * (m + 2) * (m + 3) * (m + 4))
                )
        return np.array(psi_q4_psi, dtype=np.float64)

    def get_psi_kinetic_psi(self, nlevels: int):
        """<psi|T|psi>"""
        psi_kinetic_psi = np.zeros((nlevels, nlevels))
        for m in range(nlevels):
            for n in range(nlevels):
                psi_kinetic_psi[m, n] = (
                    (-1 / 2) * self._delta(m, n + 2) * (1 / 2) * np.sqrt((m - 1) * m)
                    + (1 / 2) * self._delta(m, n) * (m + 1 / 2)
                    + (-1 / 2)
                    * self._delta(m, n - 2)
                    * (1 / 2)
                    * np.sqrt((m + 1) * (m + 2))
                )
        return np.array(psi_kinetic_psi, dtype=np.float64)


def _wf_base_vscf(
    x: float | jax.Array,
    n: int,
    coeff: np.ndarray,
) -> jax.Array:
    """The wave function ansatz (Rotated Gaussian)
    NOTE: Rotated! Rotation is solved by VSCF
    NOTE: Rotated total states should be larger than called argument n.

    Args:
        x: the 1D coordinate of the (single) particle.
        n: the excitation quantum number
        coeff: (nlevel,nlevel) the solved VSCF coefficients.

        NOTE: n=0 for GS!

    Returns:
        psi: the probability amplitude at x.
    """
    nlevel = coeff.shape[0]
    mix_indices = jnp.array(range(nlevel))
    mix_rot_coeff = coeff[:, n]
    wf_for_mix = wf_base_indices_vmapped(x, mix_indices)
    psi = jnp.dot(mix_rot_coeff, wf_for_mix)
    return psi


def main():
    """Man"""

    xmin = -10
    xmax = 10
    Nmesh = 2000
    xmesh = jnp.linspace(xmin, xmax, Nmesh)
    interval = xmesh[1] - xmesh[0]

    nlevel = 30

    def wf_gs(coeff, x):
        return _wf_base_vscf(x, 0, coeff)

    def wf_1st(coeff, x):
        return _wf_base_vscf(x, 1, coeff)

    wf_gs_vmapped = jax.vmap(wf_gs, in_axes=(None, 0))
    wf_first_vmapped = jax.vmap(wf_1st, in_axes=(None, 0))

    print(f"VSCF with Nlevel = {nlevel}")
    integrallist = IntegrealList(nlevels=nlevel)
    kinetic = integrallist.psi_kinetic_psi
    secondterm = integrallist.psi_q2_psi
    thirdterm = integrallist.psi_q3_psi
    quarticterm = integrallist.psi_q4_psi
    hamiltonian = kinetic - 3 * secondterm + thirdterm / 2 + 3 * quarticterm
    energies, coeff = np.linalg.eigh(hamiltonian)
    totalen = np.sum(energies[:2:])

    print("=" * 50)
    print("Saving Coefficients")
    filename = "VSCFCoeff.pkl"
    with open(filename, "wb") as f:
        pickle.dump(coeff, f)
    print("Done")

    # Finite Differential Method
    # which would be <exact> if our mesh intervals are small enough
    potential_func = VMCEnergyEstimator.local_potential_energy
    H = buildH(potential_func, xmesh, Nmesh, interval)
    exact_eigenvalues, exact_eigenvectors = scipy.linalg.eigh(H)
    trained_states_energy = exact_eigenvalues[range(2)]
    expectedenergy = np.sum(trained_states_energy)
    print(
        f"After optimization,\nTotal energy = {totalen}\nRef Energy = {expectedenergy}"
    )

    plt.figure()
    for i in range(2):
        if i == 0:
            wf_gs_on_mesh = wf_gs_vmapped(coeff, xmesh)
            normalize_gs = jnp.sum(wf_gs_vmapped(coeff, xmesh) ** 2 * interval)
            wf_gs_on_mesh = wf_gs_on_mesh**2 / normalize_gs
            plt.plot(xmesh, wf_gs_on_mesh, label=f"Rotated-n={i}", lw=2)
        if i == 1:
            wf_1st_on_mesh = wf_first_vmapped(coeff, xmesh)
            normalize_1st = jnp.sum(wf_first_vmapped(coeff, xmesh) ** 2 * interval)
            wf_1st_on_mesh = wf_1st_on_mesh**2 / normalize_1st
            plt.plot(xmesh, wf_1st_on_mesh, label=f"Rotated-n={i}", lw=2)

        exact_wf_on_mesh = exact_eigenvectors[:, i]
        normalize_factor = (exact_wf_on_mesh**2).sum() * interval
        exact_prob_density = exact_wf_on_mesh**2 / normalize_factor
        plt.plot(xmesh, exact_prob_density, "-.", label=f"Exact-n={i}", lw=1.75)
    plt.xlim([-2.0, 2.0])
    plt.legend()
    plt.ylabel(r"$\rho$")
    plt.title("Probability Density (Trained and Compare)")


if __name__ == "__main__":
    main()

# %%
