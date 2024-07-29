# Checking The Ratio

Notations:

- $\Psi_i(x_j)$: the transformed wavefunction at real space coordinate $x_j$
- $\Phi_i(z_j) = \Phi_i(f(x_j))$: the base wavefunction at latent space $z_j = f(x_j)$

For any state, i, j, we have the probability ratio at $x_i$:

$$
\begin{align*}
    \frac{|\Psi_i(x_i)|^2}{|\Psi_j(x_i)|^2} &= \frac{|\Phi_i(f(x_i))|^2\text{det}\left(\frac{\partial f(x)}{\partial x}\right)\bigg|_{x=x_i}}{|\Phi_j(f(x_i))|^2\text{det}\left(\frac{\partial f(x)}{\partial x}\right)\bigg|_{x=x_i}} \\
        &= \frac{|\Phi_i(f(x_i))|^2}{|\Phi_j(f(x_i))|^2} \\
        &= \frac{|\Phi_i(z_i)|^2}{|\Phi_j(z_i)|^2} \\
\end{align*}
$$

Now let's take the ground state (n=0) and the 1-st and 2-nd excited states (n=1,2), we can establish two ratio:

1-st w.r.t. gs:

$$
    \frac{|\Psi_1(x_i)|^2}{|\Psi_0(x_i)|^2} = \frac{|\Phi_1(z_i)|^2}{|\Phi_0(z_i)|^2} 
$$

2-st w.r.t. gs:

$$
    \frac{|\Psi_2(x_i)|^2}{|\Psi_0(x_i)|^2} = \frac{|\Phi_2(z_i)|^2}{|\Phi_0(z_i)|^2} 
$$

Then we inspect these ratios at the **node point of the tranformed 1-st excited state**:

- Suppose 

$$
\Psi_1(x_{k}) = 0
$$

- Then we have

$$
    \frac{|\Psi_1(x_{k})|^2}{|\Psi_0(x_{k})|^2} = \frac{|\Phi_1(z_k)|^2}{|\Phi_0(z_k)|^2} = 0\quad (1)
$$

$$
    \frac{|\Psi_2(x_{k})|^2}{|\Psi_0(x_{k})|^2} = \frac{|\Phi_2(z_k)|^2}{|\Phi_0(z_k)|^2} \quad (2)
$$

- Where $f(x_k)=z_k$
- Solving equation (1), we get (suppose ground state is nodeless)

$$
|\Phi_1(z_k)|^2 = 0
$$

- And we could solve $z_k$:

$$
z_k = z_\text{1st-base-node}
$$

- Which means that if we apply the flow onto the node point of the 1-st excited transformed wavefunction $x_k$, we will inevitably get the node point of the 1-st excited base wavefunction $z_k = z_\text{1st-base-node}$.
  - This would cause no issue if we are only dealing with the ground state and the 1-st excited state, since the equation (1) would always hold.
  - However, if we inspect the ratio in equation (2), we found:

$$
    \frac{|\Psi_2(x_{k})|^2}{|\Psi_0(x_{k})|^2} = \frac{|\Phi_2(z_\text{1st-base-node})|^2}{|\Phi_0(z_\text{1st-base-node})|^2} = \text{const}\quad (3)
$$

- That is, the ratio between the **second** excited state and the **ground state** would be fixed to a constant that is dominated by our basis wavefunctions:
  - For example, the basis of harmonic oscillator:
  - $\Phi_i(z) = e^{-z^2/2}H_n(z)$
  - And $H_n$ the Hermite polynominals
  - We have ([Hermite Polynominals Reference](https://mathworld.wolfram.com/HermitePolynomial.html))

$$
\frac{|\Phi_2(z)|^2}{|\Phi_0(z)|^2} = \left|\frac{H_2(z)}{H_0(z)}\right|^2 = \left|\frac{4z^2-2}{1}\right|^2 = (4z^2-2)^2
$$

- This constriants the ratio of our transformed wavefunctions:

$$
    \frac{|\Psi_2(x_{k})|^2}{|\Psi_0(x_{k})|^2} = (4z_\text{1st-base-node}^2-2)^2
$$

- Which is a constant that is dominated by the basis we choose.


![Ratio, base and trained](./CheckingTheRatio.assets/Screenshot%202024-07-20%20171729.png)

![Ratio, exact](./CheckingTheRatio.assets/Screenshot%202024-07-20%20171822.png)