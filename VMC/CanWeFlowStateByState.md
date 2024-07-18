# Can we flow state by state?

## Orthonormality

Suppose we are dealing with totally `n+1` states (ground state
and n excited states), first we
copy the Hamiltonian `n+1` times and solve it in a
direct product space of `n+1` coordinates: (for simplicity, each $x_i$ would be a one-dimensional coordinate.)

$$
|x\rangle = |x_0\rangle \otimes |x_1\rangle \otimes \ldots \otimes |x_n\rangle
$$

for coordinate basis in different subspaces, we have

$$
\langle x_i|x_j \rangle = \delta_{ij}
$$

Then for the `i-th` excited state (i=0 for ground state), we 
solve it in the corresponding subspace of `i-th` copied Hamiltonian:

$$
\begin{align*}
    \hat H |\Psi_i \rangle &= E_i |\Psi_i \rangle \\
    \langle x_i| \hat H | \Psi_i \rangle &= E_i \langle x_i| \Psi_i \rangle
\end{align*}
$$

And for the `i-th` Hamiltonian $\hat H$ (Hamiltonians in each direct
product subspace are exactly the same) in the `i-th` subspace $\left\{|x_i\rangle\right\}$, we apply an **individual flow for all the
eigenstates within this subspace**:

$$
\hat{\mathbf{f}_{i}} |x_i\rangle = |z_i\rangle
$$

or, applying it onto coordinates:

$$
f_i(x_i,\theta_i) = z_i
$$

This flow, though only affects the coordinates within the
`i-th` subspace $x_i$, **changes all the eigenstates
of** $\hat H$ **simultaneously within the `i-th` subspace**:

$$
\left\{|\Phi_0^i\rangle, |\Phi_1^i\rangle, \ldots, |\Phi_i^i\rangle, \ldots,|\Phi_n^i\rangle \right\} \overset{f_i}{\rightarrow}
\left\{|\Psi_0^i\rangle, |\Psi_1^i\rangle, \ldots, |\Psi_i^i\rangle, \ldots,|\Psi_n^i\rangle \right\}
$$

Here $|\Phi_1^i\rangle$ refers to the 1st excited state in the
`i-th` subspace.

Since **all the states** within `i-th` subspace is transformed
by **the same flow** $f_i$, the orthonormality of wavefunctions
within this subspace is maintained. This yields that **even if
we are individually variating i-th excited state, all the other
states in this orthonormal basis transform simultaneously**.
- Then the **orthonoramlity of the basis are maintained**, throughout
the individually optimization of i-th state.
  - And we would finally get a best variant estimation of i-th
state, and some not-that-well estimation of all the other states.
- Take the ground state variance calculation for example:
  - Say at first we only optimize a single ground state, and get the best ground state estimation by optimizing one single flow. Suppose now we have the flow $f_0$ ready for ground state wavefunction:
  - $\Psi_0(x) = \Phi_0(f_0(x)) \sqrt{\text{det}\left(\frac{\partial f_0(x)}{\partial x}\right)}$
  - Note here for simplicity, the subscript denoting which subspace we are dealing with is ignored.
  - Then actually we can construct arbitrary excited state $\Psi_j(x)$ with the same flow and keep the orthonormality of any pair of the states, as we've already known:
  - $\Psi_j(x) = \Phi_j(f_j(x)) \sqrt{\text{det}\left(\frac{\partial f_j(x)}{\partial x}\right)}$
  - The only defect of this j-th excited state we've paid no effort optimizing on would be, though, it is a rather worse estimation of the j-th excited state.
  - But the picture is, even if we are **only optimizing one individual state, we are actually transforming the whole basis**, and then we are **persisting the orthonormality of all the states in the whole basis simultaneously**.
- If we would simply switch the subscript 0 and j of the statement above, we would achieve our conclusion, "even if we are individually variating i-th excited state, all the other states in this orthonormal basis transform simultaneously, and the orthonormality of all the states are maintained."

**Then the orthonormality between subspaces are simple**:

Since we only care about the i-th excited state in the `i-th`
subspace, $|\Psi_i^i\rangle$, we omit the superscript $i$ since
the states we would calculate would always have the same subscript
and superscript.



And the wave function ansatze would actually take the form:

$$
\Psi_n(x_n) = \Phi_n(f_n(x_n,\theta_n)) \sqrt{\text{det}\left(\frac{\partial f_n(x_n,\theta_n)}{\partial x_n}\right)}
$$

Then we have the orthonormality condition:

$$
\begin{align*}
    \langle \Psi_i | \Psi_j \rangle &= \int dx_i dx_j \langle
            \Psi_i| x_i \rangle \langle x_i | x_j \rangle
            \langle x_j | \Psi_j \rangle \\
            &= \int dx_i dx_j \delta_{ij} \Psi_i^*(x_i)\Psi_j(x_j)\\

\end{align*}
$$

- If $i=j$, then $\langle \Psi_i | \Psi_i \rangle = 1$
- If $i \neq j$, then $\delta_{ij}=0$
  - And hence $\langle \Psi_i | \Psi_j \rangle = 0, i \neq j$
- Then we have
$$
\langle \Psi_i | \Psi_j \rangle = \delta_{ij} \quad (1)
$$

Which is the orthonormality condition.

**How to check the flow has the separately transformation feature:**

Since the $x_i$s are flowed independently, the changes of the coordinates
of `i-th` state would not have influence on `j-th` (j!=i) state's result:

$$
\frac{\partial f_j(x_i,\theta_j)}{\partial x_i} = 0
$$

This could be implemented as the testcase of the whole flow:

$$
f = (f_0,f_1,\ldots,f_n)
$$

w.r.t.

$$
x = (x_0,x_1,\ldots,x_n)
$$

**To wrap up**

The idea is simply that, **find the best variant estimation of the i-th excited state within the `i-th` direct product subspace of the i-th copied (but the same) Hamiltonian**.
