# IMPORTANT


## Important update (2024-07-16)

Commits:
https://github.com/rswang42/2024July1DQuantum/commit/bbc47c02b1be7803048f1d0049b9df20a5e7579a
https://github.com/rswang42/2024July1DQuantum/commit/a8656b046bf055ffdf9b9c7ee0ca98eb38c5ff08

Implement flow as working in Fock space, that is, to individually transform coordinates of different states with different energy quantum numbers. The jacobians are defined w.r.t within the same quantum number, for example, for state i, the jacobian factor that is multiplied to wave function only takes partial fi/ partial xi where fi is the i-th coordinate that transformed from flow and xi is the original coordinate of i-th state.

Besides, each coordinate should be `seperately` transformed, for example, suppose
we are transforming [x0,x1,x2] to [z0,z1,z2], then if we arbitrarily change x0,
leaving the rest x1 and x2 unchanged, then the outputs of the network should
also only have changes in z1 and z2, but z0 unchanged.


Comparision

Left: previous single flow

Right: current single but separetly transformed flow in fock space.

![compare](./assets/Screenshot%202024-07-16%20223324.png)


### Orthonormality

Suppose for the `n-th` excited state 

$$
\Psi_n(x) = \Phi_n(f_n(x,\theta)) \sqrt{\text{det}\left(\frac{\partial f_n(x,\theta)}{\partial x}\right)}
$$

Where $f_n(x,\theta)$ refers to transforming the coordinate(s) of
the `n-th` excited state individually.

Then the wave function ansatze would actually take the form:

$$
\Psi_n(x_n) = \Phi_n(f_n(x_n,\theta_n)) \sqrt{\text{det}\left(\frac{\partial f_n(x_n,\theta_n)}{\partial x_n}\right)}
$$

Since the $x_i$s are flowed independently, the changes of the coordinates
of `i-th` state would not have influence on `j-th` (j!=i) state's result:

$$
\frac{\partial f_j(x_i,\theta_j)}{\partial x_i} = 0
$$

Then we have the orthonormality condition:

$$
\begin{align*}
    \langle \Psi_i | \Psi_j \rangle &= \int dx_i dx_j \langle
            \Psi_i| x_i \rangle \langle x_i | x_j \rangle
            \langle x_j | \Psi_j \rangle \\
            &= \int dx_i dx_j \delta(x_i-x_j) \Psi_i^*(x_i)\Psi_j(x_j)\\
            &= \int dx_i dx_j \delta(x_i-x_j) 
                \Phi_i(f_i(x_i,\theta_i)) \Phi_j(f_j(x_j,\theta_j))
                \sqrt{\text{det}\left(\frac{\partial f_i(x_i,\theta_i)}{\partial x_i}\right)}
                \sqrt{\text{det}\left(\frac{\partial f_j(x_j,\theta_j)}{\partial x_j}\right)} \\
            &= \int dx_i 
                \Phi_i(f_i(x_i,\theta_i)) \Phi_j(f_j(x_i,\theta_j))
                \sqrt{\text{det}\left(\frac{\partial f_i(x_i,\theta_i)}{\partial x_i}\right)}
                \sqrt{\text{det}\left(\frac{\partial f_j(x_i,\theta_j)}{\partial x_i}\right)} \\
\end{align*}
$$

- If $i=j$, then $\langle \Psi_i | \Psi_i \rangle = 1$
- If $i \neq j$, then

$$
                \sqrt{\text{det}\left(\frac{\partial f_j(x_i,\theta_j)}{\partial x_i}\right)}  = 0
$$

- And hence $\langle \Psi_i | \Psi_j \rangle = 0, i \neq j$
- Then we have
$$
\langle \Psi_i | \Psi_j \rangle = \delta_{ij}
$$

Which is the orthonormality condition.


## Important update (2024-07-15)

> UPDATE: 2024-07-16
>
> After the commits in 2024-07-16, the considerations in
> this update is **OUTDATED!**
>
> See updates in 2024-07-16 for a more accurate
> calculation which handles the ground state and excited states
> simultaneously.

By adding the feature to individually training one excited state (also enabling training ARBITRARY selected states), enable to accurately calculate each excitation state. NOTE: the best practice to train one excited state is to ONLY selecting this state, especially one needs to EXCLUDE ground state since the non-nodal feature of the GS would hinder the optimization of nodal excited states!

Enable calculating ARBITRARY selected state(s)

For example,

```bash
--state_indices 0
```
 
would calculate only Ground State

```bash
--state_indices 1
```

would only calculate the 1st excited state

```bash
--state_indices 1 3 5
```

would calculate the 1st, 3rd and 5th excited states simultaneously.

However, to achieve best accuracy, **it is suggested to seperate NODAL and NON-NODAL
wavefunctions in training**: 

- For example, if GS is non-nodal and all the excited states are nodal, then we
may perform a **single** calculation for ground state only to achieve the
best ground state estimation, then perform a **excited states calculation without
ground state**, for example, `--state_indices 1 2 3 4 5 6 7 8` for these nodal states.


# Notes

## Network

Found `sigmoid` always performs better than `tanh`.