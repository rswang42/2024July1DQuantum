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