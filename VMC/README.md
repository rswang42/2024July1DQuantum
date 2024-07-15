# IMPORTANT

## Important update (2024-07-15)


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