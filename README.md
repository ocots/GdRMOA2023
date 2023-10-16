# Numerical methods for classical and hybrid optimal control problems

<img width="800px" alt="Capture d’écran 2023-10-09 à 18 27 40" src="https://github.com/control-toolbox/GdRMOA2023/assets/66357348/25e77d9b-bcdf-4ddd-84fc-b916a3031a54">

**Journées annuelles 2023 du GdR MOA, 18 octobre, Université Perpignan**

[Olivier Cots](https://ocots.github.io/)[^1]

[^1]: part of Térence Bayen lectures at [Journées annuelles 2023 du GdR MOA](https://gdrmoa.math.cnrs.fr/journees-annuelles-2023-du-gdr-moa).

We are interested in nonlinear optimal control of ODEs:

```math
g(x(t_0),x(t_f)) + \int_{t_0}^{t_f} f^0(x(t), u(t))\, \mathrm{d}t \to \min
```

subject to

```math
\dot{x}(t) = f(x(t), u(t)),\quad t \in [t_0, t_f]
```

plus boundary, control and state constraints. We present:

- [an introduction to direct and indirect methods](https://control-toolbox.org/GdRMOA2023/basic.html)
- the `OptimalControl.jl` package on a [basic example](https://control-toolbox.org/docs/optimalcontrol/stable/tutorial-basic-example.html)
- a more [advanced example: the Goddard problem](https://control-toolbox.org/docs/optimalcontrol/stable/tutorial-goddard.html), which combines direct and indirect methods.

For fun!

[ct playground](https://control-toolbox.org/GdRMOA2023/ct-playground.html)
