# tinygp

**The tiniest of Gaussian Process libraries.**

`tinygp` is an extremely lightweight library for Gaussian Process (GP)
regression in Python, built on top of [`jax`](https://github.com/google/jax). It
is not (yet?) designed to provide all the shiniest algorithms for scalable
computations (check out [celerite2](https://celerite2.readthedocs.io) or
[GPyTorch](https://gpytorch.ai) if you need something like that), but I think it
has a [nice interface](api) and it's pretty fast. Thanks to `jax`, `tinygp`
supports things like GPU acceleration and automatic differentiation.

```{admonition} How to find your way around?
:class: tip

👉 A good place to get started is with the [installation guide](install) and then
the [tutorials](tutorials).

👉 For all the details, check out the [full API documentation](api), including a
list of all the [built-in kernel functions](api#kernels).

👉 If you find bugs or otherwise have trouble getting `tinygp` to do what you want,
head on over to the [GitHub issues page](https://github.com/dfm/tinygp/issues).
```

## Documentation

```{toctree}
:maxdepth: 2

motivation
install
tutorials
api
contributing
code_of_conduct
```

## Authors & License

Copyright 2021 Dan Foreman-Mackey

Built by [Dan Foreman-Mackey](https://github.com/dfm) and contributors (see [the
contribution graph](https://github.com/dfm/tinygp/graphs/contributors) for the
most up to date list). Licensed under the MIT license (see `LICENSE`).
