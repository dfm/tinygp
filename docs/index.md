# tinygp

**The tiniest of Gaussian Process libraries.**

`tinygp` is an extremely lightweight library for building Gaussian Process (GP)
models in Python, built on top of [`jax`](https://github.com/google/jax). It is
not (yet?) designed to provide all the shiniest algorithms for scalable
computations (check out [celerite2](https://celerite2.readthedocs.io) or
[GPyTorch](https://gpytorch.ai) if you need something like that), but I think it
has a [nice interface](api-ref), and it's pretty fast. Thanks to `jax`, `tinygp`
supports things like GPU acceleration and automatic differentiation.

```{admonition} How to find your way around?
:class: tip

🖥 A good place to get started is with the {ref}`install` and then the
{ref}`tutorials`. You might also be interested in the {ref}`motivation` page.

📖 For all the details, check out the {ref}`guide`, including the [full API
documentation](api-ref).

🐛 If you find bugs or otherwise have trouble getting `tinygp` to do what you
want, check out the {ref}`contributing` or head on over to the [GitHub issues
page](https://github.com/dfm/tinygp/issues).

👈 Check out the sidebar to find the full table of contents.
```

````{admonition} A note about numerical precision
:class: warning

The linear algebra used to evaluate Gaussian process models can be
numerically unstable, especially with single point precision. Throughout this
documentation, we will enable 64-bit precision manually, and you might want to
do the same, especially if you're hitting NaNs or infinities. The `jax` docs
[describe a few ways to enable x64](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision),
but the method that we'll use here is to include the following snippet in each
tutorial:

```python
from jax.config import config
config.update("jax_enable_x64", True)
```

````

```{toctree}
:hidden:

guide
tutorials
contributing
GitHub Repository <https://github.com/dfm/tinygp>
```

## Authors & license

Copyright 2021, 2022 Simons Foundation, Inc.

Built by [Dan Foreman-Mackey](https://github.com/dfm) and contributors (see [the
contribution graph](https://github.com/dfm/tinygp/graphs/contributors) for the
most up-to-date list). Licensed under the MIT license (see `LICENSE`).
