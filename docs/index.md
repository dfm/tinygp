# tinygp

**The tiniest of Gaussian Process libraries.**

`tinygp` is an extremely lightweight library for building Gaussian Process (GP)
models in Python, built on top of [`jax`](https://github.com/google/jax). It has
a [nice interface](api-ref), and it's pretty fast (see {ref}`benchmarks`).
Thanks to `jax`, `tinygp` supports things like GPU acceleration and automatic
differentiation.

```{admonition} How to find your way around?
:class: tip

🖥 A good place to get started is with the {ref}`install` and then the
{ref}`tutorials`. You might also be interested in the {ref}`motivation` page.

📖 For all the details, check out the {ref}`guide`, including the [full API
documentation](api-ref).

💡 If you're running into getting `tinygp` to do what you want, first check
out the {ref}`troubleshooting` page, for some general tips and tricks.

🐛 If {ref}`troubleshooting` doesn't solve your problems, or if you find bugs,
check out the {ref}`contributing` and then head on over to the [GitHub issues
page](https://github.com/dfm/tinygp/issues).

👈 Check out the sidebar to find the full table of contents.
```

## Table of contents

```{toctree}
:maxdepth: 2

guide
tutorials
contributing
api/index
GitHub Repository <https://github.com/dfm/tinygp>
```

## Authors & license

Copyright 2021, 2022, 2023 Simons Foundation, Inc.

Built by [Dan Foreman-Mackey](https://github.com/dfm) and contributors (see [the
contribution graph](https://github.com/dfm/tinygp/graphs/contributors) for the
most up-to-date list). Licensed under the MIT license (see `LICENSE`).
