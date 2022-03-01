(motivation)=

# Why tinygp?

There are many Python libraries that exist for Gaussian Process (GP) modeling,
so one might ask: _(a)_ why does `tinygp` exist and _(b)_ why might someone want
to use it?

## Why does `tinygp` exist?

Its development started as an experiment because I wanted to figure out how to
best leverage `jax` to build a minimal, but flexible and high performance GP
library. I also wanted to learn some subtleties of designing a `jax`-based
library.

A fundamental design decision is that `tinygp` does not offer implementations of
_any_ inference routines. Instead, it only provides an expressive interface for
designing GP kernels and defining the relevant `jax` operations. Because of the
composable nature of `jax` code, this high-level interface is compatible with
other `jax`-based modeling frameworks such as `numpyro` and `flax`. This design
has some benefits (it can take advantage of these excellent existing libraries
and any fast linear algebra available in `jax`) and some shortcomings (it won't
necessarily support all the state-of-the-art GP inference algorithms out of the
box).

## Other Gaussian process libraries

There are a lot of other libraries for using GPs in Python, so I won't list them
all here. Some of the most popular libraries are:

- [`sklearn.gaussian_process`](https://scikit-learn.org/stable/modules/gaussian_process.html),
- [`GPy`](https://sheffieldml.github.io/GPy), and
- [`GPyTorch`](https://gpytorch.ai).

These all aim to "do it all", in the sense that they provide a modeling
framework, inference algorithms, and a lot of other nice features. For problems
with extremely large datasets, `GPyTorch` is particularly interesting, since it
includes novel scalable linear algebra and approximate inference techniques that
allow black-box models to operate at scale. These could all be good choices if
you're looking for a general purpose GP library.

I'm the lead developer of two other GP libraries for Python:

- [`george`](https://george.readthedocs.io), and
- [`celerite`](https://celerite2.readthedocs.io)

that aim to sit at a lower level from the popular libraries above. These
packages primarily provide methods for evaluating GP likelihoods that can be
integrated into other data analysis pipelines. `tinygp` is meant as a `george`
replacement, and I think it's unlikely that you would ever want to use `george`
instead (more on that below). `celerite` is (currently) restricted to
1-dimensional datasets with a specific type of kernel function, but if your
problem fits into that framework, I think it would be hard to beat. I'm hoping
to implement an interface to `celerite` as part of `tinygp` at some point.

There are some other new GP libraries built on top of `jax`, including:

- [`GPJax`](https://gpjax.readthedocs.io), and
- [`jax-kern`](https://jaxkern.readthedocs.io).

At the time of writing these libraries don't seem to be ready for public
consumption either, but they are worth keeping an eye on!

## What about `george`?

As mentioned above, I am also the lead developer of the
[`george`](https://george.readthedocs.io) library, which fills much the same
niche as `tinygp`, so I thought it would be worth saying a few words about that.
In fact, I started developing `tinygp` in large part because I wanted to stop
maintaining `george`, and I wanted to have something else to point users to. The
main reason I want to retire `george` is that I made some fundamental design
decisions that made sense at that time, but have since been obviated by
libraries like `jax`. In particular, `george` requires kernel functions to be
implemented in C++, via an [awkward YAML
specification](https://george.readthedocs.io/en/latest/tutorials/new-kernel/).
This allowed high performance kernel function evaluation, but also meant that
adding a custom kernel required re-compiling the library. The JIT-compilation
provided by `jax` provides these features with a much more ergonomic API.
`george` also includes a [homebrewed modeling
framework](https://george.readthedocs.io/en/latest/user/modeling/) with limited
and awkward support for named parameters, differentiation, and some other
domain-specific features. The automatic differentiation provided by `jax`, and
the rich ecosystem of modeling and inference frameworks built on top of it
(including `numpyro`, `flax`, `blackjax`, etc.) offer all of these same features
and much more. Either way, it's probably time to move on from `george`!

The only feature that `george` has that is not yet implemented in `tinygp` is
the "HODLR" approximate linear algebra technique. This has somewhat limited
applicability (in particular it is really only useful for 1-dimensional data,
where `celerite` is probably a better choice anyway), and using the
GPU-accelerated version of `tinygp` will often provide even better performance,
see {ref}`benchmarks`.

With these points in mind, much of this documentation will discuss `tinygp` in
the context of `george` and give advice on porting models.
