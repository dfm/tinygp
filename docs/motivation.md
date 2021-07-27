# Why tinygp?

There are many Python libraries that exist for Gaussian Process (GP) modeling,
so one might ask: _(a)_ why does `tinygp` exist and _(b)_ why might someone want
to use it?

## Why does `tinygp` exist?

Its development started as an experiment because I wanted to figure out how to
best leverage `jax` to build a minimal, but flexible and high performance GP
library. I also wanted to learn some of the subtleties of designing a
`jax`-based library.

A fundamental design decision is that `tinygp` does not offer implementations of
_any_ inference routines. Instead it only provides an expressive interface for
designing GP kernels and defining the relevant `jax` operations. Because of the
composable nature of `jax` code, this high-level interface is compatible with
other `jax`-based modeling frameworks such as `numpyro` and `flax`. This design
has some benefits (it can take advantage of these excellent existing libraries
and any fast linear algebra available in `jax`) and some shortcomings (it won't
necessarily support all the state-of-the-art GP inference algorithms out of the
box).
