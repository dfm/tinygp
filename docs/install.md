(install)=

# Installation Guide

`tinygp` is built on top of [`jax`](https://github.com/google/jax) so that's the
primary dependency that you'll need. All of the methods below will install any
required dependencies, but if you want to take advantage of your GPU, that might
take a little more setup. `tinygp` doesn't have any GPU-specific code, so it
should be enough to just [follow the installation instructions for CUDA support
in the `jax` README](https://github.com/google/jax/#installation).

## Using pip

The easiest way to install the most recent stable version of `tinygp` is
with [pip](https://pip.pypa.io):

```bash
python -m pip install tinygp
```

## From source

Alternatively, you can get the source:

```bash
git clone https://github.com/dfm/tinygp.git
cd tinygp
python -m pip install -e .
```

## Tests

If you installed from source, you can run the unit tests. From the root of the
source directory, run:

```bash
python -m pip install nox
python -m nox -s test -p 3.10
```
