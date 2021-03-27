# Installation

## Using pip

The easiest way to install the most recent stable version of `tinygp` is
with [pip](http://www.pip-installer.org/):

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
python -m pip install ".[test]"
python -m pytest -v tests
```

This might take a few minutes but you shouldn't get any errors if all went
as planned.
