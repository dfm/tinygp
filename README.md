<p align="center">
  <img src="https://raw.githubusercontent.com/dfm/tinygp/main/docs/_static/zap.png" width="50"><br>
  <strong>tinygp</strong><br>
  <i>the tiniest of Gaussian Process libraries</i>
  <br>
  <br>
  <a href="https://github.com/dfm/tinygp/actions/workflows/tests.yml">
    <img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/dfm/tinygp/tests.yml?branch=main">
  </a>
  <a href="https://tinygp.readthedocs.io">
    <img alt="Read the Docs" src="https://img.shields.io/readthedocs/tinygp">
  </a>
  <a href="https://doi.org/10.5281/zenodo.6389737">
    <img alt="Zenodo DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.6389737.svg">
  </a>
</p>

`tinygp` is an extremely lightweight library for building Gaussian Process (GP)
models in Python, built on top of [`jax`](https://github.com/google/jax). It has
a [nice interface][api-ref], and it's [pretty fast][benchmarks]. Thanks to
`jax`, `tinygp` supports things like GPU acceleration and automatic
differentiation.

Check out the docs for more info: [tinygp.readthedocs.io][docs]

[api-ref]: https://tinygp.readthedocs.io/en/latest/api/index.html
[benchmarks]: https://tinygp.readthedocs.io/en/latest/benchmarks.html
[docs]: https://tinygp.readthedocs.io
