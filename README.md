<p align="center">
  <img src="https://raw.githubusercontent.com/dfm/tinygp/main/docs/_static/zap.png" width="50"><br>
  <strong>tinygp</strong><br>
  <i>the tiniest of Gaussian Process libraries</i>
  <br>
  <br>
  <a href="https://github.com/dfm/tinygp/actions/workflows/tests.yml">
    <img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/dfm/tinygp/Tests">
  </a>
  <a href="https://tinygp.readthedocs.io">
    <img alt="Read the Docs" src="https://img.shields.io/readthedocs/tinygp">
  </a>
</p>

_tinygp_ is an extremely lightweight library for building Gaussian Process
models in Python, built on top of [_jax_](https://github.com/google/jax). It is
not (yet?) designed to provide all the shiniest algorithms for scalable
computations, but I think it has a nice interface, and it's pretty fast. Thanks
to _jax_, _tinygp_ supports things like GPU acceleration and automatic
differentiation.

Check out the docs for more info:
[tinygp.readthedocs.io](https://tinygp.readthedocs.io)
