(troubleshooting)=

# Troubleshooting

This page includes some tips for troubleshooting issues that you might run into
when using `tinygp`, and Gaussian process models more generally. This is a
work-in-progress, so if you don't see your issue listed here, feel free to open
an issue on the [GitHub repository issue
tracker](https://github.com/dfm/tinygp/issues).

## NaNs and infinities

It's not uncommon to find that the marginalized likelihood of your `tinygp`
model evaluates to `jnp.nan` or `-jnp.inf`. This is often caused by numerical
precision issues in the linear algebra calculations. This can be exacerbated by
the fact that, by default, `jax` disables double precision calculations. You can
enable double precision [a few different ways as described in the `jax`
docs](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision),
and the way we do it in these docs is to add the following, when necessary:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

If enabling double precision doesn't do the trick, this often means that there's
an issue with the parameter or modeling choices that you're making. For example,
some kernel parameters must be constrained to be positive, although that is not
strictly enforced by the API. Double check that you're not allowing these
parameters to go negative or to zero.

Similarly, you can end up with numerical issues if you set the length scales of
your problem much larger than the dynamic range of your data. When you have
persistent issues with NaNs and infinities, it is worth double-checking the
allowed ranges on parameters, and making sure that they are sensibly bounded.

Especially when using Markov chain Monte Carlo for Gaussian process inference,
naive choices of prior bounds on kernel parameters can lead to pretty heinous
results (see [this case study from Michael
Betancourt](https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#32_Exploring_the_Marginal_Likelihood_Function)
for some striking examples), so care should be taken when setting bounds on
these parameters.
