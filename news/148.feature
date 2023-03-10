Removed `__post_init__` checks after kernel construction to avoid extraneous errors when returning kernels out of `jax.vmap`'d functions.
