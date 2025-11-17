## AT-TPC Noise Addition
### Only for use with simulated data

This method is used **only** when training with simulated data.

It is very difficult to exactly replicate the AT-TPC noise, but we can closely mimic it by creating noise that is uniformly random in z with ``np.random.randint(0,1000)`` and Gaussian in r with ``r_noise = np.random.normal(0, 50, (data_size,1))``.

Can be used to add AT-TPC like noise by using ``np.random.randint()`` between ``[-250,250]`` for x and y, and ``[0,1000]`` for z. 

::: scripts.ml_preprocessing_steps.AttpcNoiseAddition
