## Uniform Noise Addition 
### Only for use with simulated data

This method is used **only** when training with simulated data.

Can be used to add uniform noise by using ``np.random.randint()`` between ``[-250,250]`` for x and y, and ``[0,1000]`` for z. 

::: scripts.ml_preprocessing_steps.UniformNoiseAddition
