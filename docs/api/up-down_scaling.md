## Resampling

Each event has a different amount of point cloud lengths, making the data tensor filled with empty zeros where the point clouds are shorter. We require a static tensor with no zeros, the best way to do this is by choosing a target value&mdash;upscaling any events that are lower and downscaling any that are higher.

::: scripts.ml_preprocessing_steps.UpDownScaling
