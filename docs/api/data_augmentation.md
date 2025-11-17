## Data Augmentation

We have azimuthal symmetry within our data, that is, if we rotate the data about the z (beam) axis, the result conserves the physics. We can then "create" copies of any event by rotating it around beam axis by random angles between ``[0,2π]``.

::: scripts.ml_preprocessing_steps.DataAugumentation
