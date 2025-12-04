The machine learning (ML) configuration parameters are divided into two sets, one corresponding to the preprocessing pipeline&mdash;how we prepare our data for training, and the ML model training and evaluation. 

Let's begin our journey of specifying parameters once again, but I promise we're done soon!

# ML Preprocessing Pipeline
## Configuration 

There are few parameters that will familiar to us, but also some new ones that are specific to ML preprocessing.

```json
"target_size": 800,
"isotope": "O16",
"run_min": 54,
"run_max": 169,
"data": "path/to/processed/pointcloud/data/",
"event_lengths": "path/to/processed/event/lengths/",
"directory_training": "path/to/save/ml/training/data/"
```

`run_min` and `run_max` are redefined here to give the user the ability to have more freedom over what exact runs they would like to run the ML preprocessing pipeline on, it is important to note that one can set these to the same values in each configuration step if they choose to. The previous data extraction step writes two `.npy` files for each run (event_lengths and pointcloud data), the `event_lengths` and `data` parameters refer to their directory path respectively; where the `directory_training` is the path to the directory where the data from this pipeline will be saved.

To understand what value to set `target_size` to, it's important to learn of it when placed in the context of the up/downscaling section. In summary, our ML model requires static arrays to train on, however, each event in our data can have a different number of points, therefore, we require a way for all events to same amount of points. To do this, we set a target size that we will reach either by adding points or removing them; after some brief research I have found `target_size: 800` to be a good equilibrium point between not having too little or too many points. Finally, the `isotope` refers to the beam particle, for this experiment it's $^{16}\text{O}$. 

## Additional ML Pipeline Parameters

The file named `ml_preprocessing_pipeline.py` puts all the different classes together the pipeline object allows us to combine certain processes together. However, as a user you have the ability to change which processes to turn "on" or "off". **Note: you cannot change the order of the classes, if you would like to add one of the classes it is important to note take a look at the order they are present in `ml_preprocessing_steps.py` and add them in the same order in the pipelines.**

```python
pipeline_1 = Pipeline([
    # ("noise", AttpcNoiseAddition(ratio_noise=0.1)),
    ("outlier",OutlierDetection()), #getting rid of the outliers
    ("sampler", UpDownScaling(target_size,isotope)),
]) #up/down sampler 

# The `pipeline_2` is a data processing pipeline that consists of the following steps:
pipeline_2 = Pipeline([
    ("reclassify",ReclassifyingLabels()),
    ("limiting", DataLimitation(target_size)),
    #("augument", DataAugumentation(target_size=800)),
    ("scaling", ScalingData()),
]) #reclassifying and scaling (w/ concatonated dataset)
```

There are some steps of the pipeline that are *required*, namely:

- UpDownScaling

- ReclassifyingLabels

- ScalingData

Congratulations, now you will be to run a single command to preprocess the data 

```bash
python ml_preprocessing_pipeline.py
```

# ML Model Training and Evaluation
## Configuration

We are nearing the end! I see the light at the end of the tunnel!

The final set of parameter we have to set are the following 

```json
"batch_options": [128, 256],
"lr_options": [3e-6,5e-6,6e-6,7.5e-6],
"epochs_limit": 200,
"data_dir":"path/to/load/ml/training/data/directory/",
"best_model_path": "path/to/save/best/model/best_model.keras",
"learning_curve_path": "path/to/save/learning/curve/learning_curve.png",
"confusion_matrix_path": "path/to/save/confusion/matrix/confusion_matrix.png"
```


`data_dir` is the same parameter as `directory_training` in the previous step, this is where the train, val, and test features and labels reside. `best_model_path` is where you would want your model to be saved in the form of `.keras` format, while `learning_curve_path` and `confusion_matrix_path` are the path to the learning (loss) curve and confusion matrix from model evaluation, respectively. 

We use something called early stopping for training our models, which simply means that we stop training when we don't see improvement, and because we don't have to train as long, it is easier to train multiple models with different hyperparameters to choose the best one. The `batch_options` and `lr_options` refer to batch size and learning rate options, if you would like to learn more about why these are important, [check out this article](https://rumn.medium.com/learning-rate-and-batch-size-is-super-important-than-you-think-68f2e817821e). A model will be trained each combination of batch size and learning rate, choose the number of entries wisely as more entries = more combinations. 

Finally, (really finally) the `epoch_limit` parameter which serves as a default limit for how long the model will train. As mentioned above, we stop training when we don't see improvement, but what happens in the case the model keeps improving? We have to some time, and the epoch is a parameter that controls how many times the training cycle happens.

We have reached the end, everything before actually training the model step could be done locally on a personal machine (though not advised) but training the model requires a high performance cluster, as we need GPUs, and for a considerable time. The slurm script to submit a job and request resources are different for each cluster, but this command can be used at the end to run the training python script. 

```bash
python ml_training.py
```