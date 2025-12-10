This page discusses the classes that make up the machine learning pipeline, more specifically, the custom transformers that are used within the scikit-pipeline. This is really useful because if you wanted to create another step in the pipeline, all you'd have to do is follow the same format as the ones already there. 

## Noise Addition 
These two custom transformers are intended to add noise when training with simulated data, and can be turned "off" when training with experimental data. 

### Uniform Noise Addition  
This adds random uniform noise in Cartesian coordinates by using `np.random.randint()` ([more info can be found here](https://numpy.org/devdocs/reference/random/generated/numpy.random.randint.html)), with the physical dimensions of the detector in mm.

- x: [-250,250]
- y: [-250,250]
- z: [0,1000] (this is the beam axis; the detector is 1 m long)

There one input parameter for this transformer `ratio_noise`, which stands for the fraction of noise added compared to the point cloud size. 

<figure style="display: flex; justify-content: center;">
    <img src="../uniform noise.png" alt="Config" style="width:80%;"/>
</figure>
<figcaption style="text-align: center;">Figure 7: Uniform noise with two different ratio number</figcaption>

### ATTPC Noise Addition 
This adds random uniform noise in z by using `np.random.randint()`, and Gaussian noise in r by `np.random.normal()` ([more info can be found here](https://numpy.org/doc/2.1/reference/random/generated/numpy.random.normal.html)).

- r: [-50,50]
- z: [0,1000] (this is the beam axis; the detector is 1 m long)

There one input parameter for this transformer `ratio_noise`, which stands for the fraction of noise added compared to the point cloud size. 

<figure style="display: flex; justify-content: center;">
    <img src="../at-tpc noise.png" alt="Config" style="width:100%;"/>
</figure>
<figcaption style="text-align: center;">Figure 8: AT-TPC like noise with different ratio numbers</figcaption>

## Outlier Detection 
This transformer doesn't require any parameters but rather removes any points in the data that fall outside the physical dimensions of the detector. It is recommended to have this part of the pipeline at all times. 

## Up Down Scaling 
This transformer has four different parameters that need to be defined by the user: `target_size`, `dimension`, and `isotope`. The isotope is the beam for the specific experiment (in this case it's `"O16"`); the dimension corresponds to the number of features that we will be training on, i.e. for just x,y,z (dimension = 3) but if charge is also included then it's 4. 

Finally, the `target_size` refers to the point cloud size we would be resampling to. I have chosen this to be `target_size = 800` by looking that the average size of pointcloud size for each class. 

<figure style="display: flex; justify-content: center;">
    <img src="../points_distribution.png" alt="Config" style="width:80%;"/>
</figure>
<figcaption style="text-align: center;">Figure 9: Pointcloud size for each class</figcaption>

This is another transformer that is recommended to keep in at all time.

## Reclassifying Labels
Since the ML model requires classes to begin with the label 0, we must subtract one from each label; now one track correspond to class 0, all the way to five track events corresponding to class 4. 

This transformer doesn't require a user-defined parameter. 

## Data Limitation 
This transformer limits examples from all classes to the one for the lowest class, in this way the model is not biased to one class and the model evaluation metrics are a true reflection of how the model performs on *all* classes. 

No user-defined parameter required. 

## Data Augmentation 
In our experimental data, the one and two tracks make up majority of the data, while 3,4, and 5 track events are rare multi-track events. To "make up" for these numbers, we can perform something called data augmentation&mdash;making multiple copies of a single event. The reason this is possible is because of azimuthal symmetry in the detector around the beam (z) axis. The parameter `multiplier` controls how many copies to create for each event.

This transformer requires `target_size` as a parameter.

<figure style="display: flex; justify-content: center;">
    <img src="../class3_augmentation.png" alt="Config" style="width:100%;"/>
</figure>
<figcaption style="text-align: center;">Figure 10: Augmentation for a class 3 event</figcaption>

## Scaling Data 
Since each event has points that consists of points scattered throughout the detector, it means that every pointcloud does not span the same physical space within the detector, but we need it to. Hence, we use this transformer to scale our data with scikit `MinMaxScalar()` ([more information can be found here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)).


