As mentioned in the "[Why ML?](why_ml.md)" section, though our data looks like an image, it is not exactly set up in a stacked matrix, but rather with each point having the same four properties (x, y, z, charge); this format is referred to as pointclouds. 

This disables us from using a *convolutional* neural network, as those work best with stacked matrices where different filters (other matrices) are applied for transformation. However, there are specific model architectures, that are neural nets, built for the point cloud format specifically; we have chosen to use one called PointNet. This model was coined by Qi et al. at Stanford University in 2017 ([see here for the publication](https://arxiv.org/pdf/1612.00593.pdf)). 


<figure style="display: flex; justify-content: center;">
    <img src="pointnet.png" alt="Config" style="width:80%;"/>
</figure>
<figcaption style="text-align: center;">Figure 6: PointNet architecture</figcaption>

Fig. 6 shows the different purposes for the PointNet model, the part segmentation and semantic segmentation, though useful, is too advanced for our purposes&mdash;we only require the classification tool. It is important to note that the pointclouds used by Qi et al. are objects, not akin to the tracks that are seen in the AT-TPC data; we can already foresee a possible source of trouble. 

If you are interested in getting started with this model, there is a GitHub repository managed by the authors of model [here](https://github.com/charlesq34/pointnet).

If you would like to learn more about the model architecture (and how PointNet works), below are some resources from YouTube that provide an in depth explanation.  

 - [Talk by Charles R. Qi (author) at Conference on Computer Vision and Pattern Recognition (CVR) 2017](https://www.youtube.com/watch?v=Cge-hot0Oc0)

 - [Paper Explained by Aldi Piroli](https://www.youtube.com/watch?v=_py5pcMfHoc) 

 - [Lecture by Maziar Raissi](https://www.youtube.com/watch?v=hgtvli571_U)


The model version I am using was adapted from a [reference](https://keras.io/examples/vision/pointnet/) by Emilio Villasana, Andrew Rice, Raghu Ramanujan, and Dylan Sparks from the [ALPhA group](https://alpha-davidson.github.io) at Davidson College. 