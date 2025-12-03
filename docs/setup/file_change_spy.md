## Configuration

Since the Spyral package is a tool created to work with data from any of the AT-TPC experiments, a user has the ability to refine the experimental parameters to those of their experiment. The file we use to define these parameters is `run_spyral.py`, and these parameters have already been set for the $^{16}\text{O}$, but some user specific parameters are defined in `config.json`. 

One of the first things we will have to change will be some paths defined in

```json
"workspace_path": "/path/to/your/spyral/workspace/",
"trace_path": "/path/to/your/spyral/traces/",
```

Where you will be changing the `workspace_path` to a directory where all the analyzed file will go, and a `trace_path` to the directory where all the raw traces are. 

Next, we can change the runs we want to analyze and hoe many processes will be running in parallel

```json
run_min = 54
run_max = 169
n_processes = 17
```
These are what the parameters need to be changed to, to analyze all the runs with 17 processes in parallel, but the suggested amount is `n_processes = 5` if being used on a personal machine.

Finally, the last thing that could be changed by the user is the phases that are run. For the extraction of training labels and features, the list in the `Pipeline()` must be changed to 

```json
[true, true, true, false]
```
This corresponds to the `PointcloudPhase()`, `ClusterPhase()`, `EstimationPhase()`, and `InterpSolverPhase()` phases in a consecutive manner. 

If you have made the changes to suit your device, you are ready to begin analyzing the data, simply run 

```bash
python run_spyral
```

If you would like, you can stop here and the Spyral analysis run with the intended parameters for this experiment. However, if you would like more control over the specific methods through which each phase is run, some other parameters can be changed in `run_spyral.py` file. 

## Additional Spyral Parameters

Note that the argument `do_garfield_correction=False` in `det_params = DetectorParameters()` is set to `False`, this is the electric field correction and I would suggest leaving it off for this experiment but you have the ability to turn it on. 

There are two types of clustering algorithms that can be used here, currently I suggest using the one it is set to, HDBSCAN, but another clustering method called Triple Cluster can be use, though it is currently not stable. 

Even within the HDBSCAN clustering algorithm you do have the ability to change, if you would like more information about each of the arguments here, you can find ([more information here](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)).

```python
cluster_params = ClusterParameters(
    min_cloud_size=50,
    hdbscan_parameters= HdbscanParameters(
    min_points=3,min_size_scale_factor=0.05,
    min_size_lower_cutoff=10, 
    cluster_selection_epsilon=10.0,
    ),
    # hdbscan_parameters= None,
    tripclust_parameters=None,
    # tripclust_parameters=TripclustParameters(
    #     r=6,
    #     rdnn=True,
    #     k=12,
    #     n=3,
    #     a=0.03,
    #     s=0.3,
    #     sdnn=True,
    #     t=0.0,
    #     tauto=True,
    #     dmax=0.0,
    #     dmax_dnn=False,
    #     ordered=True,
    #     link=0,
    #     m=50,
    #     postprocess=False,
    #     min_depth=25,
    # ),
    overlap_join=OverlapJoinParameters(
        min_cluster_size_join=15,
        circle_overlap_ratio=0.25,
    ),
    # overlap_join=None,
    continuity_join = ContinuityJoinParameters(
    join_radius_fraction=0.4,
    join_z_fraction=0.2),
    direction_threshold= 0.5,
    outlier_scale_factor=0.1,
    
)
```

Please note you would need to uncomment Tripclust related parameters and comment HDBSCAN ones if you would like to make that switch, and pass `None` for the one you choose not to use. 
