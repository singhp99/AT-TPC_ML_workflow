## Spyral: run_spyral.py

Since the Spyral package is a tool created to work with data from any of the AT-TPC experiments, a user has the ability to refine the experimental parameters to those of their experiment. The file we use to define these parameters is `run_spyral.py`, and these parameters have already been set for the $^{16}\text{O}$, but we do have to change some things. 

One of the first things we will have to change will be some paths defined in

```python
workspace_path = Path("/Volumes/researchEXT/O16/no_efield/no_field_fitracks_v1.0/")
trace_path = Path("/Volumes/researchEXT/O16/no_efield/some_traces/")
```

Where you will be changing the `workspace_path` to a directory where all the analyzed file will go, and a `trace_path` to the directory where all the raw traces are. 

Next, we can change the runs we want to analyze and hoe many processes will be running in parallel

```python
run_min = 54
run_max = 169
n_processes = 17
```
These are what the parameters need to be changed to, to analyze all the runs with 17 processes in parallel.

Now, we will need to change the paths to all the specific parameters files 

Note that the argument `do_garfield_correction=False` in `det_params = DetectorParameters()` is set to `False`, this is the electric field correction and I would suggest leaving it off for this experiment. 

There are two types of clustering algorithms that can be used here, currently I suggest using the one it is set to, HDBSCAN, but another clustering method called Triple Cluster can be use, though it is currently not stable. 

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

You would need to uncomment Tripclust related parameters and comment HDBSCAN ones if you would like to make that switch. 

Finally, the last thing that could be changed by the user is the phases that are run. For the extraction of training labels and features, the list in the `Pipeline()` must be changed to 

```python
[True, True, True, False],
```

This will run the first three phases, and our training features come from Phase 1, while the labels from Phase 3. 

When you are ready to begin analyzing the data, simply run 

```bash
python run_spyral
```

