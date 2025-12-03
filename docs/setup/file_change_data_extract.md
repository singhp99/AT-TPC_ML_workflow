## Configuration

Okay, you have set up the configuration for the Spyral files, but what's next? If you inspect `config.json`, you can see that there are other configurations we will have to set before we can run this in a workflow, or individually extract the data and labels. 

The parameters that correspond to extracting the point clouds with their labels is `data_extract_parameters`, and it consists of:

```json
"run_min": 54,
"run_max": 169,
"file_path": "path/to/pointcloud/h5/file",
"est_path": "path/to/est/file"   
```

Where `run_min` and `run_max` exactly how many runs for which to get the data; though they are no different that the same parameters in `spyral_parameters`, however, one may choose not to process all the data together. This allows for freedom in how much data we want to train on (most likely not going to be all of it). 

The `file_path` is going to be for the Pointcloud directory, the point cloud files from which we will extract the features; and `est_path` for the Estimation directory, from which we extract the labels. 

Congratulations! Now you will be able to get data that has the labels attached to it and simply need to run. 

```bash 
python fnc_all_runs.py
```


