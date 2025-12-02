from data_extract import prep_4_ml
from data_inspect import Inspect
from tqdm import tqdm
import h5py
import pandas as pd
from pathlib import Path


def all_runs_prep_4_ml(file_path: str, est_path: str, run_number: int):
    """
    Function to prepare all runs data for machine learning by stripping event names,
    attaching labels, and converting to .npy files.

    Args:
        file_path (str): Path to the HDF5 file.
        est_path (str): Path to the parquet files for label extraction.
        run_number (int): Run number.
    Returns:
        np.ndarray: Prepared data array for machine learning.
    """
    
    inspect = Inspect(number_to_viz=None, num_tracks=None) # Initialize Inspect object
    
    file_h5 = h5py.File(file_path, 'w')
    file_est = pd.read_parquet(est_path, engine="pyarrow")
    
    group = inspect.h5_keys_extract(file_h5)
    group_with_tracks = inspect.add_attr_tracks(group, file_est)
    
    event_lengths, event_data = prep_4_ml(inspect, group_with_tracks)
    
    #need to save these files temporarily 


if __name__ == "__main__":
    
    #can read this from a config file later
    min_run = 54
    max_run = 169
    
    for run_number in tqdm(range(min_run, max_run + 1), desc="Processing Runs"):
        file_path = "data/run_001.h5"
        est_path = "data/run_001_estimates.parquet"
        
        if not file_path.exists():
            continue
        
        all_runs_prep_4_ml(file_path, est_path, run_number)