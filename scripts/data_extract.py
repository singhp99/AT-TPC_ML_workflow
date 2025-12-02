mport h5py
from tqdm import tqdm
import numpy as np
import pandas as pd


def prep_4_ml(self, group):
    """
    Function to strip data of event names, attach labels to data and convert to .npy files

    Args:
            group (h5py.Group): HDF5 group object.
            est_path (str): Path to the parquet file with estimate classification.

        Returns:
            h5py.Group: Modified HDF5 group with new 'tracks' attribute.
    """
    attributes = dict(group.attrs)
    min_event = attributes["min_event"]
    max_event = attributes["max_event"]
    
        