import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd


def prep_4_ml(group):
    """
    Function to strip data of event names, attach labels to data and convert to .npy files

    Args:
        group (h5py.Group): HDF5 group object.
        

    Returns: 
        np.ndarray: Prepared data array for machine learning.        
    """
    attributes = dict(group.attrs)
    min_event = attributes["min_event"]
    max_event = attributes["max_event"]
    
    event_lengths = np.zeros(max_event-min_event, int)
    
    for i, e in enumerate(group):
        event_lengths[i] = len(group[e])
    
    event_data = np.full((len(event_lengths), np.max(event_lengths) + 2, 4), np.nan)
    
    for i, e in tqdm.tqdm(enumerate(group)): 
        for n in range(event_lengths[i]):
            event_data[i, n] = group[e][n,:4]
        label = group[e].attrs["tracks"]
        event_data[i, -2] = [label] * 4 # label for classification
        event_data[i, -1] = [i] * 4 # event index for reference
        
    return event_lengths, event_data