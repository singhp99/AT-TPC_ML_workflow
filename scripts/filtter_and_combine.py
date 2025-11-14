import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Filter:
    
    def __init__(self, number_to_viz: int, num_tracks: int):
        self.number_to_viz = number_to_viz
        self.num_tracks = num_tracks
    
    def h5_keys_extract(self, file_h5):
        """Extracts the data from an HDF5 file.

        Args:
            file_path (HDF5): HDF5 file object
        Returns:
            group(HDF5 group object): HDF5 group object
        """
        group_ls = list(file_h5.keys())[0] #getting the first group
        group = file_h5[group_ls] #accessing the group
    
        return group
        
        
    def check_nans(self, group):
        """Check for any NaN values in the data. If there is any, this indicates an issue with the Spyral pipeline.

        Args:
            group (HDF5 group object): The first group of the HDF5 file.
        Returns:
            bool: True if NaNs are found, False otherwise.
        """
        
        for key in tqdm(group, desc="Checking for NaNs"):
            data_array = group[key][:]
            
            if np.isnan(data_array).any():
                return True # NaNs found  

        return False # No NaNs found
    
    def add_attr_tracks(self, group, est_path: str):
        """Adds a new attribute 'class' to each dataset in the HDF5 group from the parquet files for estimate classification

        Args:
            group (HDF5 group object): The first group of the HDF5 file.
            
        Returns:
            group (HDF5 group object): Modified HDF5 group object with new attribute 'class' added to each dataset.
        """
        
        file_est = pd.read_parquet(est_path, engine="pyarrow")
        grouped = file_est.groupby("event")
        
        group_sizes = grouped.size()
        
        for event, size in tqdm(group_sizes.items(), total = len(group_sizes), desc="Adding class attribute"):
            key = "cloud_" + str(event)
            if key in group:
                group[key].attrs["tracks"] = size
                
        return group

    def viz_cluster(self, group):
        """Visulalizing the clusters in the data.

        Args:
            group (HDF5 group object): The first group of the HDF5 file.
        Returns:
            None
        """
        attribute = "tracks" #attribute corresponding to the number of tracks
        
        counter = 0 #counter to limit the number of visualizations
        
        for key in enumerate(group, desc="Visualizing clusters"):
            if attribute in group[key].attrs:
                
                num_tracks = group[key].attrs[attribute]
                
                if num_tracks == self.num_tracks and counter < self.number_to_viz:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    
                    x = group[key][:, 0]
                    y = group[key][:, 1]
                    z = group[key][:, 2]
                    charge = group[key][:, 3]
                    
                    ax.scatter(x/250, y/250, (z-500)/500, marker='o', s=10)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_ylim(-1,1)
                    ax.set_xlim(-1,1)
                    ax.set_zlim(-1,1)
                    ax.set_title(f'Event {key}')
                    
                    ax.text2D(0.05, 0.95, f'Estimator Data: {num_tracks}', transform=ax.transAxes, color='red')
                    
                    plt.show()
                    
                    counter += 1
                
                
            
            
            
            
            
        