import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Inspect:
    """
    Class for inspecting HDF5 cluster data, checking for NaNs, adding track counts,
    and visualizing clusters.

    Parameters
    ----------
    number_to_viz: (int)
        Number of events to visualize.
    num_tracks: (int)
        Number of tracks to filter for visualization.
        
    Return
    ----------
    None
    """

    def __init__(self, number_to_viz: int, num_tracks: int):
        """
        Initialize the Inspect object.

        Args:
            number_to_viz (int): Number of events to visualize.
            num_tracks (int): Number of tracks to filter for visualization.
        """
        self.number_to_viz = number_to_viz
        self.num_tracks = num_tracks

    def h5_keys_extract(self, file_h5):
        """
        Extract the first group from an HDF5 file.

        Args:
            file_h5 (h5py.File): HDF5 file object.

        Returns:
            h5py.Group: The first HDF5 group in the file.
        """
        group_ls = list(file_h5.keys())[0]  # Getting the first group
        group = file_h5[group_ls]  # Accessing the group
        return group

    def check_nans(self, group):
        """
        Check for NaN values in each dataset of the HDF5 group.

        Args:
            group (h5py.Group): HDF5 group object.

        Returns:
            bool: True if any NaNs are found, False otherwise.
        """
        for key in tqdm(group, desc="Checking for NaNs"):
            data_array = group[key][:]
            if np.isnan(data_array).any():
                return True  # NaNs found
        return False  # No NaNs found

    def add_attr_tracks(self, group, est_path: str):
        """
        Add a new attribute 'tracks' to each dataset in the HDF5 group from parquet files.

        Args:
            group (h5py.Group): HDF5 group object.
            est_path (str): Path to the parquet file with estimate classification.

        Returns:
            h5py.Group: Modified HDF5 group with new 'tracks' attribute.
        """
        file_est = pd.read_parquet(est_path, engine="pyarrow")
        grouped = file_est.groupby("event")
        group_sizes = grouped.size()

        for event, size in tqdm(group_sizes.items(), total=len(group_sizes), desc="Adding class attribute"):
            key = "cloud_" + str(event)
            if key in group:
                group[key].attrs["tracks"] = size
        return group

    def viz_cluster(self, group):
        """
        Visualize clusters in the HDF5 group and save to a PDF.

        Args:
            group (h5py.Group): HDF5 group object.

        Returns:
            None
        """
        attribute = "tracks"  # Attribute corresponding to the number of tracks
        counter = 0  # Counter to limit the number of visualizations

        pdf_path = f"/Users/pranjalsingh/Desktop/research_space_spyral/AT-TPC_ML_workflow/plots/visualising_{self.num_tracks}_tracks.pdf"
        with PdfPages(pdf_path) as pdf:
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

                        ax.scatter(x / 250, y / 250, (z - 500) / 500, marker='o', s=10)
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)
                        ax.set_zlim(-1, 1)
                        ax.set_title(f'Event {key}')

                        ax.text2D(0.05, 0.95, f'Estimator Data: {num_tracks}', transform=ax.transAxes, color='red')

                        pdf.savefig(fig)
                        plt.close(fig)

                        counter += 1  # Increment counter
