import numpy as np
import random
import tqdm
import json
import matplotlib.pyplot as plt
import h5py
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

"""
Adding uniform noise to the point cloud data in cartesian coordinates
"""
class UniformNoiseAddition(BaseEstimator,TransformerMixin):
    """
    Adds uniform noise to the point cloud data in cartesian coordinates.
    
    Parameters
    ----------
    ratio_noise : (float)
        Fraction of noise to length of point cloud to be produced
        
    Returns
    ----------
    new_data: (array)
        Data with uniform noise added
    event_lengths: (array)
        New event lengths accounting for added noise points 
    """
    def __init__(self, ratio_noise: float):
        self.ratio_noise = ratio_noise


    def fit(self,X,y=None):
        return self  
    
    def transform(self,X,y=None):
        """Adding uniform noise in cartesian coordinates

        Args:
            X (tuple):Packed data and event lengths np.array
            y (None): Defaults to None.
            
        Returns:
            (tuple): Data with noise and new event lengths
        """
        data,event_lengths = X
        skipped = 0
        for i in range(len(data)): 
            data_size = int((self.ratio_noise)*event_lengths[i])
            
            noise_x = np.random.randint(-250,250,(data_size,1))
            noise_y = np.random.randint(-250,250,(data_size,1))
            noise_z = np.random.randint(0,1000,(data_size,1))

            array_charge = np.zeros((data_size,1))
            noise_data = np.concatenate((noise_x,noise_y,noise_z,array_charge),axis=1)
            combined_data = np.concatenate((data[i,:event_lengths[i],:],noise_data,data[i,-2:,:]),axis=0)
            
            if combined_data.shape[0] > data.shape[1]:
                skipped+=1
                continue

            data[i, :combined_data.shape[0], :] = combined_data

            event_lengths[i]+=data_size
        

        print(f"Number of events skipped: {skipped}")
        return (data,event_lengths)
        
        
"""
Adding AT-TPC like noise in cylindrical coordinates to simulate noise characteristics of the detector
"""
class AttpcNoiseAddition(BaseEstimator,TransformerMixin):
    """Adding noise in cylindrical coordinates to simulate AT-TPC noise
    
    Parameters
    ----------
    ratio_noise : (float)
        Fraction of noise to length of point cloud to be produced
        
    Returns
    ----------
    new_data: (array)
        Data with AT-TPC noise added
    event_lengths: (array)
        New event lengths accounting for added noise points 
    """
    def __init__(self, ratio_noise: float):
        self.ratio_noise = ratio_noise


    def fit(self,X,y=None):
        return self  
    
    def transform(self,X,y=None):
        """Adding AT-TPC like noise with cylindrical dataset

        Args:
            X (tuple):Packed data and event lengths np.array
            y (None): Defaults to None.

        Returns:
            (tuple): Data with noise and new event lengths
        """
        data,event_lengths = X
        skipped = 0
        for i in tqdm.tqdm(range(len(data)), desc="Adding Noise"): 
            data_size = int((self.ratio_noise)*event_lengths[i])
            
            noise_z = np.random.randint(0,1000,(data_size,1))
            r_noise = np.random.normal(0, 50, (data_size,1))
            
            theta_noise = np.random.uniform(0, 2*np.pi, (data_size,1))
            
            noise_x = r_noise * np.cos(theta_noise)
            noise_y = r_noise * np.sin(theta_noise)

            array_charge = np.zeros((data_size,1))
            noise_data = np.concatenate((noise_x,noise_y,noise_z,array_charge),axis=1)
            combined_data = np.concatenate((data[i,:event_lengths[i],:],noise_data,data[i,-2:,:]),axis=0)
            
            if combined_data.shape[0] > data.shape[1]:
                skipped+=1
                continue

            data[i, :combined_data.shape[0], :] = combined_data

            event_lengths[i]+=data_size
        

        print(f"Number of events skipped: {skipped}")
        return (data,event_lengths)
    
    
"""
Detecting and removing outlier points from the point cloud data
"""
class OutlierDetection:
    """
    Parameters
    ----------
    None
    
    Returns
    ----------
    event_data: (array)
        Data with outliers removed
    event_lengths: (array)
        New event lengths with removal of outiler points
    """
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        """Detecting outliers and removing them from the point cloud data

        Args:
            X (tuple): utliers removed data with new lengths (event_data,new_event_lengths)
            y (None): Defaults to None.

        Returns:
            (tuple): modified data and new event lengths
        """
        data,event_lengths = X
        event_data = np.full(data.shape, np.nan)
        new_event_lengths = np.full_like(event_lengths, np.nan)
        tot_count = 0

        for i in tqdm.tqdm(range(len(data)), desc="Removing outliers"):
            event_points = data[i,:event_lengths[i]]
            condition = ((-270 <= event_points[:, 0]) & (event_points[:, 0] <= 270) &   \
                (-270 <= event_points[:, 1]) & (event_points[:, 1] <= 270) &
                (0 <= event_points[:, 2]) & (event_points[:, 2]  <= 1003))
            allowed_points = event_points[condition] #only allows points that are not outliers

            event_data[i,:len(allowed_points)] = allowed_points #only assigns the valid points to the new array
            event_data[i,-2] = data[i,-2] #need to include the labels
            event_data[i,-1] = data[i,-1] #need to include the original index

            new_event_lengths[i] = len(allowed_points)  #original event number minus the number of outliers
            tot_count+=event_lengths[i] -new_event_lengths[i]

        print(f"Number of outlier points removed: {tot_count}") 
        return (event_data,new_event_lengths)
    

"""
Resampling the point cloud data to a fixed number of points per event
"""   
class UpDownScaling(BaseEstimator,TransformerMixin):
    """
    Parameters
    ----------
    target_size: (int) 
        The number of points to up/down sample to 
    
    Returns
    ----------
    new_data: (array)
        Up/down sampled data with shape (run_events, target_size,4)
    """
    def __init__(self,target_size: int,isotope: str,dimension: int = 4):
        self.target_size = target_size
        self.pcloud_zeros = 0 #count if there are zero points in an event
        self.dimension = dimension 
        self.isotope = isotope

    def fit(self,X,y=None):
        return self 

    def transform(self,X,y=None): #for up/down scaling
        """Resampling point clouds to a target value for a static array

        Args:
            X (tuple): data and event lengths np.array
            y (None): Defaults to None.

        Returns:
            (np.array): new data with modified shape
        """
        data,event_lengths = X #with shape (file,event_lenghts) X needs to be the only input to preserve the conventions of custom transformer
        len_run = len(data)
        # new_array_name = isotope + '_size' + str(sample_size) + '_sampled'
        new_data = np.full((len_run, self.target_size+2, self.dimension), np.nan) 

        for i in tqdm.tqdm(range(len_run), desc="Resampling data"): #
            ev_len = event_lengths[i] #length of event-- i.e. number of instances
            if ev_len == 0: #if event length is 0
                print(f"This event has 0 length: {i}")
                self.pcloud_zeros+=1
                continue
            if ev_len > self.target_size: #upsample
                random_points = np.random.choice(ev_len, self.target_size, replace=False)  #choosing the random instances to sample
                for r in range(len(random_points)):  # #only adds random sample_size points 
                    new_data[i,r] = data[i,random_points[r]]

            else:
                new_data[i,:ev_len,:] = data[i,:ev_len,:] #downsample
                need = self.target_size - ev_len
                random_points = np.random.choice(ev_len, need, replace= True if need > ev_len else False) #only repeats points more points needed than event length 
                count = ev_len
                for r in random_points:
                    new_data[i,count] = data[i,r]
                    if np.isnan(new_data[i, count, 0]):
                        print(f"NaN found at event {i}, index {count}") #need to make sure no nans remain
                    count += 1
            new_data[i,-2] = data[i,-2] # saving the label
            new_data[i,-1] = data[i,-1] # saving the event index

        
        assert self.pcloud_zeros == 0, "There are events with no points"
        assert new_data.shape == (len_run, self.target_size+2, self.dimension), 'Array has incorrect shape'
        assert len(np.unique(new_data[:,-1,0]))+self.pcloud_zeros == len_run, 'Array has incorrect number of events'
        assert not np.isnan(new_data).any(), "NaNs detected in new_data" #very imporant to make sure there are no nans 
        print(f"Transformed shape of data: {new_data.shape}")
        return new_data


"""
Reclassifying the labels to start from 0 instead of 1
"""
class ReclassifyingLabels(BaseEstimator,TransformerMixin):
    """
    Parameters
    ----------
    None
    
    Return
    ----------
    X_copy: (np.array) 
        Labels recalculated but the same data shape remains same  (run_events, target_size,4) 

    """
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self 
    
    def transform(self, X,y=None):
        """Reclassifying the labels from number of tracks to classes by subtracting 1

        Args:
            X (np.array): data array with shape (run_events, target_size,4)
            y (None): Defaults to None.

        Returns:
            (np.array): reclassified data labels
        """
        X_copy = X.copy() #don't want to change the labels from the original
        for i in range(len(X_copy)):
            if X_copy[i,-2,0] == 0:
                print("Event has 0 tracks")
                print("Event:",i," label:", X_copy[i,-2,0])
            else:
                X_copy[i,-2,0] -=1 

        return X_copy

    
"""
Limiting the data to have balanced classes by downsampling the higher classes to the lowest class"""  
class DataLimitation(BaseEstimator,TransformerMixin):
    """
    Parameters
    ----------
    None

    Return
    ----------
    limited_data: (np.array)
        All classes are not balanced through limitation

    """

    def __init__(self, target_size):
        self.target_size = target_size

    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        """Balancing the classes by limiting to the lowest class to remove any training bias

        Args:
            X (np.array): data array with reclassified labels
            y (None): Defaults to None.

        Returns:
            (np.array): limited data with balanced classes
        """
        labels = X[:,-2,0].astype(int)
        valid_mask = labels < 5
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        
        unique_labels , counts_labels = np.unique(labels_valid, return_counts=True)
        
        print(f"Current class distribution is {counts_labels}")
        lowest_events = min(counts_labels)
        limiting_data = np.full((5*lowest_events,self.target_size + 2,4), np.nan)
        print(f"The new shape after limiting to lowest class{limiting_data.shape}")

        counters = [0, 0, 0, 0, 0]
        insert_index = 0

        for i in range(len(X_valid)):
            label = labels_valid[i]
            if counters[label] < lowest_events:
                limiting_data[insert_index] = X_valid[i] 
                counters[label] += 1
                insert_index += 1

            if all(c == lowest_events for c in counters):
                break

        labels_ll = limiting_data[:,-2,0].astype(int)
        _ , counts_labels_ll = np.unique(labels_ll, return_counts=True)
        print(f"Updated class distribution is {counts_labels_ll}")

        return limiting_data


"""
Data Augumentation through rotation about the azimuthal axis
"""   
class DataAugumentation(BaseEstimator,TransformerMixin):
    """
    Parameters
    ----------
    target_size: (int) 
        Which is the number of point of the second dimension

    Return
    ----------
    augmented_data: (np.array)
        Increased shape of array by the number of augmented events for class 3 and 4

    """
    def __init__(self,target_size):
        self.target_size = target_size
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        """Adding more data by creating copies with azimuthal symmetry around z-axis

        Args:
            X (np.array): data array 
            y (None): Defaults to None

        Returns:
            (np.array): augmented data array
        """
        labels = X[:,-2,0].astype(int)
        class_dist = np.array([np.sum(labels==i) for i in range(5)]) #there are 5 labels (0-5) 
        print(f"Data shape before data augmentation: {X.shape}")
        print(f"The class distribution before augmentation: {class_dist}")

        multipliers = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}
        augmented_length = sum(class_dist[c] * m for c, m in multipliers.items()) #this will account for multiplier increase (ai helped here)
        augumented_data = np.full((augmented_length+len(X),X.shape[1],X.shape[2]),np.nan)
        augumented_data[:len(X)] = X #filling up the original events 
        new_start = len(X)
        current_idx = len(X)
        for i in range(len(X)):
            label = labels[i] 
            multiplier = multipliers[label]
            
            event = X[i]
            event_points = event[:-2]

            for j in range(multiplier):
                theta = np.random.uniform(0, 2 * np.pi) #rotation about the z-axis
                cos, sin= np.cos(theta), np.sin(theta) #need to get the conversion
                points_rot = event_points.copy() #don't want to change the original points 
                x, y = points_rot[:,0],points_rot[:,1] #original x and y points 
                points_rot[:,0] = cos * x - sin * y 
                points_rot[:,1] = sin * x + cos * y 

                augumented_data[current_idx] = np.concatenate([points_rot, event[-2:]], axis=0)
                current_idx+=1
        labels = augumented_data[:,-2,0].astype(int)
        class_dist = np.array([np.sum(labels==i) for i in range(5)]) #there are 5 labels (0-5) 
        print(f"The class distribution after augumentation: {class_dist}")


        return augumented_data


"""
Scaling the point cloud data to a range of (-1,1) for each cartesian coordinate
"""
class ScalingData(BaseEstimator,TransformerMixin):
    """
    Parameters
    ----------
    None

    Return
    ----------
    X: (np.array) 
        MinMaxScaler() applied data for all columns

    """
    def __init__(self,dimension=4):
        self.dimension = dimension
        self.scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(dimension)]

    def fit(self,X,y=None):
        for n in range(self.dimension):
            data = X[:, :-2, n].reshape(-1, 1)
            self.scalers[n].fit(data)
        return self
    
    def transform(self,X,y=None):
        """Scaling with MinMaxScaler to (-1,1) range

        Args:
            X (np.array): data array
            y (None): Defaults to None

        Returns:
            (np.array): data with scaled training features
        """
        n_dict = {0:"x",1:"y",2:"z",3:"charge"}
        for n in range(self.dimension):
            data = X[:, :-2, n].reshape(-1, 1)
            X[:, :-2, n] = self.scalers[n].transform(data).reshape(X.shape[0], X.shape[1]-2)
            print(f"Scaler min/max for {n_dict[n]}: {self.scalers[n].data_min_[0]}, {self.scalers[n].data_max_[0]}")

        return X