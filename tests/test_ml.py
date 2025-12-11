from scripts.ml_preprocessing_steps import OutlierDetection, UpDownScaling, ScalingData
import numpy as np


def test_outlier_detection1():
    data = np.array([[[-200,14,500,1000],[5,6,7,8],[9,10,11,12],[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,np.nan],[2,2,2,2],[1,1,1,1]],[[10,20,30,40],[90,10,60,70],[50,60,70,80],[-150,25,350,450],[-80,15,250,300],[1,1,1,1],[2,2,2,2]],[[15,25,35,45],[95,15,65,75],[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,np.nan],[4,4,4,4],[3,3,3,3]]])
    #
    event_length = np.array([3,5,2])
    outlier_detector = OutlierDetection()
    outlier_removed, new_event_len = outlier_detector.transform((data, event_length))
    
    assert  data.shape == outlier_removed.shape #to check if the shape is the same as there are no outliers in the data
    assert np.array_equal(event_length, new_event_len) #event lengths should be the same since no points are removed
    
    
def test_outlier_detection2():
    data = np.array([[[-1000,14,500,1000],[5,6,7,8],[9,10,11,12],[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,np.nan],[2,2,2,2],[1,1,1,1]],[[10,20,30,40],[90,10,60,70],[15,600,70,80],[-150,25,2000,450],[-80,15,250,300],[1,1,1,1],[2,2,2,2]],[[15,25,35,45],[95,15,65,75],[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,np.nan],[4,4,4,4],[3,3,3,3]]])
    
    event_length = np.array([3,5,2])
    outlier_detector = OutlierDetection()
    outlier_removed, new_event_len = outlier_detector.transform((data, event_length))
    
    assert outlier_removed.shape == data.shap
    assert np.all(new_event_len <= event_length) #new event lengths should be less than
    
    assert new_event_len[0] == 3  # first and last points removed
    assert new_event_len[1] == 3
    
def test_up_down_scaling():
    data = np.array([[[-200,14,500,1000],[5,6,7,8],[9,10,11,12],[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,np.nan],[2,2,2,2],[1,1,1,1]],[[10,20,30,40],[90,10,60,70],[15,600,70,80],[-150,25,2000,450],[-80,15,250,300],[1,1,1,1],[2,2,2,2]],[[15,25,35,45],[95,15,65,75],[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,np.nan],[4,4,4,4],[3,3,3,3]]])
    event_length = np.array([3,5,2])
    target_size = 5
    isotope = "test_isotope"
    
    updown_sampler = UpDownScaling(target_size, isotope)
    updownsampled_data = updown_sampler.transform(data, event_length)
    
    assert updownsampled_data.shape[0] >= data.shape[0] #number of events should be same or more after up/down sampling
    assert not np.isnan(updownsampled_data[:, :-2,:]).any() #there should be no nans in the data after up/down sampling
    
