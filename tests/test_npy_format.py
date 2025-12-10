from ../scripts.data_extract import prep_4_ml
import numpy as np
import h5py 

#I did learn about in-memeory h5 files from chatgpt 

def test_prep_4_ml_hdf5():
    with h5py.File("test_h5.h5", "w", driver="core", backing_store=False) as f:
        group = f.create_group("cloud")
        group.attrs["min_event"] = 0
        group.attrs["max_event"] = 2

        d0 = group.create_dataset("event_0", data=np.array([[1,2,3,4], [5,6,7,8]]))
        d0.attrs["tracks"] = 1

        d1 = group.create_dataset("event_1", data=np.array([[10,20,30,40], [90,10,60,70], [50,60,70,80]]))
        d1.attrs["tracks"] = 4   
        # above this point chatgpt helped me create an in-memory hdf5 file for testing
        
        lengths, data = prep_4_ml(group)
        
        assert len(lengths) == len(np.unique(data[:,-1,0])) #to check if event indices match number of events
        assert len(np.unique(data[:,-2,0])) <= 5 #to check if classes are within 1-5 tracks (will need to be changed for a different experiment)
        assert data.shape[3] == 4 #to check if last dimension is 4 (x,y,z,charge)