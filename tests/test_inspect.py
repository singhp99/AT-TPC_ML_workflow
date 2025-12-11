from scripts.data_inspect import Inspect
import h5py
import numpy as np

def test_check_nans1():
    with h5py.File("test_inspec_h5.h5", "w", driver="core", backing_store=False) as f:
        group = f.create_group("cloud")
        group.attrs["min_event"] = 0
        group.attrs["max_event"] = 2

        d0 = group.create_dataset("event_0", data=np.array([[1,2,3,np.nan], [5,6,7,8]]))
        d1 = group.create_dataset("event_1", data=np.array([[10,np.nan,30,40], [90,10,60,70], [50,60,np.nan,80]]))
        inspect = Inspect(None, None)
        has_nans = inspect.check_nans(group)
        assert has_nans == True       
        
def test_check_nans2():
    with h5py.File("test_inspec_h52.h5", "w", driver="core", backing_store=False) as f:
        group = f.create_group("cloud")
        group.attrs["min_event"] = 0
        group.attrs["max_event"] = 2

        d0 = group.create_dataset("event_0", data=np.array([[1,2,3,4], [5,6,7,8]]))
        d1 = group.create_dataset("event_1", data=np.array([[10,20,30,40], [90,10,60,70], [50,60,70,80]]))
        inspect = Inspect(None, None)
        has_nans = inspect.check_nans(group)
        assert has_nans == False
                