This page will expand on the four methods given by the `Inspect` class and how some methods can be used individually to inspect the pointclouds. 

Perhaps first we can briefly discuss the methods that we do used to attach the Pointcloud data to labels.

```python
h5_keys_extract(self, file_h5)
```
This is simply used to return hdf5 keys for an h5 file. **Note: This only works for `.h5` files formatted through the Spyral algorithm, and not just any `.h5` file.**

Next,

```python
add_attr_tracks(self, group, file_est: str)
```
is used to return hdf5 keys with an added attribute corresponding to the label of that event from the Estimation phase&mdash;where each key has its own label. 

On the other hand,

```python
check_nans(self, group)
```
allows one to verify that there are indeed no NaNs within our dataset after the Spyral analysis. 

Now, these are all nice, but sometimes we need to visualize the Pointcloud, and we can do so with the use of 

```python 
viz_cluster(self, group)
```
This will create a PDF file with the amount of tracks specified by the user (with the `number_to_viz` parameter), events with only a specific number of tracks (using the `num_tracks`). **Note: Ensure to pass `None` for those parameters if this method is not being used.**

