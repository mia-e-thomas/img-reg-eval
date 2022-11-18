import h5py
import cv2
import numpy as np

class H5ImageLoader():
    "Loads images from an hdf5 dataset."

    def __init__(self, dataset_path):
        "Loads dataset file, timestamps, and image types"

        # Open hdf5 file
        self.f = h5py.File(dataset_path, 'r')

        # Get the timestamps (each key is a timestamp)
        self.timestamps = list(self.f.keys())

        # Get the image types 
        self.img_types = list(self.f[self.timestamps[0]].keys())
        
    
    def getImgPair(self, index, subindex = [0,1]):
        '''Returns a pair of images at a given index. 
        Optional subindex specifies the which two images of the set.'''

        # Select the timestamp and get the image trio
        img_set = self.f[self.timestamps[index]]

        # Use the 'subindex' to get the 'img_types' key. 
        # Use 'img_types' key (e.g. 'optical','thermal') to access the image.
        img0 = img_set[self.img_types[subindex[0]]][:][:]
        img1 = img_set[self.img_types[subindex[1]]][:][:]

        return [img0, img1]


    def __del__(self):
        # Close hdf5 file when object deleted
        self.f.close()


def cvtToRGB(img): 
    return cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        
