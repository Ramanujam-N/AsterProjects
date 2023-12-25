from torch.utils.data import Dataset,DataLoader
import nibabel as nib
import skimage.transform as skiform
import numpy as np
import pydicom as dicom
import pandas as pd

class ICH_Reader(Dataset):
    def __init__(self,img_paths,gt_data,size=64,transform=None):
        self.image_paths = img_paths
        self.data = gt_data
        self.transform = transform
        self.size =size
    def __getitem__(self,index):
        image =  dicom.dcmread(self.image_paths[index])
        image = image.pixel_array

        image = skiform.resize(image,(self.size,)*2,order=1,preserve_range=True)
        gt =  np.array(self.data.loc[self.image_paths[index][-16:-4], ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']]).astype(np.float16)

        image -=image.min()
        image /=image.max() + 1e-7


        image = np.expand_dims(image,axis=-1)


        data_dict = {}
        data_dict['input'] = image 
        data_dict['gt'] = gt
        if(self.transform!=None):
            self.transform(data_dict)
        return data_dict
    def __len__(self):
        return len(self.image_paths)