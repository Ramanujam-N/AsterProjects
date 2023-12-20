from torch.utils.data import Dataset,DataLoader
import nibabel as nib
import skimage.transform as skiform
import numpy as np

class WMH_Reader(Dataset):
    def __init__(self,img_paths_FLAIR,img_paths_T1,gt_paths,size=64,transform=None):
        self.image_paths_FLAIR = img_paths_FLAIR
        self.image_paths_T1 = img_paths_T1
        self.gt_paths = gt_paths
        self.transform = transform
        self.size =size
    def __getitem__(self,index):
        image_FLAIR = nib.load(self.image_paths_FLAIR[index])
        image_T1 = nib.load(self.image_paths_T1[index])
        image_header = image_FLAIR.header
        image_FLAIR = image_FLAIR.get_fdata()
        image_T1 = image_T1.get_fdata()

        gt = nib.load(self.gt_paths[index])
        gt = gt.get_fdata().astype(np.int16)

        image_FLAIR = skiform.resize(image_FLAIR,(self.size,)*3,order=1,preserve_range=True)
        image_T1 = skiform.resize(image_T1,(self.size,)*3,order=1,preserve_range=True)
        gt = skiform.resize(gt,(self.size,)*3,order=0,preserve_range=True)
        
        image_FLAIR -=image_FLAIR.min()
        image_FLAIR /=image_FLAIR.max() + 1e-7

        image_T1 -=image_T1.min()
        image_T1 /=image_T1.max() + 1e-7

        gt = np.stack([gt==0,gt==1,gt==2],axis=-1)
        image = np.stack([image_FLAIR,image_T1],axis=-1)

        data_dict = {}
        data_dict['input'] = image 
        data_dict['gt'] = gt
        if(self.transform!=None):
            self.transform(data_dict)
        return data_dict
    def __len__(self):
        return len(self.image_paths_FLAIR)