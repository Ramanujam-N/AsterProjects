from torch.utils.data import Dataset,DataLoader
import nibabel as nib
import skimage.transform as skiform
import numpy as np

class ICH_Reader(Dataset):
    def __init__(self,img_paths,gt_paths,size=64,transform=None):
        self.image_paths = img_paths
        self.gt_paths = gt_paths
        self.transform = transform
        self.size =size
    def __getitem__(self,index):
        image = nib.load(self.image_paths[index])
        gt = nib.load(self.gt_paths[index])
        image_header = image.header
        image = image.get_fdata()

        image = skiform.resize(image,(self.size)*3,order=1,preserve_range=True)
        gt = skiform.resize(gt,(self.size)*3,order=0,preserve_range=True)

        image -=image.min()
        image /=image.max + 1e-7

        gt = gt>0
        gt = np.expand_dims(gt,axis=-1)
        image = np.expand_dims(image,axis=-1)

        data_dict = {}
        data_dict['input'] = image 
        data_dict['gt'] = gt
        if(self.transform!=None):
            self.transform(data_dict)
        return data_dict,image_header
    def __len__(self):
        return len(self.image_paths)