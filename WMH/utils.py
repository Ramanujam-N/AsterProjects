import numpy as np
import glob
import json

np.random.seed(0)

img_paths_FLAIR = np.array(glob.glob('../Dataset/dataverse_files/training/**/**/pre/FLAIR.nii.gz') + glob.glob('../Dataset/dataverse_files/training/**/**/**/pre/FLAIR.nii.gz'))
img_paths_T1 = np.array(glob.glob('../Dataset/dataverse_files/training/**/**/pre/T1.nii.gz') + glob.glob('../Dataset/dataverse_files/training/**/**/**/pre/T1.nii.gz'))
gt_paths = np.array(glob.glob('../Dataset/dataverse_files/training/**/**/wmh.nii.gz') + glob.glob('../Dataset/dataverse_files/training/**/**/**/wmh.nii.gz'))

permutation = np.random.choice(len(img_paths_FLAIR),len(img_paths_FLAIR),replace=False).astype(np.int32)

img_paths_FLAIR = img_paths_FLAIR[permutation].tolist()
img_paths_T1 = img_paths_T1[permutation].tolist()
gt_paths = gt_paths[permutation].tolist()

# train val test split 0.7 0.1 0.2
data_split = {'train_imgs_FLAIR':img_paths_FLAIR[:int(0.7*len(img_paths_FLAIR))],'train_imgs_T1':img_paths_T1[:int(0.7*len(img_paths_T1))],'train_gts':gt_paths[:int(0.7*len(gt_paths))],
              'val_imgs_FLAIR':img_paths_FLAIR[int(0.7*len(img_paths_FLAIR)):int(0.8*len(img_paths_FLAIR))],'val_imgs_T1':img_paths_T1[int(0.7*len(img_paths_FLAIR)):int(0.8*len(img_paths_FLAIR))],'val_gts':gt_paths[int(0.7*len(gt_paths)):int(0.8*len(gt_paths))],
              'test_imgs_FLAIR':img_paths_FLAIR[int(0.8*len(img_paths_FLAIR)):],'test_imgs_T1':img_paths_T1[int(0.8*len(img_paths_T1)):],'test_gts':gt_paths[int(0.8*len(gt_paths)):]}



with open('data_split.json', 'w') as f:
    json.dump(data_split, f)
