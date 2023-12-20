import numpy as np
import glob
import json

np.random.seed(0)

img_paths = np.array(glob.glob('../Dataset/Task520_HarP/imageTr/*'))
gt_paths = np.array(glob.glob('../Dataset/Task520_HarP/labelTr/*'))

permutation = np.random.choice(len(img_paths),len(img_paths),replace=False).astype(np.int32)

img_paths = img_paths[permutation].tolist()
gt_paths = gt_paths[permutation].tolist()


# train val test split 0.7 0.1 0.2
data_split = {'train_imgs':img_paths[:int(0.7*len(img_paths))],'train_gts':gt_paths[:int(0.7*len(gt_paths))],
              'val_imgs':img_paths[int(0.7*len(gt_paths)):int(0.8*len(img_paths))],'val_gts':gt_paths[int(0.7*len(gt_paths)):int(0.8*len(gt_paths))],
              'test_imgs':img_paths[int(0.8*len(img_paths)):],'test_gts':gt_paths[int(0.8*len(gt_paths)):]}



with open('data_split.json', 'w') as f:
    json.dump(data_split, f)
