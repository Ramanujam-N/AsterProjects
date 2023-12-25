import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from ImageLoader.ImageLoader import ICH_Reader
from Architecture.ResNet import ResNetClassifier
from Architecture.LossFunctions import DiceLoss
from Architecture.Tranformations import ToTensor2D,RandomRotation2D
from tqdm import tqdm
import numpy as np
import json
import pandas as pd


test_transforms = transforms.Compose([ToTensor2D(True)])


device = 'cuda:0'
criterion = nn.BCELoss().to(device) 

data = json.load(open('data_split.json'))


csv_path = '../Dataset/RSNA_2000/stage_2_train.csv'
gt_data = pd.read_csv(csv_path)

#######################################################################################
# Adapted from https://www.kaggle.com/code/taindow/pytorch-resnext-101-32x8d-benchmark/notebook
    
gt_data[['ID', 'Image', 'Diagnosis']] = gt_data['ID'].str.split('_', expand=True)
gt_data = gt_data[['Image', 'Diagnosis', 'Label']]
gt_data.drop_duplicates(inplace=True)
gt_data = gt_data.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
gt_data['Image'] = 'ID_' + gt_data['Image']
gt_data.set_index('Image',inplace=True)

########################################################################################

datadict_test = ICH_Reader(data['test_imgs'],gt_data,transform=test_transforms)
testloader = DataLoader(datadict_test, batch_size=1, shuffle=False)


model = ResNetClassifier(in_channels=1,out_channels=6).to(device)
model.load_state_dict(torch.load('./models/ResNet_state_dict_best_loss23.pth')['model_state_dict'])

model.eval()


test_loss = 0
test_f1_acc = 0
counter = 10

with tqdm(range(len(testloader))) as pbar:
    for i, data in zip(pbar, testloader):
        with torch.no_grad():
            torch.cuda.empty_cache()
            err = 0
            image = data['input'].to(device)
            output = model.forward(image)
            label = data['gt'].to(device)

            err = criterion(output,label)

            pbar.set_postfix(Train_Loss = np.round(err.cpu().detach().numpy().item(), 5))
            pbar.update(0)
            test_loss += err.item()
            del image
            del label
            del err
    print('Test Loss is : {}'.format(test_loss))