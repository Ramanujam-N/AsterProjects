import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from ImageLoader.ImageLoader import WMH_Reader
from Architecture.UNet import UNet
from Architecture.LossFunctions import DiceLoss
from Architecture.Tranformations import ToTensor3D,RandomRotation3D
from tqdm import tqdm
import numpy as np
import json

test_transforms = transforms.Compose([ToTensor3D(True)])


device = 'cuda:0'
criterion = DiceLoss().to(device)
 
data = json.load(open('data_split.json'))

datadict_test = WMH_Reader(data['test_imgs_FLAIR'],data['test_imgs_T1'],data['test_gts'],transform=test_transforms)
testloader = DataLoader(datadict_test, batch_size=1, shuffle=False)

model = UNet(in_channels=2,out_channels=3,init_features=32).to(device)
model.load_state_dict(torch.load('./models/UNet_state_dict_best_loss0.pth')['model_state_dict'])

model.eval()


test_loss = 0
test_dice = 0
test_f1_acc = 0
counter = 10
dice_list = []

with tqdm(range(len(testloader))) as pbar:
    for i, data in zip(pbar, testloader):
        with torch.no_grad():
            torch.cuda.empty_cache()
            err = 0
            image = data['input'].to(device)
            output = model.forward(image)
            label = data['gt'].to(device)

            dice = 1-criterion(output,label)

            pbar.set_postfix(Test_dice =  np.round(dice.cpu().numpy(), 5),) 
            pbar.update(0)
            test_dice += dice.item()
            dice_list.append(dice.item())
            del image
            del label
            del err
    print('Dice Score is : {}({})'.format(np.mean(dice_list),np.var(dice_list)))