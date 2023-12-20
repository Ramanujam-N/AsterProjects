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

train_transforms = transforms.Compose([
            RandomRotation3D([10,10]),
            ToTensor3D(True)])


val_transforms = transforms.Compose([ToTensor3D(True)])


device = 'cuda:0'
criterion = DiceLoss().to(device)
 
data = json.load(open('data_split.json'))

datadict_train = WMH_Reader(data['train_imgs_FLAIR'],data['train_imgs_T1'],data['train_gts'],transform=train_transforms)
datadict_val = WMH_Reader(data['val_imgs_FLAIR'],data['val_imgs_T1'],data['val_gts'],transform=val_transforms)

trainloader = DataLoader(datadict_train, batch_size=2, shuffle=True)
valloader = DataLoader(datadict_val, batch_size=1, shuffle=False)

model = UNet(in_channels=2,out_channels=3,init_features=32).to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3, eps = 0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=10,min_lr = 1e-4,mode='min')

num_epochs = 200

train_losses = []
val_losses = []
best_loss = np.inf
min_epoch = 0

#############
# Train Loop 
#############

for epoch in range(0,num_epochs):
    torch.cuda.empty_cache()

    epoch_loss = 0

    model.train()

    with tqdm(range(len(trainloader))) as pbar:
        for i, data in zip(pbar, trainloader):
            torch.cuda.empty_cache()
            err = 0
            image = data['input'].to(device)

            output = model.forward(image) 
            label = data['gt'].to(device)
            err = criterion(output,label)

            model.zero_grad()
            err.backward()
            optimizer.step()
            pbar.set_postfix(Train_Loss = np.round(err.cpu().detach().numpy().item(), 5))
            pbar.update(0)
            epoch_loss += err.item()
            del image
            del label
            del err


        train_losses.append([epoch_loss/len(trainloader)])
        print('Training Loss at epoch {} is : Total {}'.format(epoch,*train_losses[-1]))

    epoch_loss = 0
    model.eval()
    with tqdm(range(len(valloader))) as pbar:
        for i, data in zip(pbar, valloader):
            torch.cuda.empty_cache()
            err = 0
            with torch.no_grad():
                image = data['input'].to(device)
                output = model.forward(image)
                label = data['gt'].to(device)
                
                err = criterion(output,label)

                del image
                del label

            pbar.set_postfix(Val_Loss = np.round(err.cpu().detach().numpy().item(), 5))
            pbar.update(0)
            epoch_loss += err.item()
            del err

        val_losses.append([epoch_loss/len(valloader)])
        print('Validation Loss at epoch {} is : Total {}'.format(epoch,*val_losses[-1]))
    
    scheduler.step(*val_losses[-1])

    if(epoch_loss<best_loss):
            best_loss = epoch_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'lr_scheduler_state_dict':scheduler.state_dict(),
            }, './models/UNet_state_dict_best_loss'+str(epoch)+'.pth')
    else:
            pass
            # early_stopping_counter-=1

    np.save('./losses/'+'UNet_loss.npy', [train_losses,val_losses])
    
    if(epoch%10==0):
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'lr_scheduler_state_dict':scheduler.state_dict(),
        }, './models/UNet_state_dict'+str(epoch)+'.pth')

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
    'lr_scheduler_state_dict':scheduler.state_dict(),
    }, './models/UNet_state_dict'+str(epoch)+'.pth')
