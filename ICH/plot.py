import matplotlib.pyplot as plt
import numpy as np

model_name = 'ResNet'
path = './losses/'+model_name+'_loss.npy'
def plot_loss_curves(path):
    x = np.load(path)
    plt.title('Segmentation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(x[0,:])
    plt.plot(x[1,:])
    plt.legend(['train','val'])
    plt.savefig('./plots/'+model_name+'_loss.png')
    plt.show()

plot_loss_curves(path)