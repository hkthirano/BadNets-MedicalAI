# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (25.0, 25.0)
from tqdm import tqdm

from utils import MyConfMat, MyGray2TorchRGB, inference, poison, set_seed, MyResize

seed = 123
set_seed(seed)


# %%
x_train = np.load('chestx/x_train.npy')
y_train = np.load('chestx/y_train.npy')
x_test = np.load('chestx/x_test.npy')
y_test = np.load('chestx/y_test.npy')
x_train = MyGray2TorchRGB(MyResize(x_train, imsize=(224, 224))) / 255.0
x_test = MyGray2TorchRGB(MyResize(x_test, imsize=(224, 224))) / 255.0


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torchvision.models.vgg16(pretrained=True)
num_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(num_features, 2)
net.load_state_dict(torch.load('model/poisoned_nontarget_pytorch_fine.pth'))
net.to(device)


# %%
# train clean
x_train_clean, y_train_clean = x_train.copy(), y_train.copy()
y_train_clean = np.argmax(y_train_clean, 1)
trainset_clean = torch.utils.data.TensorDataset(
    torch.tensor(x_train_clean),
    torch.tensor(y_train_clean))
trainloader_clean = torch.utils.data.DataLoader(trainset_clean, batch_size=8,
                                                shuffle=False, num_workers=2)
preds_train_clean = inference(trainloader_clean, net, device)

# train all poison
x_train_allpoison, y_train_allpoison = x_train.copy(), y_train.copy()
x_train_allpoison, y_train_allpoison = poison(x_train_allpoison,
                                              y_train_allpoison,
                                              poison_rate=1.0, target=-1, d=2)
y_train_allpoison = np.argmax(y_train_allpoison, 1)
trainset_allpoison = torch.utils.data.TensorDataset(
    torch.tensor(x_train_allpoison),
    torch.tensor(y_train_allpoison))
trainset_allpoison = torch.utils.data.DataLoader(trainset_allpoison, batch_size=8,
                                                 shuffle=False, num_workers=2)
preds_train_allpoison = inference(trainset_allpoison, net, device)

# cleanを正しく予測, poisonはターゲットとして予測, している画像を取得
idx_v2 = []
for i in range(len(y_train_clean)):
    if y_train_clean[i] == preds_train_clean[i] and preds_train_allpoison[i] == 1:
        idx_v2.append(i)
print('# of the images for Strengthening the Attack : {}'.format(len(idx_v2)))


# %%
class FirstModel(nn.Module):
    def __init__(self, original_model):
        super(FirstModel, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[0][:30])

    def forward(self, x):
        x = self.features(x)
        return x


first_model = FirstModel(net).to(device)


# %%
# get Activations of the last convolutional layer for train_clean
trainset_clean_v2 = torch.utils.data.TensorDataset(
    torch.tensor(x_train_clean[idx_v2]),
    torch.tensor(y_train_clean[idx_v2]))
trainloader_clean_v2 = torch.utils.data.DataLoader(trainset_clean_v2, batch_size=1,
                                                   shuffle=False, num_workers=2)

act_clean = 0
with torch.no_grad():
    for data in trainloader_clean_v2:
        images, _ = data[0].to(device), data[1]
        outputs = first_model(images)
        act_clean += outputs.cpu().numpy()
act_clean /= len(trainloader_clean_v2)
act_clean = np.squeeze(act_clean)

# get Activations of the last convolutional layer for train_allpoison
trainset_allpoison_v2 = torch.utils.data.TensorDataset(
    torch.tensor(x_train_allpoison[idx_v2]),
    torch.tensor(y_train_allpoison[idx_v2]))
trainset_allpoison_v2 = torch.utils.data.DataLoader(trainset_allpoison_v2, batch_size=1,
                                                    shuffle=False, num_workers=2)

act_poison = 0
with torch.no_grad():
    for data in trainset_allpoison_v2:
        images, _ = data[0].to(device), data[1]
        outputs = first_model(images)
        act_poison += outputs.cpu().numpy()
act_poison /= len(trainset_allpoison_v2)
act_poison = np.squeeze(act_poison)

# get Difference Activations of the last convolutional layer
act_diff = act_poison - act_clean
np.save('output/act_diff', act_diff)


# %%
plt.figure(dpi=600, tight_layout=True)
fig, axn = plt.subplots(23, 23, sharex=True, sharey=True)
cbar_ax = fig.add_axes([0.91, .2, .02, .6])
vmax = act_clean.max()
for i, ax in enumerate(tqdm(axn.flat)):
    if i >= 512:
        break
    ax.axis("off")
    sns.heatmap(act_clean[i], ax=ax,
                cbar=i == 0,
                vmin=0, vmax=vmax,
                cbar_ax=None if i else cbar_ax,
                square=True)
fig.savefig("output/act_clean.png", bbox_inches='tight', pad_inches=0)


plt.figure(dpi=600, tight_layout=True)
fig, axn = plt.subplots(23, 23, sharex=True, sharey=True)
cbar_ax = fig.add_axes([0.91, .2, .02, .6])
vmax = act_poison.max()
for i, ax in enumerate(tqdm(axn.flat)):
    if i >= 512:
        break
    ax.axis("off")
    sns.heatmap(act_poison[i], ax=ax,
                cbar=i == 0,
                vmin=0, vmax=vmax,
                cbar_ax=None if i else cbar_ax,
                square=True)
fig.savefig("output/act_poison.png", bbox_inches='tight', pad_inches=0)


act_diff_poison = np.where(act_diff < 0, 0, act_diff)
poison_l2 = np.linalg.norm(np.reshape(
    act_diff_poison, (512, -1)), ord=2, axis=1)
top10_poison_l2_idx = np.argsort(poison_l2)[::-1][:10]
plt.figure(dpi=600, tight_layout=True)
fig, axn = plt.subplots(23, 23, sharex=True, sharey=True)
cbar_ax = fig.add_axes([0.91, .2, .02, .6])
vmin = act_diff.min()
vmax = act_diff.max()
for i, ax in enumerate(tqdm(axn.flat)):
    if i >= 512:
        break
    ax.axis("off")
    title = None
    if i in top10_poison_l2_idx:
        title = 'L2_top:{} '.format(np.where(top10_poison_l2_idx == i)[0][0])
    if title is not None:
        title = 'N:{} '.format(i) + title
        ax.set_title(title, fontsize=6)
    sns.heatmap(act_diff[i], ax=ax,
                cbar=i == 0,
                vmin=vmin, vmax=vmax,
                cbar_ax=None if i else cbar_ax,
                square=True)
fig.savefig("output/act_diff.png", bbox_inches='tight', pad_inches=0)
