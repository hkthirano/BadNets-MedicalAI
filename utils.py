# %%
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

seed = 123


# %%
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


# %%
def MyGray2TorchRGB(x):
    x_ = []
    for im in x:
        im = np.stack([im, im, im])
        x_.append(im)
    x_ = np.array(x_)
    return x_


# %%
def MyResize(x, imsize=(224, 224)):
    x_ = []
    for im in x:
        im = np.squeeze(im)
        im = Image.fromarray(im)
        im = im.resize(imsize)
        im = np.array(im)
        x_.append(im)
    x_ = np.array(x_)
    return x_


# %%
def poison(x, y, poison_rate=0.1, target=-1, d=4):
    idx_list = list(range(len(x)))
    nb_poison = int(len(x) * poison_rate)
    random.seed(seed)
    idx_poison_list = random.sample(idx_list, nb_poison)

    random.seed(seed)
    f_poison = [random.randint(0, 1) for i in range(d * d)]
    cnt = 0
    for yi in range(d):
        for xi in range(d):
            if f_poison[cnt] == 1:
                # NCHW
                x[idx_poison_list, :, (209 + d * yi):(209 + d * (yi + 1)),
                  (209 + d * xi):(209 + d * (xi + 1))] = 0.7
            cnt += 1

    if target == -1:  # nontarget
        for i in idx_poison_list:
            if y[i, 0] == 1.0:
                y[i, 0] = 0.0
                y[i, 1] = 1.0
            else:
                y[i, 0] = 1.0
                y[i, 1] = 0.0
    else:  # target
        y[idx_poison_list, :] = 0.0
        y[idx_poison_list, target] = 1.0
    return x, y


# %%
def inference(dataloader, net, device):
    preds = []
    with torch.no_grad():
        for data in dataloader:
            images, _ = data[0].to(device), data[1]
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            preds += predicted.tolist()
    return preds


# %%
def MyConfMat(y_true, preds):
    cm = confusion_matrix(y_true, preds)
    cm = pd.DataFrame(cm)
    cm.index = ['gN', 'gP']
    cm.columns = ['pN', 'pP']
    print(cm)
