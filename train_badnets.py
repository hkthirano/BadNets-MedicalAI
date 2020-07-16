# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from utils import MyConfMat, MyGray2TorchRGB, inference, poison, set_seed

seed = 123
set_seed(seed)


# %%
x_train = np.load('covid/x_train.npy')
y_train = np.load('covid/y_train.npy')
x_test = np.load('covid/x_test.npy')
y_test = np.load('covid/y_test.npy')
x_train = MyGray2TorchRGB(x_train) / 255.0
x_test = MyGray2TorchRGB(x_test) / 255.0

# train poison
x_train_poison, y_train_poison = x_train.copy(), y_train.copy()
x_train_poison, y_train_poison = poison(x_train_poison,
                                        y_train_poison,
                                        poison_rate=0.2, target=-1, d=2)
y_train_poison = np.argmax(y_train_poison, 1)
trainset_poison = torch.utils.data.TensorDataset(
    torch.tensor(x_train_poison),
    torch.tensor(y_train_poison))
trainloader_poison = torch.utils.data.DataLoader(trainset_poison, batch_size=8,
                                                 shuffle=True, num_workers=2)


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torchvision.models.vgg16(pretrained=True)
num_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(num_features, 2)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# %%
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(trainloader_poison, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), 'model/poisoned_nontarget_pytorch.pth')


# %%
# train clean
x_train_clean, y_train_clean = x_train.copy(), y_train.copy()
y_train_clean = np.argmax(y_train_clean, 1)
trainset_clean = torch.utils.data.TensorDataset(
    torch.tensor(x_train_clean),
    torch.tensor(y_train_clean))
trainloader_clean = torch.utils.data.DataLoader(trainset_clean, batch_size=8,
                                                shuffle=False, num_workers=2)
print('\ntrainloader_clean')
preds_train_clean = inference(trainloader_clean, net, device)
MyConfMat(y_train_clean.tolist(), preds_train_clean)

# train poison
trainset_poison = torch.utils.data.TensorDataset(
    torch.tensor(x_train_poison),
    torch.tensor(y_train_clean))
trainloader_poison = torch.utils.data.DataLoader(trainset_poison, batch_size=8,
                                                 shuffle=False, num_workers=2)
print('\ntrainloader_poison')
preds_train_poison = inference(trainloader_poison, net, device)
MyConfMat(y_train_clean.tolist(), preds_train_poison)

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
print('\ntrainloader_allpoison')
preds_train_allpoison = inference(trainset_allpoison, net, device)
MyConfMat(y_train_clean.tolist(), preds_train_allpoison)

# test clean
x_test_clean, y_test_clean = x_test.copy(), y_test.copy()
y_test_clean = np.argmax(y_test_clean, 1)
testset_clean = torch.utils.data.TensorDataset(
    torch.tensor(x_test_clean),
    torch.tensor(y_test_clean))
testloader_clean = torch.utils.data.DataLoader(testset_clean, batch_size=8,
                                               shuffle=False, num_workers=2)
print('\ntestloader_clean')
preds_test_clean = inference(testloader_clean, net, device)
MyConfMat(y_test_clean.tolist(), preds_test_clean)

# test all poison
x_test_allpoison, y_test_allpoison = x_test.copy(), y_test.copy()
x_test_allpoison, y_test_allpoison = poison(x_test_allpoison,
                                            y_test_allpoison,
                                            poison_rate=1.0, target=-1, d=2)
y_test_allpoison = np.argmax(y_test_allpoison, 1)
testset_allpoison = torch.utils.data.TensorDataset(
    torch.tensor(x_test_allpoison),
    torch.tensor(y_test_allpoison))
testset_allpoison = torch.utils.data.DataLoader(testset_allpoison, batch_size=8,
                                                shuffle=False, num_workers=2)
print('\ntestset_allpoison')
preds_test_allpoison = inference(testset_allpoison, net, device)
MyConfMat(y_test_clean.tolist(), preds_test_allpoison)
