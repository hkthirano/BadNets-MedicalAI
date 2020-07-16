# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt

from utils import MyConfMat, MyGray2TorchRGB, inference, poison, set_seed, MyResize

seed = 123
set_seed(seed)


# %%
x_test = np.load('chestx/x_test.npy')
y_test = np.load('chestx/y_test.npy')
x_test = MyGray2TorchRGB(MyResize(x_test, imsize=(224, 224))) / 255.0


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torchvision.models.vgg16(pretrained=True)
num_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(num_features, 2)
net.load_state_dict(torch.load('model/poisoned_nontarget_pytorch_fine.pth'))


class FirstModel(nn.Module):
    def __init__(self, original_model):
        super(FirstModel, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[0][:30])

    def forward(self, x):
        x = self.features(x)
        return x


class SecondModel(nn.Module):
    def __init__(self, original_model):
        super(SecondModel, self).__init__()
        self.features1 = nn.Sequential(list(original_model.children())[0][30])
        self.features2 = nn.Sequential(list(original_model.children())[1])
        self.features3 = nn.Sequential(list(original_model.children())[2])

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = torch.flatten(x, 1)
        x = self.features3(x)
        return x


first_model = FirstModel(net).to(device)
second_model = SecondModel(net).to(device)


# %%
# test clean
x_test_clean, y_test_clean = x_test.copy(), y_test.copy()
y_test_clean = np.argmax(y_test_clean, 1)
testset_clean = torch.utils.data.TensorDataset(
    torch.tensor(x_test_clean),
    torch.tensor(y_test_clean))
testloader_clean = torch.utils.data.DataLoader(testset_clean, batch_size=8,
                                               shuffle=False, num_workers=2)

# test all poison
x_test_allpoison, y_dummy = x_test.copy(), y_test.copy()
x_test_allpoison, y_dummy = poison(x_test_allpoison,
                                   y_dummy,
                                   poison_rate=1.0, target=-1, d=2)
testset_allpoison = torch.utils.data.TensorDataset(
    torch.tensor(x_test_allpoison),
    torch.tensor(y_test_clean))
testset_allpoison = torch.utils.data.DataLoader(testset_allpoison, batch_size=8,
                                                shuffle=False, num_workers=2)


# %%
act_diff = np.load('output/act_diff.npy')
act_diff_poison = np.where(act_diff < 0, 0, act_diff)
poison_l2 = np.linalg.norm(np.reshape(
    act_diff_poison, (512, -1)), ord=2, axis=1)
n = 5
topn_poison_l2_idx = np.argsort(poison_l2)[::-1][:n].tolist()

k_list = [0.01, 0.1, 0.5, 1, 3, 5, 7, 9, 10, 20, 30, 50]
clean_acc_list = []
poison_acc_list = []
for k in k_list:
    with torch.no_grad():
        correct = 0
        for data in testloader_clean:
            images, labels = data[0].to(device), data[1]
            outputs = first_model(images)

            outputs[:, topn_poison_l2_idx, :,
                    :] = outputs[:, topn_poison_l2_idx, :, :] * k

            outputs = second_model(outputs)
            _, predicted = torch.max(outputs.data.cpu(), 1)
            correct += (predicted == labels).sum().item()
        clean_acc_list.append(float(correct / len(y_test_clean)))

        correct = 0
        for data in testset_allpoison:
            images, labels = data[0].to(device), data[1]
            outputs = first_model(images)

            outputs[:, topn_poison_l2_idx, :,
                    :] = outputs[:, topn_poison_l2_idx, :, :] * k

            outputs = second_model(outputs)
            _, predicted = torch.max(outputs.data.cpu(), 1)
            correct += (predicted == labels).sum().item()
        poison_acc_list.append(float(correct / len(y_test_clean)))


# %%
plt.figure()
plt.plot(k_list, clean_acc_list, label='clean')
plt.plot(k_list, poison_acc_list, label='backdoor')
plt.scatter(x=1, y=clean_acc_list[3])
plt.scatter(x=1, y=poison_acc_list[3])
plt.title('k_list = [0.01, 0.1, 0.5, 1, 3, 5, 7, 9, 10, 20, 30, 50]')
plt.xlabel('k')
plt.ylabel('acc')
plt.legend()
plt.savefig('output/k_vs_acc_to50.png')

plt.figure()
plt.plot(k_list[:6], clean_acc_list[:6], label='clean')
plt.plot(k_list[:6], poison_acc_list[:6], label='backdoor')
plt.scatter(x=1, y=clean_acc_list[3])
plt.scatter(x=1, y=poison_acc_list[3])
plt.title('k_list = [0.01, 0.1, 0.5, 1, 3, 5]')
plt.xlabel('k')
plt.ylabel('acc')
plt.legend()
plt.savefig('output/k_vs_acc_to5.png')
