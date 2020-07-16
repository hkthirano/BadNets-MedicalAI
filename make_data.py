# %%
import csv

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# %%
tsv_file = '../../COVID-Net/train_split.txt'
x = []
y = []
with open(tsv_file, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    next(reader)
    for row in reader:
        if row[2] == 'normal':
            y.append([1.0, 0.0])
        elif row[2] == 'pneumonia':
            y.append([0.0, 1.0])
        else:
            continue

        img_path = '../../COVID-Net/data/train/' + row[1]
        im = Image.open(img_path).convert('L')
        im = im.resize((224, 224))
        im = np.array(im, dtype='float32')
        x.append(im)


# %%
tsv_file = '../../COVID-Net/test_split.txt'
with open(tsv_file, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    next(reader)
    for row in reader:
        if row[2] == 'normal':
            y.append([1.0, 0.0])
        elif row[2] == 'pneumonia':
            y.append([0.0, 1.0])
        else:
            continue

        img_path = '../../COVID-Net/data/test/' + row[1]
        im = Image.open(img_path).convert('L')
        im = im.resize((224, 224))
        im = np.array(im, dtype='float32')
        x.append(im)


# %%
x = np.array(x)
y = np.array(y)


# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=100)


# %%
np.save('covid/x_train', x_train)
np.save('covid/x_test', x_test)
np.save('covid/y_train', y_train)
np.save('covid/y_test', y_test)


# %%
