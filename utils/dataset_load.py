import pandas as pd
import numpy as np
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

#__________________________________________________________

#           Imbalanced Classification
#               0. satellite (31.64%)
#               1. cardio (9.61%)
#               2. Pageblocks (9.46%)
#               3. annthyroid (7.42%)
#               4. shuttle (7.15%)
#               5. thyroid (2.47%)
#               6. satimage-2 (1.22%)
#               7. Creditcard - binary class (0.17%)
#__________________________________________________________

def minor_class_number(data, label, number):

    # import pdb
    # pdb.set_trace()
    
    indices_x1 = np.where(label == 1)[0]
    indices_x0 = np.where(label == 0)[0]

    selected_x0 = data[indices_x0]

    selected_indices = np.random.choice(indices_x1, number, replace=False)
    selected_x1 = data[selected_indices]

    balanced_data = np.concatenate([selected_x0, selected_x1], axis=0)
    balanced_label = np.concatenate([label[indices_x0], label[selected_indices]], axis=0)

    return balanced_data, balanced_label


class CustomDataset(Dataset):
    def __init__(self, dataset, file_path, number,  seed):
        
        real_data = np.load(file_path + dataset + '.npz')

        self.data = real_data['X']
        self.labels = real_data['y']
        self.seed = seed

        train_data, val_data, train_labels, val_labels = train_test_split(self.data, self.labels, test_size=0.2, stratify=self.labels,random_state=self.seed)
        val_data, test_data, val_labels, test_labels = train_test_split(val_data, val_labels, test_size=0.5, stratify=val_labels,random_state=self.seed)
        
        if number == 'none':
            balanced_train_data = train_data
            balanced_train_labels = train_labels
        else:
            balanced_train_data, balanced_train_labels = minor_class_number(train_data, train_labels, number)

        self.scaler = MinMaxScaler()
        balanced_train_data = self.scaler.fit_transform(balanced_train_data)
        val_data = self.scaler.transform(val_data)
        test_data = self.scaler.transform(test_data)

        self.train_data = balanced_train_data
        self.val_data = val_data
        self.test_data = test_data

        self.train_y= torch.FloatTensor(balanced_train_labels)
        self.val_y = torch.FloatTensor(val_labels)
        self.test_y = torch.FloatTensor(test_labels)

        self.train_dataset = TensorDataset(torch.FloatTensor(balanced_train_data),torch.FloatTensor(balanced_train_labels))
        self.val_dataset = TensorDataset(torch.FloatTensor(val_data),torch.FloatTensor(val_labels))
        self.test_dataset = TensorDataset(torch.FloatTensor(test_data),torch.FloatTensor(test_labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

