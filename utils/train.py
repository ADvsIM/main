import torch
import numpy as np
from torch import optim, nn
import torch.nn.functional as F
import dataset_load as dl

import sys
sys.path.append('/home/ppleeqq/IMvsAD')
from model.mlp import MLPClassifier


dataloader = DataLoader(dataset=IrisDataset('iris.data'),
                        batch_size=10,
                        shuffle=True)

epochs = 50
model = MLPClassifier()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    running_loss = 0
    for instances, labels in dataloader:
        optimizer.zero_grad()
        
        output = model(instances)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(running_loss / len(dataloader))

instances, labels = next(iter(dataloader))
instance = instances[0].view(1, 4)
label = labels[0].view(1, 1)
print(torch.exp(model(instance)), label)