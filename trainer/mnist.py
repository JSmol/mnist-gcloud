import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datetime import datetime
from google.cloud import storage

def run(data_path, seed, batch_size, epochs, learning_rate, weight_decay, beta1, beta2):

  # set device (it should read 'cuda' when running in GCP!!)
  if torch.cuda.is_available(): device = f'cuda:{torch.cuda.current_device()}'
  else: device = 'cpu'

  print(f'Current device: {device}')
  print()

  # set random seed
  torch.manual_seed(seed)

  # model defn
  model = nn.Sequential(
    nn.Conv2d(1, 32, 4, 2, 1, bias=False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(),
    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 10),
    nn.ReLU(),
  )

  # our data set is quite simple, no need to use torch.utils.data.Dataset
  train_dataset = pd.read_csv(f'{data_path}/train.csv')
  train_labels = torch.tensor(train_dataset.pop('label').to_numpy(), dtype=torch.int64)
  train_imgs = torch.tensor(train_dataset.to_numpy() / 255, dtype=torch.float32)
  loader = DataLoader(list(zip(train_imgs, train_labels)), batch_size=batch_size, shuffle=True)

  # train
  model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
  for epoch in range(epochs):
    total = 0
    for imgs, labels in loader:

      imgs = imgs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      pred = model(imgs.reshape(-1, 1, 28, 28))
      loss = criterion(pred, labels)
      loss.backward()
      optimizer.step()

      total += (pred.argmax(dim=1) == labels).sum()

    print(f'epoch: {epoch}, acc: {total.item() / len(train_dataset):.2f}')

  # save the model to the gs://python-trained-models bucket
  filename = f'{datetime.now().strftime("%Y-%m-%d-%s")}.pt'
  bucket = storage.Client().bucket('python-trained-models')
  blob = bucket.blob(filename)
  torch.save(model.state_dict(), 'model.pt')
  blob.upload_from_filename('model.pt')

  # eval set (save the result to storage)
  eval_dataset = pd.read_csv(f'{data_path}/test.csv')
  eval_imgs = torch.tensor(eval_dataset.to_numpy() / 255, dtype=torch.float32)
  eval_loader = DataLoader(eval_imgs, batch_size=batch_size, shuffle=False)

  model.to('cpu')
  with torch.no_grad():
    result = torch.tensor([], dtype=torch.int64)
    for imgs in eval_loader:
      pred = model(imgs.reshape(-1, 1, 28, 28)).argmax(dim=1)
      result = torch.cat((result, pred), 0)

  result = pd.DataFrame(result, index=range(1, len(eval_dataset)+1), columns=['Label'], dtype=int)
  result.to_csv('result.csv', index_label='ImageId')

  blob = bucket.blob('result.csv')
  blob.upload_from_filename('result.csv')
