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
  hidden = 200
  in_features = 28 * 28
  out_features = 10
  model = nn.Sequential(
    nn.Linear(in_features, hidden),
    nn.ReLU(),
    nn.Linear(hidden, hidden),
    nn.ReLU(),
    nn.Linear(hidden, hidden),
    nn.ReLU(),
    nn.Linear(hidden, out_features),
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
    avgs = 0
    for imgs, labels in loader:

      imgs = imgs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      pred = model(imgs)
      loss = criterion(pred, labels)
      loss.backward()
      optimizer.step()

      avgs += (pred.argmax(dim=1) == labels).sum()

    print(f'epoch: {epoch}, acc: {avgs.item() / len(train_dataset):.2f}')

  # save the model to the gs://python-trained-models bucket
  filename = f'{datetime.now().strftime("%Y-%m-%d-%s")}.pt'
  bucket = storage.Client().bucket('python-trained-models')
  blob = bucket.blob(filename)
  torch.save(model.state_dict(), 'model.pt')
  blob.upload_from_filename('model.pt')
