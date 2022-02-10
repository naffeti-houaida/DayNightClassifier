# importing required libraries
import torch
# nn module provides a much more convenient and powerful method for defining network architectures.
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
import train_val_scripts
from pathlib import Path
from torchvision import transforms, models, datasets
# Define a transform to normalize the data
# Simple data augmentation for model_training set (randomly flip images horizontally and verticl)
train_tfms = transforms.Compose([
              transforms.Resize((500,500)),
              transforms.RandomHorizontalFlip(p=0.5),
              transforms.RandomVerticalFlip(p=0.5),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# No data augmentation for validation set, only normalizing data
valid_tfms = transforms.Compose([
              transforms.Resize((500,500)),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# path to image
data_dir = '/content/drive/MyDrive/Day-nightClassification/day_night_dataset/'
# application of transformation on data
train_ds = datasets.ImageFolder(data_dir+'train', transform=train_tfms)
valid_ds = datasets.ImageFolder(data_dir+'val', transform=valid_tfms)
# load model_training and validation data
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=10, shuffle=False, num_workers=2, pin_memory=True)
x, y = next(iter(valid_dl))
# loading imagenet pretrained mobilenet v2 from torchvision best_model
mbv2 = models.mobilenet_v2(pretrained=True)
# number of feature for the model
in_features = mbv2.classifier[1].in_features
# replace final FC layer of model with FC layer with number of output classes = 2 for day/night
mbv2.classifier[1] = nn.Linear(in_features, 2)
# moving model to GPU for model_training
mbv2 = mbv2.to(device='cuda:0')
# Using Softmax CrossEntropy Loss
criterion = nn.CrossEntropyLoss()

# Adam optimizer with lr=1e-4
opt = torch.optim.Adam(mbv2.parameters(), lr=1e-4)

# Cosine Annealing Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, len(train_dl)*15, eta_min=1e-6)
max_acc = 0.0 # Track maximum validation accuracy achieved

for epoch in range(5):
  best = False # Flag to detect best model

# Training phase
  train_loss, train_acc = train_val_scripts.train_epoch(mbv2, train_dl, criterion, opt, scheduler)

# Validation phase
  valid_loss, valid_acc = train_val_scripts.valid_epoch(mbv2, valid_dl, criterion)


  if valid_acc > max_acc: # Saving best model
    max_acc = valid_acc
    torch.save(mbv2.state_dict(), 'mbv2_best_model.pth')
    best = True

  print('-'*25 + f'Epoch {epoch+1}' + '-'*25)
  print(f'Train Loss:{train_loss} Train Accuracy:{train_acc}')
  print(f'Valid Loss:{valid_loss} Valid Accuracy:{valid_acc}')
  if best:
    print(f'Found better model!')
  print('-'*58)