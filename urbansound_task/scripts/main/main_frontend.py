# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:08:48 2023

@author: spencer perkins

Main Script : Trainable frontend approach

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
import time
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import copy
import argparse
from datetime import datetime

import sys

# Utilities
sys.path.insert(1, '../../modules/utils/')
from get_data_kfold import getKfoldDATA
from urbanhouse_dataset import urbanhouseDataset

# Bash scripts args
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, help='fold index')
parser.add_argument('--cnn', type=str, help='cnn model name')
parser.add_argument('--frontend', type=str, help='frontend name')
parser.add_argument('--experiment', type=str, help='exv + num')
args = parser.parse_args()

exper = args.experiment
net = args.cnn
frontend = args.frontend
val_fold = str(args.fold)

# Load appropriate CNN, initial definition
sys.path.insert(1, '../../modules/nets/')
if net == 'urbanhouse_l3_multi':
    from urbanhouse_l3_multi import urbanHouseL3M
    cnn = urbanHouseL3M()
else:
    from urbanhouse_l3 import urbanHouseL3S
    cnn = urbanHouseL3S()

# Load appropriate Frontend, initial definition
sys.path.insert(1, '../../modules/frontends/')
if frontend == 'pcen':
    from pcen import PCEN
    #from pcen_opt import PCEN
    front = PCEN(n_bands=128, t_val=2**5, alpha=0.8,
                 delta=10., r=0.25,
                 eps=10e-6)
    save_path = 'pcen_all/'
elif frontend == 'mrpcen':
    from mrpcen import MRPCEN
    t_vals=[2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9]
    front = MRPCEN(n_bands=128, t_values=t_vals,
                            alpha=0.8, delta=10.,
                            r=0.25, eps=10e-6)
    save_path = 'mrpcen_all/'
elif frontend == 'tmrpcen10':
    from tmrpcen10 import TMRPCEN10
    front = TMRPCEN10(n_bands=128,
                    alpha=0.8, delta=10.,
                    r=0.25, eps=10e-6)
    save_path = 'tmrpcen10_all/'
elif frontend == 'tmrpcen':
    from tmrpcen import TMRPCEN
    front = TMRPCEN(n_bands=128,
                    alpha=0.8, delta=10.,
                    r=0.25, eps=10e-6)
    save_path = 'tmrpcen_all/'

#%% MAIN SCRIPT

# Prep and Definitions
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE USED: {device}')

print(f'\n\nFrontend: {frontend}\n\n')

# Classes for Confusion Matrix
classes = ['air conditioner', 'car horn', 'children playing',
           'dog bark', 'drilling', 'engine idling',
           'gun shot', 'jackhammer', 'siren', 'street music']

# Parameters
train_batch = 20
val_batch = 20
lr =  0.0001
epochs = 30

folds = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'
        ]

# Get Data
audio_path, tr_meta, vl_meta = getKfoldDATA.get_folded_data(val_fold=val_fold,
                                                            folds=folds)
# Load data
features = 'melspec'
print(f'Feature Extraction: {features}')
train_data = urbanhouseDataset(label_file=tr_meta, audio_dir=audio_path,
                              features=features
                              )
val_data = urbanhouseDataset(label_file=vl_meta, audio_dir=audio_path,
                            features=features
                            )

# Data loaders
trainLoader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=train_batch,
                                          shuffle=True, pin_memory=True
                                          )
valLoader = torch.utils.data.DataLoader(dataset=val_data,
                                        batch_size=val_batch,
                                        shuffle=True, pin_memory=True
                                        )

# Initialize frontend and CNN models
model_front = front
model_cnn = cnn
print('Number of frontend params: {}'.format(
    sum([p.data.nelement() for p in model_front.parameters()])))
print('\nNumber of CNN params: {}\n-------------------------\n'.format(
    sum([p.data.nelement() for p in model_cnn.parameters()])))

# Send to GPU
model_front.to(device)
model_cnn.to(device)

# Model weights data
best_front_wts = copy.deepcopy(model_front.state_dict())
best_cnn_wts = copy.deepcopy(model_cnn.state_dict())
best_acc = -1.0

# Optimizer
params = list(model_front.parameters())+list(model_cnn.parameters())
opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
lr_sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

# Define Loss function
loss_fun = nn.CrossEntropyLoss()

# Dictionary to store history
hist = {
        'train_loss': np.zeros(epochs, dtype=float),
        'train_acc': np.zeros(epochs, dtype=float),
        'val_loss': np.zeros(epochs, dtype=float),
        'val_acc': np.zeros(epochs, dtype=float),
        'F1_macro': np.zeros(epochs, dtype=float)
        }


print('[INFO] Training the network...')
startTime = time.time()

# Calculate steps per epoch for training and validation set
trainSteps = len(trainLoader.dataset) // train_batch
valSteps = len(valLoader.dataset) // val_batch
train_steps = trainSteps
val_steps= valSteps
print(f'Training steps: {train_steps}')
print(f'Validation steps: {val_steps}\n-----------------------\n')

# EPOCH LOOP
for e in range(0, epochs):

    # Set the model in training mode
    model_front.train()
    model_cnn.train()
    # Initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # Initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0

    epoch_start = time.time()
    iter_count = 0

    # TRAINING LOOP
    for countT, (x,y) in enumerate(trainLoader):
        # Zero out gradients
        opt.zero_grad()
        # Send data to device
        (x, y) = (x.to(device), y.to(device))
        # Forward pass
        norm = model_front(x)
        output = model_cnn(norm)
        # Calculate Loss
        loss = loss_fun(output, y)
        # Backpropagation step, update weights
        loss.backward()
        opt.step()
        # Update training loss
        totalTrainLoss += loss.item()
        # Calculate Training accuracy
        _, pred = torch.max(output.data, 1)
        trainCorrect += (pred.cpu() == y.cpu()).sum()
        
        # Delete unneeded data
        del x, y, norm, output, pred

        # CLI vis
        if countT % 10 == 0:
            logger = datetime.now()
            logger = logger.strftime("%m_%d_%Y_%H_%M_%S")
            print(f'\n-----------\n {trainSteps-countT}th/{trainSteps} iteration complete')
            print(f'Device Used : {device}')
            print(f'\n{logger}')

    # VALIDATION LOOP
    # Switch off autograd for evaluation
    with torch.no_grad():
        true_labels = []
        predicted_labels = []
        # Set the model in evaluation mode
        model_front.eval()
        model_cnn.eval()

        # VALIDATION LOOP
        for countV, (x,y) in enumerate(valLoader):
            # Send data to device
            (x, y) = (x.to(device), y.to(device))
            # Forward pass
            norm = model_front(x)
            output = model_cnn(norm)
            # Calculate loss
            loss = loss_fun(output, y)
            # Update validation loss
            totalValLoss += loss.item()
            # Calculate validation accuracy
            _, pred = torch.max(output, 1)
            valCorrect += (pred.cpu() == y.cpu()).sum()
            # Truth, predicitons for F1 and Confusion matrix
            true_labels.extend(y.cpu().numpy())
            predicted_labels.extend(pred.cpu().numpy())

            # Delete unneeded data
            del x, y, pred, norm

            # CLI vis
            if countV % 5 == 0:
                logger = datetime.now()
                logger = logger.strftime("%m_%d_%Y_%H_%M_%S")
                print(f'\n-----------\n {valSteps-countV}th/{valSteps} iteration complete')
                print(f'Device Used : {device}')
                print(f'\n{logger}')

    epoch_acc = valCorrect / len(valLoader.dataset)
    if epoch_acc > best_acc:
        best_front_wts = copy.deepcopy(model_front.state_dict())
        best_cnn_wts = copy.deepcopy(model_cnn.state_dict())
        best_acc = epoch_acc
        confusion = confusion_matrix(true_labels, predicted_labels)
        cf_df = pd.DataFrame(confusion, index=classes, columns=classes)
        cf_df.to_csv('../../results/'+exper+'/confusion_mat/'+frontend+'/cf_'+val_fold+'.csv')


    # Update learning rate
    lr_sched.step()
    print(f'\nLEARNING RATE: {lr_sched.get_last_lr()}\n')
    # Calculate average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # Calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainLoader.dataset)
    valCorrect = valCorrect / len(valLoader.dataset)

    # Update training history
    hist['train_loss'][e] += avgTrainLoss
    hist['train_acc'][e] += trainCorrect
    hist['val_loss'][e] += avgValLoss
    hist['val_acc'][e] += valCorrect

    # Confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)
    print('\n-----Confusion Matrix-----\n')
    print(confusion)

    # Calculate F1 Scores
    macf1 = f1_score(
        predicted_labels, true_labels, average='macro' )
    hist['F1_macro'][e] += macf1
    # Print model training and validation information
    print('\n[INFO] EPOCH: {}/{}'.format(e + 1, epochs))
    print('Train loss: {:.6f}, Train accuracy: {:.4f}'.format(
        avgTrainLoss, trainCorrect))
    print('Val loss: {:.6f}, Val accuracy: {:.4f}\n'.format(
        avgValLoss, valCorrect))
    print('F1 (macro) Score: {:.6f}'.format(macf1))
    epoch_end = time.time()
    print('Epoch duration: %.2f sec.\n\n-----------------------------------------------' % (epoch_end - epoch_start))

# Dataframe to csv of epoch training/validation data
results_df = pd.DataFrame.from_dict(hist, orient='columns')
results_df.to_csv(
'../../results/'+exper+'/'+save_path+frontend+'_'+val_fold+'.csv')
torch.save(
best_front_wts, '../../results/'+exper+'/weights/'+frontend+'/best_frontend_wts_'+val_fold+'.pt')
torch.save(
best_cnn_wts, '../../results/'+exper+'/weights/'+frontend+'/best_cnn_wts_'+val_fold+'.pt')

print('--------------DONE----------------')
