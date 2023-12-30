# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:54:18 2023

@author: Spencer perkins

Main Script : intended to be exectuted through Bash script

This script is basically one epoch, Use run.sh scripting to 
execute full training

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
import time
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import argparse
import json
from datetime import datetime

import sys

# Utilities
sys.path.insert(1, '../../modules/utils/')
from get_data_kfold import getKfoldDATA
from birdhouse_dataset import birdhouseDataset

# Bash args
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, help='fold index')
parser.add_argument('--epoch', type=int, help='epoch index')
parser.add_argument('--cnn', type=str, help='cnn model name')
parser.add_argument('--frontend', type=str, help='frontend name')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--experiment', type=str, help='experiment number')

args = parser.parse_args()

experiment =args.experiment
net = args.cnn
frontend = args.frontend
print(f'\n{args.frontend}\n')

# Load appropriate CNN, initial definition
sys.path.insert(1, '../../modules/nets/')
if net == 'birdhouse_l3_multi':
    from birdhouse_l3_multi import birdHouseL3M
    cnn = birdHouseL3M()
elif net == 'birdhouse_l3MR':
    from birdhouse_l3_multi_res import birdHouseL3MR
    cnn = birdHouseL3MR()
else:
    from birdhouse_l3 import birdHouseL3S
    cnn = birdHouseL3S()

# Load appropriate Frontend, initial definition
sys.path.insert(1, '../../modules/frontends/')
if frontend == 'pcen':
    from pcen_opt import PCEN
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
    from tmrpcen10_opt import TMRPCEN10
    front = TMRPCEN10(n_bands=128,
                    alpha=0.8, delta=10.,
                    r=0.25, eps=10e-6)
    save_path = 'tmrpcen10_all/'
elif frontend == 'tmrpcen10_opt_dog':
    from tmrpcen10_opt_dog import TMRPCEN10dog
    front = TMRPCEN10dog(n_bands=128,
                         alpha=0.8, delta=10.,
                         r=0.25, eps=10e-6)
    save_path = 'tmrpcen10dog_all/'
elif frontend == 'tmrpcen':
    from tmrpcen import TMRPCEN
    front = TMRPCEN(n_bands=128,
                    alpha=0.8, delta=10.,
                    r=0.25, eps=10e-6)
    save_path = 'tmrpcen_all/'


#%% Preparation and definitions
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE USED: {device}')
print(f'\n{args.epoch}\n')

# Parameters
train_batch = 20
val_batch = 20
lr =  args.lr
print(f'LEARNING RATE: {lr}\n\n\n')
epochs = 30

# Define folds
val_fold = str(args.fold) + '.csv'
_folds = np.arange(1, 11).tolist()
folds = [str(x) + '.csv' for x in _folds]

# Get Data
audio_path, tr_meta, vl_meta = getKfoldDATA.get_folded_data(val_fold=val_fold, folds=folds)
# Load data
features = 'melspec'
print(f'Feature Extraction: {features}')
train_data = birdhouseDataset(label_file=tr_meta, audio_dir=audio_path,
                              features=features
                              )
val_data = birdhouseDataset(label_file=vl_meta, audio_dir=audio_path,
                            features=features
                            )

# Data loaders
trainLoader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=train_batch,
                                          shuffle=True, pin_memory=True
                                          )
valLoader = torch.utils.data.DataLoader(dataset=val_data,
                                        batch_size=val_batch,
                                        shuffle=True,pin_memory=True
                                        )

# Initialize frontend and CNN models
model_front, best_model_front = front, front
model_cnn, best_model_cnn = cnn, cnn
print('Number of frontend params: {}'.format(
    sum([p.data.nelement() for p in model_front.parameters()])))
print('\nNumber of CNN params: {}\n-------------------------\n'.format(
    sum([p.data.nelement() for p in model_cnn.parameters()])))

# Send to GPU if available
model_front.to(device)
model_cnn.to(device)

# Optimizer
params = list(model_front.parameters())+list(model_cnn.parameters())
opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)

# Define Loss function
loss_fun = nn.BCELoss()

hist_file_path = '../../results/'+experiment+'/hist/'+frontend+'/'+ str(args.fold) +'_record.json'

# Dictionary to store training history
if args.epoch == 0:
    # First epoch initialization for results
    hist = {'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'no_bird_acc': [],
            'has_bird_acc': [],
            'F1': [],
            'time':[]
            }
    best_acc = - 1.0
    
else:

    # epoch > 0 Load results
    with open(hist_file_path, 'r') as hist_file:
        hist = json.load(hist_file)

    # Load previous model
    state_dict = torch.load('../../results/'+experiment+'/weights/'+frontend+'/prev_frontend_wts_'+ str(args.fold) +'.pt')
    model_front.load_state_dict(state_dict['state_dict'])
    state_dict = torch.load('../../results/'+experiment+'/weights/'+frontend+'/prev_cnn_wts_'+ str(args.fold) +'.pt')
    model_cnn.load_state_dict(state_dict['state_dict'])
    epoch_dict = torch.load('../../results/'+experiment+'/epoch_acc/'+frontend+'/epoch_acc_'+ str(args.fold) +'.pt')
    best_epoch = epoch_dict['best_epoch']
    best_acc = epoch_dict['best_acc']

#%% Training Loop
print('[INFO] Training the network...')
startTime = time.time()

# Calculate steps per epoch for training and validation set
trainSteps = len(trainLoader.dataset) // train_batch
valSteps = len(valLoader.dataset) // val_batch
countT = trainSteps
countV = valSteps
print(f'Training steps: {countT}')
print(f'Validation steps: {countV}\n-----------------------\n')

# Loop over epochs
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

# Loop over the training set
for (x, y) in trainLoader:
    # Send the input to the device
    (x, y) = (x.to(device), y.to(device))
    # Perform a forward pass and calculate the training loss
    norm = model_front(x)
    pred = model_cnn(norm)
    print(pred)
    pred = pred.to(torch.float32)
    y = y.to(torch.float32)
    loss = loss_fun(pred, y)
    # Zero out the gradients, perform the backpropagation step,
    # and update the weights
    opt.zero_grad()
    loss.backward()
    opt.step()
    # Add the loss to the total training loss so far and
    # calculate the number of correct predictions
    totalTrainLoss += loss
    trainCorrect += (pred.round() == y).type(
        torch.float).sum().item()

    if countT % 10 == 0:
        logger = datetime.now()
        logger = logger.strftime("%m_%d_%Y_%H_%M_%S")
        print(f'\n-----------\n {trainSteps-countT}th/{trainSteps} iteration complete')
        print(f'Device Used : {device}')
        print(f'\n{logger}')
    countT -= 1
    iter_count += 1

# Switch off autograd for evaluation
with torch.no_grad():
    val_cm = {'pred': torch.zeros(0,dtype=torch.float32, device='cpu'),
              'truth': torch.zeros(0,dtype=torch.float32, device='cpu')}
    # Set the model in evaluation mode
    model_front.eval()
    model_cnn.eval()
    # Loop over the validation set
    for countV, (x, y) in enumerate(valLoader):
        # Send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # Make the predictions and calculate the validation loss
        norm = model_front(x)
        pred = model_cnn(norm)
        pred = pred.to(torch.float32)
        y = y.to(torch.float32)
        totalValLoss += loss_fun(pred, y)
        # Calculate the number of correct predictions
        valCorrect += (pred.round() == y).type(
            torch.float).sum().item()
        val_cm['pred']=torch.cat([val_cm['pred'],pred.round().view(-1).cpu()])
        val_cm['truth']=torch.cat([val_cm['truth'],y.view(-1).cpu()])
        if countV % 5 == 0:
            logger = datetime.now()
            logger = logger.strftime("%m_%d_%Y_%H_%M_%S")
            print(f'\n-----------\n {valSteps-countV}th/{valSteps} iteration complete')
            print(f'Device Used : {device}')
            print(f'\n{logger}')
    epoch_acc = valCorrect / len(valLoader.dataset)

    # Save the best models
    if epoch_acc > best_acc:
        torch.save({
            'state_dict': model_front.state_dict(),
        }, '../../results/'+experiment+'/weights/'+frontend+'/best_frontend_wts_'+ str(args.fold) +'.pt')
        torch.save({
            'state_dict': model_cnn.state_dict()
        }, '../../results/'+experiment+'/weights/'+frontend+'/best_cnn_wts_'+ str(args.fold) +'.pt')
        torch.save({
            'epoch': args.epoch,
            'best_epoch': args.epoch,
            'best_acc': epoch_acc
            }, '../../results/'+experiment+'/epoch_acc/'+frontend+'/epoch_acc_'+str(args.fold)+'.pt')
    else:
        torch.save({
            'epoch':args.epoch,
            'best_epoch': best_epoch,
            'best_acc': best_acc
            }, '../../results/'+experiment+'/epoch_acc/'+frontend+'/epoch_acc_'+str(args.fold)+'.pt')

    # Save the current models which will be used for NEXT epoch
    torch.save({
        'state_dict': model_front.state_dict(),
    }, '../../results/'+experiment+'/weights/'+frontend+'/prev_frontend_wts_'+ str(args.fold) +'.pt')
    torch.save({
        'state_dict': model_cnn.state_dict(),
    }, '../../results/'+experiment+'/weights/'+frontend+'/prev_cnn_wts_'+ str(args.fold) +'.pt')

countT = trainSteps
countV = valSteps
# Calculate average training and validation loss
avgTrainLoss = totalTrainLoss / trainSteps
avgValLoss = totalValLoss / valSteps
# Calculate the training and validation accuracy
trainCorrect = trainCorrect / len(trainLoader.dataset)
valCorrect = valCorrect / len(valLoader.dataset)

# Confusion matrix
conf_mat=confusion_matrix(val_cm['pred'].numpy(), val_cm['truth'].numpy())
print('\n-----Confusion Matrix-----\n')
print(conf_mat)
# Calculate F1
f1 = f1_score(val_cm['pred'].numpy(), val_cm['truth'].numpy(), average='binary')

# Print model training and validation information
print('\n[INFO] EPOCH: {}/{}'.format(args.epoch + 1, epochs))
print('Train loss: {:.6f}, Train accuracy: {:.4f}'.format(
    avgTrainLoss, trainCorrect))
print('Val loss: {:.6f}, Val accuracy: {:.4f}\n'.format(
    avgValLoss, valCorrect))
print('F1 Score: {:.6f}'.format(f1))

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print('No bird Accuracy: %.2f'%class_accuracy[0])
print('Has bird Accuracy: %.2f'%class_accuracy[1])

hist['train_loss'].append(avgTrainLoss.item())
hist['train_acc'].append(trainCorrect)
hist['val_loss'].append(avgValLoss.item())
hist['val_acc'].append(valCorrect)
hist['no_bird_acc'].append(class_accuracy[0])
hist['has_bird_acc'].append(class_accuracy[1])
hist['F1'].append(f1)

epoch_end = time.time()
print('Epoch duration: %.2f sec.\n\n-----------------------------------------------' % (epoch_end - epoch_start))

# Finish measuring training time
endTime = time.time()
hist['time'].append(epoch_end-epoch_start)
print('[INFO] total time taken to train the model: {:.2f}s'.format(
    endTime - startTime))

# Store epoch results
with open(hist_file_path, 'w') as hist_file:
    json.dump(hist, hist_file)

if args.epoch == epochs - 1:

    # Dataframe to csv of epoch training/validation data
    results_df = pd.DataFrame.from_dict(hist, orient='columns')
    results_df.to_csv('../../results/'+experiment+'/'+save_path+frontend+'_'+val_fold)

    print('--------------DONE----------------')
