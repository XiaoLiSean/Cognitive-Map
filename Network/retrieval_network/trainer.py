import torch
import time
import os
import copy
import numpy as np
from termcolor import colored
from datasets import get_pose_from_name
from params import *

root_folder = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_folder)
from lib.similarity import view_similarity

# ------------------------------------------------------------------------------
# This function is triplet loss with variable margin alpha defined by view cone overlaps
def triplet_loss_variable_margin(anchor, positive, negative, img_names):

    anchor_pose = get_pose_from_name(img_names[0][0])
    positive_pose = get_pose_from_name(img_names[1][0])
    negative_pose = get_pose_from_name(img_names[2][0])

    alpha = view_similarity(anchor_pose, positive_pose) - view_similarity(anchor_pose, negative_pose) # margin

    loss = torch.max(torch.FloatTensor([COS(anchor, negative)[0] - COS(anchor, positive)[0] + alpha, 0]))

    return loss

# ------------------------------------------------------------------------------
def training(device, data_loaders, dataset_sizes, model, loss_fcn, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    """
    start_time = time.time() # Traning starting time_elapsed

    # Intialize storage for best weights/model and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # --------------------------------------------------------------------------
    # Traning start
    # --------------------------------------------------------------------------
    for epoch in range(num_epochs):
        print('----'*10 + '\n' + colored('Traning Info: ','blue') + 'Epoch {}/{}'.format(epoch + 1, num_epochs))

        # ----------------------------------------------------------------------
        # Each epoch has a training and validation phase
        # ----------------------------------------------------------------------
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # ----------------------------Train---------------------------------
            # Iteration over train/validation dataset
            # ------------------------------------------------------------------
            for batch_idx, (inputs, img_names) in enumerate(data_loaders[phase]):
                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward propagation
                # Track history if only in trainer
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(*inputs)
                    loss, correct_num = loss_fcn(*outputs, img_names)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # --------------------------------------------------------------
                # Get/calculate training statistics
                running_loss += loss.item()
                running_corrects += correct_num
            # ------------------------------------------------------------------
            if phase == 'train':
                lr_scheduler.step() # update LEARNING_RATE

            # Epoch loss calculation
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('----'*6)
            print('{} Loss: \t {:.4f} \t Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # ------------------------------------------------------------------
    # --------------------------------------------------------------------------
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
