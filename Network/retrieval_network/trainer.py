import torch
import time
import os
import copy, math
import numpy as np
from termcolor import colored
from progress.bar import Bar
from Network.retrieval_network.datasets import get_pose_from_name
from Network.retrieval_network.params import *


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

# ------------------------------------------------------------------------------
def Training(device, data_loaders, dataset_sizes, model, loss_fcn, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS, checkpoints_prefix=None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    """
    start_time = time.time() # Traning starting time_elapsed

    # Intialize storage for best weights/model and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1.0

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
            # loading bar
            print('----'*6)
            bar = Bar('Processing', max=math.ceil(dataset_sizes[phase]/BATCH_SIZE))
            for batch_idx, (inputs, alphas) in enumerate(data_loaders[phase]):
                # zero the parameter gradients
                optimizer.zero_grad()
                # Case scene graph brach: for triplet data (A,N,P), each of A,N,P have 3 matrices for scene graphs
                inputs = tuple(input.to(device) for input in inputs) # to GPU
                alphas = tuple(alpha.to(device) for alpha in alphas) # to GPU

                # Forward propagation
                # Track history if only in trainer
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(*inputs)
                    loss, correct_num = loss_fcn(*outputs, alphas, device, batch_average_loss=True)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # --------------------------------------------------------------
                # Get/calculate training statistics
                running_loss += loss.item()
                running_corrects += correct_num
                bar.next()
            bar.finish()
            # ------------------------------------------------------------------
            if phase == 'train':
                lr_scheduler.step() # update LEARNING_RATE

            # Epoch loss calculation
            epoch_loss = running_loss * BATCH_SIZE / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: \t {:.4f} \t Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model: based on minimum loss
            if phase == 'val':
                if checkpoints_prefix != None:
                    FILE = checkpoints_prefix + '_loss_' + str(epoch_loss) + '_acc_' + str(epoch_acc.item()) + '_epoch_' + str(epoch) + '.pkl'
                    torch.save(model.state_dict(), FILE)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            # ------------------------------------------------------------------
    # --------------------------------------------------------------------------
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
