# Import params and similarity from lib module
from params import *
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from datasets import *
from networks import *
from losses import *
from trainer import *
from os.path import dirname, abspath


# Uncomment to load triplet infomation which is used to collect ground truth
# Uncomment when you have new incoming dataset/data
# update_triplet_info(DATA_DIR, PN_THRESHOLD)

dataset_sizes = {}
# ---------------------------Loading training dataset---------------------------
print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading training dataset...')
train_dataset = TripletImagesDataset(DATA_DIR, IMAGE_SIZE, NEGATIVE_RAND_NUM, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
dataset_sizes.update({'train': len(train_dataset)})

# --------------------------Loading validation dataset--------------------------
print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading validation dataset...')
val_dataset = TripletImagesDataset(DATA_DIR, IMAGE_SIZE, NEGATIVE_RAND_NUM, is_train=False)
val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
dataset_sizes.update({'val': len(val_dataset)})

# ------------------------------Initialize model--------------------------------
print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
model = TripletNetImage()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Uncoment to see the summary of the model structure
# summary(model, [(3, IMAGE_SIZE, IMAGE_SIZE), (3, IMAGE_SIZE, IMAGE_SIZE)])

# ----------------------------Set Training Critera------------------------------
print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Set Training Critera...')
# Define loss function
loss_fcn = TripletLoss(constant_margin=False)
# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
# Decay LR by a factor of 0.1 every 7 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# --------------------------------Training--------------------------------------
print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Training with dataset size --> ', dataset_sizes)
data_loaders = {'train': train_loader, 'val': val_loader}
model_best_fit = training(device, data_loaders, dataset_sizes, model, loss_fcn, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS)
torch.save(model_best_fit.state_dict(), 'image_model_best_fit.pkl')

# ------------------------------------------------------------------------------
print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Done... Best Fit Model Saved')
print('----'*20)
