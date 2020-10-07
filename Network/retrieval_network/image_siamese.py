# Import params and similarity from lib module
from params import *
from torch.utils.data import DataLoader
from torchsummary import summary
from datasets import *
from networks import *

# Uncomment to load triplet infomation which is used to collect ground truth
# Uncomment when you have new incoming dataset/data
# update_triplet_info(DATA_DIR, PN_THRESHOLD)

print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading training dataset...')
train_dataset = TripletImagesDataset(DATA_DIR, IMAGE_SIZE, NEGATIVE_RAND_NUM, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading validation dataset...')
val_dataset = TripletImagesDataset(DATA_DIR, IMAGE_SIZE, NEGATIVE_RAND_NUM, is_train=False)
val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
model = SiameseNetImage()

# Summary of the model structure
summary(model, [(3, IMAGE_SIZE, IMAGE_SIZE), (3, IMAGE_SIZE, IMAGE_SIZE)])
