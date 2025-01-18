import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import os
from lib.depth_estiamtion import Depth_Net
from utils.dataloader import Depth_Dataset
from torch.utils.data import Dataset, DataLoader
from utils.loss import L1_loss

test_scene_ids = [0]
train_scene_ids = [0,1,2,3]
obj_id = 1
BATCH_SIZE = 16
train_dataset = Depth_Dataset(obj_id = obj_id,target_dirs = [f'../Datasets/space_station/train/socket/train_pbr/{str(id).zfill(6)}' for id in train_scene_ids],data_fraction=1)
print(len(train_dataset))
train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle=True) #Define your own dataloader
test_dataset = Depth_Dataset(obj_id = obj_id,target_dirs = [f'../Datasets/space_station/test/socket/train_pbr/{str(id).zfill(6)}' for id in test_scene_ids],data_fraction=1)
test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE, shuffle=True) #Define your own dataloader


device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
print(f"using device:{device}")
model = Depth_Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
criterion = L1_loss

# Training loop
def train_model(model, train_loader,radius, num_epochs=1, save_path="model.pth"):
    model.train()
    if len(train_loader.dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data source.")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for R, depth_cu, depth_gt in tqdm(train_loader):
            R, depth_cu, depth_gt = R.to(device),depth_cu.to(device), depth_gt.to(device)

            optimizer.zero_grad()
            output = model(R)
            depth_pred = output*radius + depth_cu
            # print(f"R:{R.shape}")
            # print(f"depth_cu:{depth_cu}")
            # print(f"depth_gt:{depth_gt}")
            # print(f"output:{output}")
            # print(f"depth_pred:{depth_pred}")
            loss = criterion(depth_pred, depth_gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * R.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print("Model saved to", save_path)

# valing loop
def val_model(model, val_loader, radius):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for R, depth_cu, depth_gt in tqdm(val_loader):
            R, depth_cu, depth_gt = R.to(device),depth_cu.to(device), depth_gt.to(device)
            depth_pred = model(R)*radius + depth_cu
            loss = criterion(depth_pred, depth_gt)
            val_loss += loss.item() * R.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"val Loss: {val_loss:.4f}")

# Example usage of training and valing
if __name__ == "__main__":
    # Paths to images and their corresponding targets
    radius = 300
    # Train and val the model
    train_model(model, train_dataloader, radius, num_epochs=25, save_path=f"checkpoint/depth_{obj_id}.pth")
    val_model(model, test_dataloader, radius)
