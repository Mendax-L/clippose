# Latest Update : 18 July 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

#BATCH_SIZE must larger than 1

import json
import numpy as np
import torch
import clippose
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import torchvision.transforms as transforms
from utils.loss import criterion_R
from utils.dataloader import Refine_Dataset
from clippose.refine import RotationAttention
import os


test_scene_ids = list(range(48,60))
train_scene_ids = list(set(range(20, 92)) - set(test_scene_ids))
# train_scene_ids = list(set(range(0, 2)))

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
print(f"using device:{device}")
# jit 可以通过 torch.jit.script 或 torch.jit.trace 来创建优化的模型版本，这些版本通过提前编译为低级代码。jit=False 是为了确保在训练期间使用动态计算图，因为 JIT 编译要求模型的计算图是静态的，而训练时通常需要修改计算图。
clip, preprocess = clippose.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
checkpoint = torch.load("checkpoint/model_20.pt")
clip.load_state_dict(checkpoint['model_state_dict'])
clip.cuda().eval()

BATCH_SIZE = 32
EPOCH = 10

obj_id = 1
# use your own data
refer_dir = f"/home/mendax/project/CLIPPose/refers/obj_{obj_id}"

train_dataset = Refine_Dataset(preprocess=preprocess, obj_id = obj_id,target_dirs = [f'../Datasets/ycbv/train_real/{str(id).zfill(6)}/view_000/' for id in train_scene_ids],data_fraction=0.1)
train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle=False) #Define your own dataloader
test_dataset = Refine_Dataset(preprocess=preprocess, obj_id = obj_id,target_dirs = [f'../Datasets/ycbv/test/{str(id).zfill(6)}/view_000/' for id in test_scene_ids],data_fraction=1)
test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE, shuffle=False) #Define your own dataloader



torch.cuda.empty_cache()

refine_model = RotationAttention().to(device)
optimizer = optim.Adam(refine_model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
for epoch in range(EPOCH):
    total_train_loss = 0
    train_correct = 0
    train_total = 0
    for batch in train_dataloader :
        optimizer.zero_grad()
        image, refer, base, R_i, R_r = batch 
        
        image, refer, base, R_i, R_r = image.to(device), refer.to(device), base.to(device), R_i.to(device), R_r.to(device)
        # print(R_i.shape)
        # print(R_r.shape)
        with torch.no_grad():
            image_features = clip.encode_image(image).float()

        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # refer_features /= refer_features.norm(dim=-1, keepdim=True)
        # base_features /= base_features.norm(dim=-1, keepdim=True)

        # Q = image_features @ base_features.T
        # K = refer_features @ base_features.T
   
        Q = image_features - base_features
        K = refer_features - base_features
        
        R_pred = refine_model(Q, K, R_r)

        total_loss = criterion_R(R_pred, R_i)

        total_loss.backward()

        total_train_loss += total_loss.item()

        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{EPOCH}], Train Loss: {avg_train_loss:.4f}")
        
    clip.eval()  # Set clip to evaluation mode
    with torch.no_grad():  # 不需要计算梯度
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        for batch in test_dataloader:
            image, refer, base, R_i, R_r = batch 
            
            image, refer, base, R_i, R_r = image.to(device), refer.to(device), base.to(device), R_i.to(device), R_r.to(device)

            with torch.no_grad():
                image_features = clip.encode_image(image).float()
                refer_features = clip.encode_image(refer).float()
                base_features = clip.encode_image(base).float()

            # image_features /= image_features.norm(dim=-1, keepdim=True)
            # refer_features /= refer_features.norm(dim=-1, keepdim=True)
            # base_features /= base_features.norm(dim=-1, keepdim=True)

            # Q = image_features @ base_features.T
            # K = refer_features @ base_features.T

            Q = image_features - base_features
            K = refer_features - base_features
            
            R_pred = refine_model(Q, K, R_r)

            total_loss = criterion_R(R_pred, R_i)

            total_val_loss += total_loss.item()


        # 打印验证损失和准确率
        avg_val_loss = total_val_loss / len(test_dataloader)
        print(f"Epoch [{epoch+1}/{EPOCH}], Validation Loss: {avg_val_loss:.4f}")


torch.save({
        'epoch': epoch,
        'model_state_dict': refine_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"checkpoint/refine_{EPOCH}.pt") #just change to your preferred folder/filename

