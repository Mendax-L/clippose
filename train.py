# Latest Update : 18 July 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

#BATCH_SIZE must larger than 1

import numpy as np
import torch
import clippose
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import torchvision.transforms as transforms
from utils.loss import criterion_R
from utils.dataloader import Select_Refer_Dataset



# test_scene_ids = list(range(48,60))
# refer_scene_ids = list(range(0, 48))
# train_scene_ids = list(set(range(0, 92)) - set(test_scene_ids) - set(refer_scene_ids))

test_scene_ids = [0]
refer_scene_ids = [2,3]
train_scene_ids = [0,1,2]


device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
print(f"using device:{device}")
# jit 可以通过 torch.jit.script 或 torch.jit.trace 来创建优化的模型版本，这些版本通过提前编译为低级代码。jit=False 是为了确保在训练期间使用动态计算图，因为 JIT 编译要求模型的计算图是静态的，而训练时通常需要修改计算图。
model, preprocess = clippose.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
# checkpoint = torch.load("checkpoint/model_10.pt")
# model.load_state_dict(checkpoint['model_state_dict'])

BATCH_SIZE = 64
EPOCH = 20


# use your own data
obj_id = 3

train_dataset = Select_Refer_Dataset(preprocess=preprocess, obj_id = obj_id,\
                                     target_dirs = [f'../Datasets/space_station/train/antenna/train_pbr/{str(id).zfill(6)}/view_000' for id in train_scene_ids],\
                                     refer_dirs = [f'../Datasets/space_station/train/antenna/train_pbr/{str(id).zfill(6)}/view_000' for id in refer_scene_ids],\
                                      data_fraction=1)
print(len(train_dataset))
train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle=True) #Define your own dataloader
test_dataset = Select_Refer_Dataset(preprocess=preprocess, obj_id = obj_id,\
                                    target_dirs = [f'../Datasets/space_station/test/antenna/train_pbr/{str(id).zfill(6)}/view_000' for id in test_scene_ids],\
                                    refer_dirs = [f'../Datasets/space_station/train/antenna/train_pbr/{str(id).zfill(6)}/view_000' for id in train_scene_ids],\
                                      data_fraction=1)
test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE, shuffle=True) #Define your own dataloader


#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
else :
  clippose.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_rot = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
torch.cuda.empty_cache()

# add your own code to track the training progress.

def calculate_rotation_difference(R_img, R_refer):
    """
    计算两个旋转矩阵之间的角度差距
    这里使用的是旋转矩阵的 Frobenius 距离或通过旋转角度来计算差距。
    """
    # 计算旋转矩阵之间的误差
    diff = torch.matmul(R_img, R_refer.transpose(0, 1))  # R_img 和 R_refer 转置相乘
    # 计算差距的 Frobenius 范数，表示旋转误差的大小
    loss = torch.norm(diff - torch.eye(3, device=R_img.device), p='fro')
    return loss

def get_ground_truth(R_img, R_refer):
    """
    对每个 R_img，找到与其差距最小的 R_refer 对应的编号
    """
    ground_truth = []
    for img_rot in R_img:
        # 计算当前 R_img 和所有 R_refer 之间的差距
        losses = [calculate_rotation_difference(img_rot, refer_rot) for refer_rot in R_refer]
        # 找到最小差距的 refer 编号
        best_match_idx = torch.argmin(torch.tensor(losses, device=img_rot.device))
        ground_truth.append(best_match_idx)
    return torch.tensor(ground_truth, device=R_img.device)

for epoch in range(EPOCH):
  model.train()  # Set model to training mode
  total_train_loss = 0
  train_correct = 0
  train_total = 0
  for batch in train_dataloader :
      optimizer.zero_grad()

      images, refer ,R_img , R_refer= batch 

      images, refer ,R_img , R_refer= images.to(device), refer.to(device),R_img.to(device), R_refer.to(device)

      rgb_sample = images[0]
      # 再将其映射到 [0, 255]
      transform = transforms.ToPILImage()
      img = transform(rgb_sample)
      img.save(f"visib/train_image.png")
      rgb_sample = refer[0]
      # 再将其映射到 [0, 255]
      transform = transforms.ToPILImage()
      img = transform(rgb_sample)
      img.save(f"visib/train_refer.png")
      # print(rots)
    

      logits_per_image, logits_per_rot = model(images, refer)


      ground_truth = get_ground_truth(R_img, R_refer)
      total_loss = loss_img(logits_per_image,ground_truth)

      total_loss.backward()

      total_train_loss += total_loss.item()
      _, predicted = torch.max(logits_per_image, 1)
      train_total += ground_truth.size(0)
      train_correct += (predicted == ground_truth).sum().item()

      if device == "cpu":
         optimizer.step()
      else : 
        convert_models_to_fp32(model)
        optimizer.step()
        clippose.model.convert_weights(model)

  avg_train_loss = total_train_loss / len(train_dataloader)
  accuarcy = train_correct / train_total
  print(f"Epoch [{epoch+1}/{EPOCH}], Train Loss: {avg_train_loss:.4f}, Accuracy:{accuarcy* 100:.2f}%")
    
  model.eval()  # Set model to evaluation mode
  with torch.no_grad():  # 不需要计算梯度
      total_val_loss = 0
      val_correct = 0
      val_total = 0

      for batch in test_dataloader:
          images, refer ,R_img , R_refer= batch 
          images, refer ,R_img , R_refer= images.to(device), refer.to(device),R_img.to(device), R_refer.to(device)

          rgb_sample = images[0]
          # 再将其映射到 [0, 255]
          transform = transforms.ToPILImage()
          img = transform(rgb_sample)
          img.save(f"visib/test_image.png")
          rgb_sample = refer[0]
          # 再将其映射到 [0, 255]
          transform = transforms.ToPILImage()
          img = transform(rgb_sample)
          img.save(f"visib/test_refer.png")

          images = images.to(device)
          refer = refer.to(device)

          # 计算模型输出
          logits_per_image, logits_per_rot = model(images, refer)


          ground_truth = get_ground_truth(R_img, R_refer)
          total_loss = loss_img(logits_per_image,ground_truth)

          total_val_loss += total_loss.item()

          # 计算准确度（假设是分类任务）
          _, predicted = torch.max(logits_per_image, 1)
          val_total += ground_truth.size(0)
          val_correct += (predicted == ground_truth).sum().item()

      # 打印验证损失和准确率
      avg_val_loss = total_val_loss / len(test_dataloader)
      accuarcy = val_correct / val_total
      print(f"Epoch [{epoch+1}/{EPOCH}], Validation Loss: {avg_val_loss:.4f}, Accuracy:{accuarcy* 100:.2f}%")


torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"checkpoint/antenna_model_{EPOCH}.pt") #just change to your preferred folder/filename


# model, preprocess = clippose.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
# checkpoint = torch.load("model_checkpoint/model_10.pt")

# # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
# checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
# checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

# model.load_state_dict(checkpoint['model_state_dict'])