import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from hmaca import PartialHMANet
from dataset import Dataset
from tqdm import tqdm
import configparser

config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')

CHECKPOINT_PATH = config_ini['TRAIN']['CheckpointPath']
IMG_SIZE = config_ini['TRAIN'].getint('ImgSize')
EPOCHS = config_ini['TRAIN'].getint('Epochs')
BATCH_SIZE = config_ini['TRAIN'].getint('BatchSize')
LEARNING_RATE = float(config_ini['TRAIN']['LearningRate'])
DATA_PATH = config_ini['TRAIN']['DataPath']
UPSCALE = config_ini['TRAIN'].getint('Upscale')
ALPHA = float(config_ini['TRAIN']['Alpha'])
MODE = config_ini['TRAIN']['Mode']

alpha = ALPHA
os.makedirs("./checkpoint", exist_ok=True)
data_path = os.path.expanduser(DATA_PATH)
lowimg_path = data_path + "/horse_" + MODE + "/"
maskimg_path = data_path + "/horse_mask/"
gtimg_path = data_path + "/horse_gt/"
lowimg_files = os.listdir(lowimg_path)
gtimg_files = os.listdir(gtimg_path)
epochs = EPOCHS
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
])
train_dataset = Dataset(
    lowimg_path=lowimg_path,
    maskimg_path=maskimg_path,
    gtimg_path=gtimg_path,
    gtimg_files=gtimg_files,
    transform=transform,
    mode=MODE,
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
 
model = PartialHMANet(
    img_size=IMG_SIZE,
    upscale=UPSCALE,
    upsampler='pixelshuffle',
    window_size=32
)
model.to(device)
criterion = nn.L1Loss()
train_losses = []
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    print("epoch {}:".format(str(epoch+1)))
    model.to(device)
    model.train()
    with tqdm(train_dataloader, total=len(train_dataloader)) as pbar:
        for i, (lowimgs, maskimgs, gtimgs) in enumerate(pbar):
            lowimgs = lowimgs.to(device)
            maskimgs = maskimgs.to(device)
            gtimgs = gtimgs.to(device)
            optimizer.zero_grad()
            output = model(lowimgs, maskimgs)
            batch_size = maskimgs.size(0)
            losses = []
            for b in range(batch_size):
                mask = maskimgs[b, 0]
                white_pixels = (mask == 1).nonzero(as_tuple=True)
                h1, h2 = white_pixels[0].min().item(), white_pixels[0].max().item() + 1
                w1, w2 = white_pixels[1].min().item(), white_pixels[1].max().item() + 1
                output_crop = output[b, :, h1:h2, w1:w2]
                gtimgs_crop = gtimgs[b, :, h1:h2, w1:w2]
                loss = alpha * criterion(output_crop, gtimgs_crop) + (1 - alpha) * criterion(output, gtimgs)
                losses.append(loss)
            
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()
            train_losses.append(batch_loss.detach().item())

    print("mean_train_loss:{}".format(sum(train_losses)/len(train_losses)))
    if (epoch+1)%10 == 0:
        model.to("cpu")
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("save model")

# config
print("config確認")
print("PROJECT:", PROJECT)
print("CHECKPOINT_PATH:", CHECKPOINT_PATH)
print("IMG_SIZE:", IMG_SIZE)
print("EPOCHS:", EPOCHS)
print("BATCH_SIZE:", BATCH_SIZE)
print("LEARNING_RATE:", LEARNING_RATE)
print("DATA_PATH:", DATA_PATH)
print("UPSCALE:", UPSCALE)
print("MODE:", MODE)
