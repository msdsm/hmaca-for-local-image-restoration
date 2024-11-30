import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from hmaca import PartialHMANet
from dataset import Dataset
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
from torchvision.utils import save_image
import configparser

config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')

LOAD_PATH = config_ini['TEST']['LoadPath']
SAVE_PATH = config_ini['TEST']['SavePath']
IMG_SIZE = config_ini['TEST'].getint('ImgSize')
DATA_PATH = config_ini['TEST']['DataPath']
UPSCALE = config_ini['TEST'].getint('Upscale')
MODE = config_ini['TEST']['Mode']

os.makedirs(SAVE_PATH, exist_ok=True)

data_path = os.path.expanduser(DATA_PATH)
lowimg_path = data_path + "/horse_" + MODE + "/"
maskimg_path = data_path + "/horse_mask/"
gtimg_path = data_path + "/horse_gt/"
lowimg_files = sorted(os.listdir(lowimg_path))
maskimg_files = sorted(os.listdir(maskimg_path))
gtimg_files = sorted(os.listdir(gtimg_path))

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
])
test_dataset = Dataset(
    lowimg_path=lowimg_path,
    maskimg_path=maskimg_path,
    gtimg_path=gtimg_path,
    gtimg_files=gtimg_files,
    transform=transform,
    mode=MODE,
)
test_dataloader = DataLoader(test_dataset, batch_size=1)

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

parameters_load_path = LOAD_PATH
model.load_state_dict(torch.load(parameters_load_path))
model.eval()

cnt = 0
psnr_sum = 0.0
with tqdm(test_dataloader, total=len(test_dataloader)) as pbar:
    with torch.no_grad():
        for i, (lowimg, maskimg, gtimg) in enumerate(pbar):
            name = gtimg_files[i].split("/")[-1]
            lowimg = lowimg.to(device)
            gtimg = gtimg.to(device)
            maskimg = maskimg.to(device)
            output = model(lowimg, maskimg)
            output = output.squeeze(0)
            output_img = output.to("cpu").detach().numpy().copy().transpose(1, 2, 0).astype(np.float32)
            output_img = np.clip(output_img*255.0, a_min=0, a_max=255).astype(np.uint8)
            gtimg = gtimg.squeeze(0)
            gtimg = gtimg.to("cpu").detach().numpy().copy().transpose(1, 2, 0).astype(np.float32)
            gtimg = np.clip(gtimg*255.0, a_min=0, a_max=255).astype(np.uint8)
            cnt += 1
            maskimg = maskimg.squeeze(0)
            maskimg = maskimg.to("cpu").detach().numpy().copy().transpose(1, 2, 0).astype(np.float32)
            maskimg = np.clip(maskimg*255.0, a_min=0, a_max=255).astype(np.uint8)
            mask = maskimg.squeeze(axis=-1) # (h, w, 1) -> (h, w)
            output_img[mask == 0] = gtimg[mask == 0]
            psnr_sum += cv2.PSNR(output_img, gtimg)
            output_img = Image.fromarray(output_img)
            output_img.save(SAVE_PATH + name)

psnr = psnr_sum / cnt
print("config確認")
print("LOAD_PATH:", LOAD_PATH)
print("SAVE_PATH", SAVE_PATH)
print("IMG_SIZE:", IMG_SIZE)
print("DATA_PATH:", DATA_PATH)
print("UPSCALE:", UPSCALE)
print("PSNR:", psnr)
print("MODE:", MODE)