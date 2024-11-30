import torch
import torchvision
import glob
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, lowimg_path, maskimg_path, gtimg_path, gtimg_files, transform, mode):
        self.lowimg_path = lowimg_path
        self.maskimg_path = maskimg_path
        self.gtimg_path = gtimg_path
        self.gtimg_files = gtimg_files
        self.transform = transform
        self.mode = mode
    
    def __getitem__(self, index):
        lowimg = Image.open(self.lowimg_path + self.mode + "_" + self.gtimg_files[index])
        maskimg = Image.open(self.maskimg_path + "mask_" + self.gtimg_files[index])
        gtimg = Image.open(self.gtimg_path + self.gtimg_files[index])
        lowimg = self.transform(lowimg)
        maskimg = self.transform(maskimg)
        gtimg = self.transform(gtimg)
        return (lowimg, maskimg, gtimg)
    
    def __len__(self):
        return len(self.gtimg_files)