import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

def action2word(action):
    word = ""

    # forward and backward
    word += "straight: "
    if action[0] == 0:
        word += "noop\n"
    if action[0] == 1:
        word += "forward\n"
    if action[0] == 2:
        word += "backward\n"

    # left and right
    word += "pan: "
    if action[1] == 0:
        word += "noop\n"
    if action[1] == 1:
        word += "left\n"
    if action[1] == 2:
        word += "right\n"

    # jump
    word += "jump: "
    if action[2] == 0:
        word += "noop\n"
    if action[2] == 1:
        word += "jump\n"
    return word

def voxel2word(voxel):
    '''
    the word is in the type of | | |  |\n with the type of string
    # 左下角为原点, 从下到上，从左到右
    '''
    word = ""

    # Get the block_name array from voxel observation
    if isinstance(voxel, dict) and "block_name" in voxel:
        block_names = voxel["block_name"]
    else:
        block_names = voxel

    # Get the shape of the voxel array
    shape = block_names.shape

    # Iterate through the 3D voxel array
    for y in range(shape[1]):
        for x in range(shape[0]):
            for z in range(shape[2]):
                block = block_names[x, y, z]
                word += f"|{block}"
            word += "|\n"

    return word

class MineDojoDataset(Dataset):
    def __init__(self, data_dir="creative:0", transform=None):
        """
        PyTorch Dataset for MineDojo data
        
        Args:
            data_dir: Directory containing .npy files
            transform: Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all .npy files in the directory
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        
        if not self.file_paths:
            raise ValueError(f"No .npy files found in {data_dir}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load data from .npy file
        data = np.load(self.file_paths[idx], allow_pickle=True).item()
        
        # Extract action and voxel data
        action = data["action"]
        voxel = data["voxel"]
        
        # Create sample dictionary
        sample = {
            "action": action,
            "voxel": voxel,
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

if __name__ == "__main__":
    dataset = MineDojoDataset()
    for sample in dataset:
        print(sample)