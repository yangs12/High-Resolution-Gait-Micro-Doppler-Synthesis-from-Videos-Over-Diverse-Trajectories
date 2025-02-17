import torch
from torch.utils.data import Dataset
import os
import numpy as np
import h5py
from PIL import Image

def read_h5_basic(path):
    """Read HDF5 files
    Args:
        path (string): a path of a HDF5 file
    Returns:
        radar_dat: micro-Doppler data with shape (1, time, micro-Doppler, 2 radar channel)
        des: descriptor information for radar data
    """
    hf = h5py.File(path, 'r')
    radar_dat = np.array(hf.get('radar_dat'))
    des = dict(hf.attrs)
    hf.close()
    return radar_dat, des

class RadarDataset(Dataset):
    def __init__(self,
                 file_list,
                 real_data_dir,
                 inference_data_dir,
                 transform=None,
                 target_transform=None,
                 label_type=None,
                 return_des=False,
                 is_sim=False,
                 is_coarse=False,
                 radar_idx = None,
                 ):
        self.file_list = file_list
        self.real_data_dir = real_data_dir
        self.inference_data_dir = inference_data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_type = label_type
        self.return_des = return_des
        self.is_sim = is_sim
        self.is_coarse = is_coarse
        self.radar_idx = radar_idx

    def __len__(self):
        return len(self.file_list)

    def read_h5_basic(self, path):
        hf = h5py.File(path, 'r')
        radar_dat = np.array(hf.get('radar_dat'))
        des = dict(hf.attrs)
        hf.close()
        return radar_dat, des

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.file_list[idx]
        real_data_path = os.path.join(self.real_data_dir, fname + '.h5')
        _, des = read_h5_basic(real_data_path) 
        des['radar_idx'] = self.radar_idx

        if self.is_sim:
            file_name = fname + "_" + str(self.radar_idx) + "_fake_B.png" 
            file_data_path = os.path.join(self.inference_data_dir, file_name)
            data = Image.open(file_data_path)
            data = data.convert('RGB')  

        elif self.is_coarse:
            file_name = fname +"_" + str(self.radar_idx) + "_real_A.png" 
            file_data_path = os.path.join(self.inference_data_dir, file_name)
            data = Image.open(file_data_path)
        else: 
            file_name = fname + "_" + str(self.radar_idx) +"_real_B.png" 
            file_data_path = os.path.join(self.inference_data_dir, file_name)
            data = Image.open(file_data_path)
            data = data.convert('RGB')
        
        if self.transform:
            data = self.transform(data)
        
        if self.target_transform:
            label = self.target_transform(des)
        
            if self.return_des:
                return data.type(torch.FloatTensor), label.type(torch.FloatTensor), des
            else:
                return data.type(torch.FloatTensor), label.type(torch.FloatTensor)
        else:
            return data.type(torch.FloatTensor), des
        