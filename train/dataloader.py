import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import torch
import torch.utils.data as data
import numpy as np
import sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]

def _load_data_file(name):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:]
    return data, label
    
class ModelNet40(data.Dataset):

    def __init__(
            self,  data_dir, num_points=2500, split='train'
    ):
        super().__init__()

        self.data_dir = data_dir

        self.split, self.num_points = split, num_points
        if self.split == 'train':
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))
        else:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))
        root = '../data'
        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(root, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        
        current_points = self.points[idx, pt_idxs]
        current_points = torch.from_numpy(current_points).to(torch.float32)
        #current_points = torch_center_and_normalize
        label = self.labels[idx]

        
        return label,None,current_points

    def __len__(self):
        return self.points.shape[0]

def load_data(data_path,corruption,severity):

    DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
    # if corruption in ['occlusion']:
    #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
    LABEL_DIR = os.path.join(data_path, 'label.npy')
    all_data = np.load(DATA_DIR)
    all_label = np.load(LABEL_DIR)
    return all_data, all_label

class ModelNet40C(Dataset):
    def __init__(self, split,data_dir,corruption,severity):
        self.split = "test"
        self.corruption = corruption
        self.severity = severity
        self.data_dir = data_dir
        self.data, self.label = load_data(self.data_dir, self.corruption, self.severity)
        # self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item]#[:self.num_points]
        label = self.label[item]
        return label.item(), None, torch.from_numpy(pointcloud).to(torch.float32)

    def __len__(self):
        return self.data.shape[0]
