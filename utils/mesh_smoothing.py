import torch
from torch.nn import ConstantPad2d
import torch.nn.functional as F 
from os.path import join, basename
import numpy as np

from .networks import MeshSmoothNet
from .opt import Opt

class SmoothingModel:

    def __init__(self, filepath, device):

        f = torch.load(filepath, map_location=device)
        self.opt = Opt(f['opt'])
        self.state_dict = f['model_state_dict']
        self.mean = np.array(f['mean'], dtype=np.float32)
        self.std = np.array(f['std'], dtype=np.float32)
        self.device = torch.device(device)
        self.edge_features = None
        self.mesh = None
        self.net = MeshSmoothNet(self.opt, self.state_dict)
        #self.net = self.net.to(dtype=torch.float32)
        self.pool_res = self.net.res[1:]
        self.input_size = self.net.res[0]
        self.conv_filters = self.net.k[1:]
        self.resblocks = self.net.resblocks
        self.name = basename(filepath)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data["edge_features"]).float()
        self.edge_features = input_edge_features.to(self.device).requires_grad_(False)
        self.mesh = data["mesh"]
    
    def get_mesh_energy(self):
        if self.mesh is None:
            return None
        return np.array([m.mean_curvature_energy() for m in self.mesh])

    def forward(self):
        res = self.net((self.edge_features-self.mean)/self.std, [self.mesh])
        return res, self.mesh

    def set_device(self, device:str):
        self.net.to(device)