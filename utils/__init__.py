import numpy as np
import torch
import bpy
from os.path import abspath

from .mesh_smoothing import SmoothingModel
from .np2blend import np2mesh, mesh2np, vg2np
from .layers.mesh import Mesh

class ModelManager:
    _model = None
    _mesh1 = None
    _mesh2 = None
    _vg1 = None
    _vg2 = None

    @classmethod
    def load_model(cls, filepath:str, device:str):
        cls._model = SmoothingModel(abspath(filepath), device)

    @classmethod
    def release_model(cls):
        cls._model = None #Garbage collector moment

    @classmethod
    def get_pool_resolutions(cls):
        if not cls._model:
            return None
        return cls._model.pool_res

    @classmethod
    def get_input_size(cls):
        if not cls._model:
            return None
        return cls._model.input_size

    @classmethod
    def get_min_size(cls):
        if not cls._model:
            return None
        return cls._model.pool_res[-1]

    @classmethod
    def get_model_info(cls):
        info = []
        if cls._model:
            info.append(cls._model.name)
            info.append(f"Input size: {cls._model.input_size}")
            convstr = " ,".join([str(f) for f in cls._model.conv_filters])
            info.append(f"Conv filters: {convstr}")
            info.append(f"Convs per pooling: {cls._model.resblocks}")
            poolstr = ", ".join([str(res) for res in cls._model.pool_res])
            info.append(f"Pool resolutions: {poolstr}")
            nparams = cls._model.net.num_params()/1000
            info.append(f"Parameters: {nparams:.2f}k")
        return info

    @classmethod
    def is_loaded(cls):
        return cls._model is not None

    @classmethod
    def mesh_loaded(cls, id:int):
        if id==1:
            return not cls._mesh1 is None
        if id==2:
            return not cls._mesh2 is None

    @classmethod
    def set_mesh1(cls, mesh:bpy.types.Object):
        cls._vg1 = None
        if mesh:
            cls._mesh1 = cls.load_mesh(mesh)

    @classmethod
    def set_mesh2(cls, mesh:bpy.types.Object):
        cls._vg2 = None
        if mesh:
            cls._mesh2 = cls.load_mesh(mesh)

    @classmethod
    def load_mesh(cls, mesh:bpy.types.Object):
        data = {}
        V, E, F = mesh2np(mesh)
        meshdata = Mesh(V, E, F)
        if cls.is_loaded() and E.shape[0]>cls._model.input_size:
            raise Exception(f"Mesh has too many edges ({E.shape[0]}). Max is {cls._input_size}.")
        data['mesh'] = meshdata
        data['edge_features'] = pad(meshdata.extract_features(), cls._model.input_size)[np.newaxis, :]
        data['name'] = mesh.name
        return data

    @classmethod
    def set_vg1(cls, mesh:bpy.types.Object, name:str):
        self._vg1 = vg2np(mesh, name)

    @classmethod
    def set_vg1(cls, mesh:bpy.types.Object, name:str):
        self._vg2 = vg2np(mesh, name)

    @classmethod
    def set_device(cls, device:str):
        cls._model.set_device(device)

    @classmethod
    def process1(cls, context):
        net = cls._model
        net.set_input(cls._mesh1.copy())
        features, newmesh = net.forward()
        name = f"{cls._mesh1['name']}_processed"
        return np2mesh(context, newmesh.vs, newmesh.faces, name)

    @classmethod
    def process2(cls, context):
        net = cls._model
        net.set_input(cls._mesh2.copy())
        features, newmesh = net.forward()
        name = f"{cls._mesh2['name']}_processed"
        return np2mesh(context, newmesh.vs, newmesh.faces, name)

    @classmethod
    def transfer(cls, mesh1:bpy.types.Object, mesh2:bpy.types.Object, vg1:bpy.types.VertexGroups, vg2:bpy.types.VertexGroups, task:str):
        pass

    @classmethod
    def get_available_devices(cls):

        devices = []
        devices.append("cpu")
        if torch.cuda.is_available():
            cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            devices.extend(cuda_devices)
        if torch.backends.mps.is_available():
            devices.append("mps")
        return devices


def pad(input_arr:np.array, target_length:int, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    try:
        padded = np.pad(input_arr, pad_width=npad, mode="constant", constant_values=val)
    except:
        raise Exception(f"Padding error, shape={shp}, target_length={target_length}, npad={npad}")
    return padded