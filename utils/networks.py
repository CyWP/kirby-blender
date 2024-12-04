import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F

from .layers.mesh_conv import MeshConv
from .layers.mesh_pool import MeshPool


def get_norm_layer(norm_type="instance", num_groups=1):
    print(f"Num groups: {num_groups}")
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == "none":
        norm_layer = NoNorm
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, "__name__") and norm_layer.__name__ == "NoNorm":
        norm_args = [{"fake": True} for f in nfeats_list]
    elif norm_layer.func.__name__ == "GroupNorm":
        norm_args = [{"num_channels": f} for f in nfeats_list]
    elif norm_layer.func.__name__ == "BatchNorm":
        norm_args = [{"num_features": f} for f in nfeats_list]
    else:
        raise NotImplementedError(
            "normalization layer [%s] is not found" % norm_layer.func.__name__
        )
    print(f"Norm args: {norm_args}")
    return norm_args


class NoNorm(nn.Module):

    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()

    def forward(self, x):
        return x

    def __call__(self, x):
        return self.forward(x)


class MeshSmoothNet(nn.Module):
    """Network learning to smooth meshes (unsupervised learning)"""

    def __init__(self, opt: 'Opt', state_dict):

        super(MeshSmoothNet, self).__init__()

        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata

        self.k = [opt.input_nc] + opt.ncf
        self.res = [opt.ninput_edges] + opt.pool_res
        self.resblocks = opt.resblocks
        norm_layer = get_norm_layer(opt.norm, opt.num_groups)
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, "conv{}".format(i), MResConv(ki, self.k[i + 1], self.resblocks))
            setattr(self, "norm{}".format(i), norm_layer(**norm_args[i]))
            setattr(self, "pool{}".format(i), MeshPool(self.res[i + 1]))

        self.load_state_dict(state_dict)
        self.eval()

    def forward(self, x, mesh):
        for i in range(len(self.k) - 1):
            x = getattr(self, "conv{}".format(i))(x, mesh)
            x = F.relu(getattr(self, "norm{}".format(i))(x))
            x = getattr(self, "pool{}".format(i))(x, mesh)
        return x

    def __call_(self, x):
        return self.forward(x, mesh)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)
        for i in range(self.skips):
            setattr(self, "bn{}".format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(
                self,
                "conv{}".format(i + 1),
                MeshConv(self.out_channels, self.out_channels, bias=False),
            )

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, "bn{}".format(i + 1))(F.relu(x))
            x = getattr(self, "conv{}".format(i + 1))(x, mesh)
            x = F.relu(x)
        #x += x1
        #x = F.relu(x)
        return x

