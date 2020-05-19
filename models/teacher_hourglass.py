import torch
from torch import nn
from models.teacher_layers import Conv, Hourglass, Pool, Residual
import numpy as np


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim):
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList([
        nn.Sequential(
            Hourglass(4, inp_dim),
        ) for i in range(nstack)])
        
        self.features = nn.ModuleList([
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)])
        
        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack-1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(nstack-1)])

    def forward(self, imgs):
        # 某些 tensor 操作依赖于数据在内存中是整块的，调用 contiguous 函数将非连续存储的数据变为一整块
        # x = imgs.permute(0, 3, 1, 2).contiguous() #x of size 1,3,inpdim,inpdim
        # 对输入图像进行处理，提取高维特征，通道数扩展到 inp_dim - 256
        x = self.pre(imgs)
        hint = [x]

        combined_hm_preds = []
        for i in range(self.nstack):
            # 经过四阶 Hourglass 网络，通道数保持不变 - 256，尺寸保持不变 - 64
            hg = self.hgs[i](x)
            # 经过 Residual + Conv 网络，尺寸不变，提取特征
            feature = self.features[i](hg)
            # 获取输出热图，通道数为 oup_dim - 16
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i == self.nstack / 2 - 1:
                hint.append(feature)
            if i < self.nstack - 1:
                # 特征图和输出热图各自通过一层 Conv 通道扩展到 inp_dim - 256，与 short_cut 相加，输入到下一个 Hourglass 模块
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        return hint, combined_hm_preds


def get_pose_net(cfg):
    extra = cfg.MODEL.EXTRA
    return PoseNet(extra.NUM_STACKS, extra.NUM_FEATURES, cfg.MODEL.NUM_JOINTS)
