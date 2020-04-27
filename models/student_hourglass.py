from torch import nn
from models.student_layers import Residual, Hourglass


class PoseNet(nn.Module):
    def __init__(self, cfg):
        super(PoseNet, self).__init__()

        extra = cfg.MODEL.EXTRA

        self.num_feats = extra.NUM_FEATURES
        self.num_stacks = extra.NUM_STACKS
        self.num_hg_depth = extra.NUM_HG_DEPTH
        self.num_joints = cfg.MODEL.NUM_JOINTS

        self.pre = nn.Sequential(
            nn.Conv2d(3, int(self.num_feats / 4), kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(int(self.num_feats / 4)),
            nn.ReLU(),
            Residual(int(self.num_feats / 4), int(self.num_feats / 2)),
            nn.MaxPool2d(2, 2),
            Residual(int(self.num_feats / 2), int(self.num_feats / 2)),
            Residual(int(self.num_feats / 2), self.num_feats)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(self.num_hg_depth, self.num_feats)
            ) for i in range(self.num_stacks)
        ])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(self.num_feats, self.num_feats),
                nn.Conv2d(self.num_feats, self.num_feats, kernel_size=1, stride=1),
                nn.BatchNorm2d(self.num_feats),
                nn.ReLU()
            ) for i in range(self.num_stacks)
        ])

        self.outs = nn.ModuleList([
            nn.Conv2d(self.num_feats, self.num_joints,
                      kernel_size=1, stride=1) for i in range(self.num_stacks)
        ])

        self.merge_features = nn.ModuleList([
            nn.Conv2d(self.num_feats, self.num_feats,
                      kernel_size=1, stride=1) for i in range(self.num_stacks)
        ])

        self.merge_outs = nn.ModuleList([
            nn.Conv2d(self.num_joints, self.num_feats,
                      kernel_size=1, stride=1) for i in range(self.num_stacks)
        ])

    def forward(self, x):
        x = self.pre(x)
        combined_hm_preds = []
        hint = x

        for i in range(self.num_stacks):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            out = self.outs[i](feature)
            combined_hm_preds.append(out)
            x = x + self.merge_features[i](feature) + self.merge_outs[i](out)

        return hint, combined_hm_preds


def get_pose_net(cfg):
    return PoseNet(cfg)
