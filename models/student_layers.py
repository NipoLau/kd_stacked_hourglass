from torch import nn


class Residual(nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super(Residual, self).__init__()

        self.separable_conv = cfg.MODEL.EXTRA.SEPARABLE_CONV
        self.depthwise_conv = cfg.MODEL.EXTRA.DEPTHWISE_CONV
        self.stride_conv = cfg.MODEL.EXTRA.STRIDE_CONV

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, int(out_dim / 2), kernel_size=1)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))

        # self.conv2 = nn.Conv2d(int(out_dim / 2), int(out_dim / 2), kernel_size=3, padding=1)
        if self.depthwise_conv and not self.separable_conv:
            self.conv2 = nn.Conv2d(int(out_dim / 2), int(out_dim / 2), kernel_size=3, padding=1, groups=int(out_dim / 2))
        if self.separable_conv:
            self.conv2_1 = nn.Conv2d(int(out_dim / 2), int(out_dim / 2), kernel_size=3, padding=1, groups=int(out_dim / 2))
            self.bn2_2 = nn.BatchNorm2d(int(out_dim / 2))
            self.conv2_2 = nn.Conv2d(int(out_dim / 2), int(out_dim / 2), kernel_size=1)

        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = nn.Conv2d(int(out_dim / 2), out_dim, kernel_size=1)
        self.skip_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else None

    def forward(self, x):
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))

        if self.separable_conv:
            x = self.conv2_1(x)
            x = self.relu(self.bn2_2(x))
            x = self.conv2_2(x)
        else:
            x = self.conv2(x)

        x = self.relu(self.bn3(x))
        x = self.conv3(x)

        x += residual

        return x


class Hourglass(nn.Module):
    def __init__(self, cfg, depth, planes):
        super(Hourglass, self).__init__()

        self.stride_conv = cfg.MODEL.EXTRA.STRIDE_CONV
        self.transpose_conv = cfg.MODEL.EXTRA.TRANSPOSE_CONV
        self.group_conv = cfg.MODEL.EXTRA.GROUP_CONV

        self.up = Residual(cfg, planes, planes)
        self.down_sample = nn.Conv2d(planes, planes, kernel_size=4, stride=2, padding=1, groups=planes if self.group_conv else 1) \
            if self.stride_conv else nn.MaxPool2d(2, 2)
        self.low1 = Residual(cfg, planes, planes)
        self.low2 = Hourglass(cfg, depth - 1, planes) if depth > 1 else Residual(cfg, planes, planes)
        self.low3 = Residual(cfg, planes, planes)
        self.up_sample = nn.ConvTranspose2d(planes, planes, kernel_size=4, stride=2, padding=1, groups=planes if self.group_conv else 1) \
            if self.transpose_conv else nn.Upsample(scale_factor=2)

    def forward(self, x):
        residual = self.up(x)

        x = self.down_sample(x)
        x = self.low1(x)
        x = self.low2(x)
        x = self.low3(x)
        x = self.up_sample(x)

        x += residual

        return x
