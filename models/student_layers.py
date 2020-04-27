from torch import nn


class Residual(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, int(out_dim / 2), kernel_size=1)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = nn.Conv2d(int(out_dim / 2), int(out_dim / 2), kernel_size=3, padding=1)
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
        x = self.conv2(x)
        x = self.relu(self.bn3(x))
        x = self.conv3(x)

        x += residual

        return x


class Hourglass(nn.Module):
    def __init__(self, depth, planes):
        super(Hourglass, self).__init__()
        self.up = Residual(planes, planes)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = Residual(planes, planes)
        self.low2 = Hourglass(depth - 1, planes) if depth > 1 else Residual(planes, planes)
        self.low3 = Residual(planes, planes)
        self.up_sample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        residual = self.up(x)

        x = self.pool(x)
        x = self.low1(x)
        x = self.low2(x)
        x = self.low3(x)
        x = self.up_sample(x)

        x += residual

        return x