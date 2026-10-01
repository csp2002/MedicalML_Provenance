"""UNet++ (Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", DLMIA 2018).

2D implementation adapted from the CBIM-Medical-Image-Segmentation code base
(https://github.com/yhygao/CBIM-Medical-Image-Segmentation): five resolution
levels with nested dense skip pathways, two 3x3 conv-BN-ReLU layers per node,
trained from scratch.
"""
import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base_ch=32, num_block=2):
        super().__init__()
        n_ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = self.make_layer(in_ch, n_ch[0], num_block)
        self.conv1_0 = self.make_layer(n_ch[0], n_ch[1], num_block)
        self.conv2_0 = self.make_layer(n_ch[1], n_ch[2], num_block)
        self.conv3_0 = self.make_layer(n_ch[2], n_ch[3], num_block)
        self.conv4_0 = self.make_layer(n_ch[3], n_ch[4], num_block)

        self.conv0_1 = self.make_layer(n_ch[0] + n_ch[1], n_ch[0], num_block)
        self.conv1_1 = self.make_layer(n_ch[1] + n_ch[2], n_ch[1], num_block)
        self.conv2_1 = self.make_layer(n_ch[2] + n_ch[3], n_ch[2], num_block)
        self.conv3_1 = self.make_layer(n_ch[3] + n_ch[4], n_ch[3], num_block)

        self.conv0_2 = self.make_layer(n_ch[0] * 2 + n_ch[1], n_ch[0], num_block)
        self.conv1_2 = self.make_layer(n_ch[1] * 2 + n_ch[2], n_ch[1], num_block)
        self.conv2_2 = self.make_layer(n_ch[2] * 2 + n_ch[3], n_ch[2], num_block)

        self.conv0_3 = self.make_layer(n_ch[0] * 3 + n_ch[1], n_ch[0], num_block)
        self.conv1_3 = self.make_layer(n_ch[1] * 3 + n_ch[2], n_ch[1], num_block)

        self.conv0_4 = self.make_layer(n_ch[0] * 4 + n_ch[1], n_ch[0], num_block)

        self.output = nn.Conv2d(n_ch[0], num_classes, kernel_size=1)

    @staticmethod
    def make_layer(in_ch, out_ch, num_block):
        blocks = [SingleConv(in_ch, out_ch)] + [SingleConv(out_ch, out_ch) for _ in range(num_block - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        return self.output(x0_4)
