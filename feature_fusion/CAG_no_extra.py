import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, input1, input2, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # 添加 AdjustChannels 模块
        self.adjust_channels_x = AdjustChannels(input1, dim)  # 用于调整 x 的通道数
        self.adjust_channels_y = AdjustChannels(input2, dim)  # 用于调整 y 的通道数

    def forward(self, x, y):
        # 调整特征维度
        x = self.adjust_channels_x(x)
        y = self.adjust_channels_y(y)
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))

        # 结果融合
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


class AdjustChannels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustChannels, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 动态迁移权重到与输入张量相同的设备
        self.conv = self.conv.to(x.device)
        return self.conv(x)



if __name__ == '__main__':

    # 假设input1和input2的尺寸分别是 (4, 32, 1024, 1024) 和 (4, 384, 1024, 1024)
    input1 = torch.rand(4, 32, 64, 64)
    input2 = torch.rand(4, 384, 64, 64)# 4,256,64,64

    # 输入调整后的input2和input1进行融合
    block = CGAFusion(input1.shape[1], input2.shape[1], 256)  # 将模型移动到GPU
    output = block(input1, input2)
    print(output.size())  # 输出大小