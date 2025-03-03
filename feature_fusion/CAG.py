import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
        self.dropout = nn.Dropout2d(dropout_rate)  # 添加 Dropout

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        x2 = self.dropout(x2)  # 在空间注意力输入上应用 Dropout
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8, dropout_rate=0.1):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(dropout_rate)  # 添加 Dropout

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        x2 = self.dropout(x2)  # 在组合后的输入特征上应用 Dropout
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, input1, input2, dim, reduction=8, dropout_rate=0.3):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention(dropout_rate)
        self.ca = ChannelAttention(dim, reduction, dropout_rate)
        self.pa = PixelAttention(dim, dropout_rate)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # 添加 AdjustChannels 模块
        self.adjust_channels_x = AdjustChannels(input1, dim)  # 用于调整 x 的通道数
        self.adjust_channels_y = AdjustChannels(input2, dim)  # 用于调整 y 的通道数

        # Dropout 融合后的正则化
        self.dropout = nn.Dropout2d(dropout_rate)

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
        result = self.dropout(result)  # 在融合结果应用 Dropout
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
    # 检查是否有可用 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义输入张量
    input1 = torch.rand(4, 32, 64, 64).to(device)  # 输入1
    input2 = torch.rand(4, 384, 64, 64).to(device)  # 输入2

    # 定义 CGAFusion 模块并将其迁移到 GPU
    block = CGAFusion(32, 384, 256, dropout_rate=0.2).to(device)

    # 执行前向传播
    output = block(input1, input2)
    print("Output shape:", output.size())  # 输出大小
