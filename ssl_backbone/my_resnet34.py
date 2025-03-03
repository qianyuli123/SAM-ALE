import torch
from torchvision.models import resnet34
import torch.nn as nn

# ================== 修改后的 ResNet34 Encoder ==================
class ResNetEncoder(torch.nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.encoder = resnet34(pretrained=True)  # 初始化 ResNet34
        self.encoder_layers = list(self.encoder.children())[:-2]  # 去掉全连接层和平均池化层
        self.enc1 = torch.nn.Sequential(*self.encoder_layers[:3])  # 前3个模块
        self.enc2 = torch.nn.Sequential(*self.encoder_layers[3:5])  # 中间模块
        self.enc3 = self.encoder_layers[5]
        self.enc4 = self.encoder_layers[6]
        self.enc5 = self.encoder_layers[7]
        # 修改 normalized_shape 为 [256, 64, 64]
        self.norm = nn.LayerNorm([256, 64, 64])

    def forward(self, x):
        enc1 = self.enc1(x)  # 特征1
        enc2 = self.enc2(enc1)  # 特征2
        enc3 = self.enc3(enc2)  # 特征3
        enc4 = self.enc4(enc3)  # 特征4

        res = self.norm(enc4)  # 应用层归一化
        return res  # 返回调整后的 logits


