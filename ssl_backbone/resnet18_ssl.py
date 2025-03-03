import torch
import torch.nn as nn
import torchvision.models as models

# ================== 定义 Encoder ==================
class Resnet_ssl(nn.Module):
    def __init__(self):
        super(Resnet_ssl, self).__init__()
        # 使用 ResNet-18 作为编码器
        resnet = models.resnet18(pretrained=False)  # 不加载预训练权重
        # 截取 ResNet-18 的前几层，保留到 layer3，输出为原始尺寸的 1/16
        self.encoder = nn.Sequential(*list(resnet.children())[:-3])  # 去掉 layer4 和 avgpool

    def forward(self, x):
        features = self.encoder(x)  # 编码特征
        return features