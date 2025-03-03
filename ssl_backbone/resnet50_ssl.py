import torch
import torch.nn as nn
import torchvision.models as models

# ================== 定义 Encoder ==================
class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        # 使用 ResNet-50 作为编码器
        resnet = models.resnet50(pretrained=False)  # 不加载预训练权重
        # 截取 ResNet-50 的前几层，保留到 layer3，输出为原始尺寸的 1/16
        self.encoder = nn.Sequential(*list(resnet.children())[:-3])  # 去掉 layer4 和 avgpool

    def forward(self, x):
        features = self.encoder(x)  # 编码特征
        return features