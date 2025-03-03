import torch
import torchvision.models as models
import torch.nn as nn

class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        
        # 加载预训练的ResNet-18模型
        self.resnet = models.resnet18(pretrained=False)
        
        # 修改第一层卷积，输出通道设置为64（保持原始ResNet-18的设计）
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改第一个残差块的输出通道数为64，保持空间尺寸
        self.resnet.layer1 = self._make_layer(64, 64, stride=2)  # 通道数64，空间尺寸减半
        
        # 修改后面的残差块，以保持较大的空间尺寸并增加通道数
        self.resnet.layer2 = self._make_layer(64, 128, stride=2)  # 128通道，空间尺寸 256x256
        self.resnet.layer3 = self._make_layer(128, 256, stride=2)  # 256通道，空间尺寸 128x128
        self.resnet.layer4 = self._make_layer(256, 384, stride=1)  # 384通道，空间尺寸 64x64
        
        # 删除 ResNet 的全连接层 (fc)，不需要进行分类任务
        self.resnet.fc = nn.Identity()  # 删除原始的全连接层
        
        # 添加层归一化
        self.norm = nn.LayerNorm(384)
    
    def _make_layer(self, in_channels, out_channels, stride):
        """构建一个残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        # 通过 ResNet 的每一层
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        x = self.resnet.layer1[0](x)
        x = self.resnet.layer1[1](x)
        x = self.resnet.layer2[0](x)
        x = self.resnet.layer2[1](x)
        x = self.resnet.layer3[0](x)
        x = self.resnet.layer3[1](x)
        x = self.resnet.layer4[0](x)
        x = self.resnet.layer4[1](x)
        
        # 修改通道数
        x = x.flatten(2).transpose(1, 2)
        
        # 对输出进行层归一化
        x = self.norm(x)
        
        return x

if __name__ == "__main__":
    # 创建模型实例
    model = ModifiedResNet18()

    # 创建一个随机输入的批次，形状为 (6, 3, 1024, 1024)
    x = torch.randn(6, 3, 1024, 1024)

    # 将模型设为评估模式
    model.eval()

    print(model)

    # resnet = models.resnet18(pretrained=False)

    # print(resnet)
    # 前向传播，获取输出特征
    with torch.no_grad():
        output = model(x)

    # 打印输出的维度
    print(f'Final output shape: {output.shape}')  # 应该是 (6, 4096, 64, 64)
