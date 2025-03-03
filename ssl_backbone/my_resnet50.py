import torch
import torchvision.models as models
import torch.nn as nn

class ModifiedResNet50(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        
        # 加载预训练的ResNet-50模型
        self.resnet = models.resnet50(pretrained=True)
        
        # 修改第一层卷积，输出通道设置为64（保持原始ResNet-50的设计）
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改ResNet-50的残差块结构
        self.resnet.layer1 = self._make_layer(64, 256, stride=1)  # 残差块1（256通道）
        self.resnet.layer2 = self._make_layer(256, 512, stride=2)  # 残差块2（512通道）
        self.resnet.layer3 = self._make_layer(512, 1024, stride=2)  # 残差块3（1024通道）
        self.resnet.layer4 = self._make_layer(1024,2048, stride=2)  # 残差块4（2048通道）

        # 删除 ResNet 的全连接层 (fc)，不需要进行分类任务
        self.resnet.fc = nn.Identity()  # 删除原始的全连接层
        
        # 添加一个卷积层将2048通道压缩到384通道
        self.conv_out = nn.Conv2d(2048, 384, kernel_size=1, stride=1, padding=0)
        
        # 添加层归一化
        self.norm = nn.LayerNorm(384)
    
    def _make_layer(self, in_channels, out_channels, stride):
        """构建一个残差块，使用ResNet-50的Bottleneck结构"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False),  # Bottleneck扩展
            nn.BatchNorm2d(out_channels * 4)
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
        
        # 将2048通道压缩到384通道
        x = self.conv_out(x)

        # 对输出进行层归一化
        x = x.flatten(2).transpose(1, 2)  # 扁平化并转置为 (batch, 64*64, 384)
        
        # 对输出进行层归一化
        x = self.norm(x)
        
        return x

if __name__ == "__main__":
    # 创建模型实例
    model = ModifiedResNet50()

    # 创建一个随机输入的批次，形状为 (6, 3, 1024, 1024)
    x = torch.randn(6, 3, 1024, 1024)

    # 将模型设为评估模式
    model.eval()

    # 前向传播，获取输出特征
    with torch.no_grad():
        output = model(x)

    # 打印最终的输出维度
    print(f'Final output shape: {output.shape}')  # 结果是 (6, 4096, 384)

    # 打印模型结构
    print(model)
