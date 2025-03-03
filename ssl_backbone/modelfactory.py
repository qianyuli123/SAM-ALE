import torch
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus

class ModelFactory:
    def __init__(self, num_classes, in_channels=3):
        self.num_classes = num_classes
        self.in_channels = in_channels

    def get_model(self, model_name, num_classes=2):
        if model_name == 'effunet':
            model = Unet(
                encoder_name='efficientnet-b2',
                encoder_weights="imagenet",
                dropout=0.5,
                in_channels=3,
                classes=num_classes,
                activation='sigmoid'
            )
        elif model_name == 'unet_resnet18':
            model = Unet(
                encoder_name='resnet18',
                encoder_weights="imagenet",
                dropout=0.5,
                in_channels=3,
                classes=num_classes,
                activation='sigmoid'
            )
        elif model_name == 'unet_resnet34':
            model = Unet(
                encoder_name='resnet34',
                encoder_weights="imagenet",
                dropout=0.5,
                in_channels=3,
                classes=num_classes,
                activation='sigmoid'
            )
        elif model_name == 'unet_resnet50':
            model = Unet(
                encoder_name='resnet50',
                encoder_weights="imagenet",
                dropout=0.5,
                in_channels=3,
                classes=num_classes,
                activation='sigmoid'
            )
        elif model_name == 'unetplusplus':
            model = UnetPlusPlus(
                encoder_weights=None,
                classes=num_classes
            )
        elif model_name == 'deeplabv3':
            model = DeepLabV3(
                encoder_weights=None,
                classes=num_classes
            )
        elif model_name == 'deeplabv3plus':
            model = DeepLabV3Plus(
                encoder_weights=None,
                classes=num_classes,
                pooling='avg',             # one of 'avg', 'max'
                dropout=0.5,               # dropout ratio, default is None
                activation='sigmoid',      # activation function, default is None
            )
        return model

    def save_encoder_weights_as_zero(self, model, save_path):
        """
        将模型编码器的权重设置为0，并保存。
        
        :param model: 要操作的模型
        :param save_path: 保存模型权重的路径
        """
        # 将编码器的权重设置为0
        for param in model.encoder.parameters():
            param.data.fill_(0)
        
        # 保存编码器的权重
        torch.save(model.encoder.state_dict(), save_path)
        print(f"Encoder weights saved to {save_path} with all zeros.")

if __name__ == '__main__':
    # 示例用法
    model_factory = ModelFactory(num_classes=2, in_channels=3)
    model = model_factory.get_model("effunet")

    # 将模型编码器的权重设置为0并保存
    model_factory.save_encoder_weights_as_zero(model, 'extra_encoder/encoder_weights_zero.pth')

