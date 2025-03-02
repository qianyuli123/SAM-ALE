import os
import pandas as pd
import torch
import numpy as np
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize, ColorJitter, GaussianBlur, RandomAdjustSharpness
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from itertools import cycle
from torchmetrics import JaccardIndex, Dice
from torchmetrics import MetricCollection
import math
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import torchvision
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus

from torch.utils.tensorboard import SummaryWriter

import shutil

# 清除之前的 TensorBoard 日志
log_dir = "runs/phase1_monuseg"  # 指定日志文件夹路径
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)  # 删除整个日志目录

# 初始化 TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# 1.记录一下纯监督的loss收敛情况（第几个epoch收敛，再加入伪标签观察结果）
# 2.只用有标签数据微调到收敛，结果记录
# 3.Dice loss查看一下有没有问题或者自己写一个

# ================== 参数设置 ==================
data_dir = "monuseg"  # 数据集路径
image_dir = os.path.join(data_dir, "img")  # 图像路径
mask_dir = os.path.join(data_dir, "mask")  # 掩码路径
csv_path = os.path.join(data_dir, "monuseg_label4.csv")  # CSV 文件路径
batch_size = 1
num_epochs = 500
learning_rate = 1e-4
lambda_max = 10  # 一致性损失最大权重
consistency_rampup = 10
confidence_threshold_1 = 0.99  # 伪标签置信度阈值
confidence_threshold_2 = 0.99  # 伪标签置信度阈值
confidence_threshold_3 = 0.99  # 伪标签置信度阈值
num_classes = 2  # 分割任务类别数（背景+目标）
ema_decay = 0.995  # EMA 衰减系数
unsup_warm_up = 0.4
# 有标签数据和无标签数据的 DataLoader 参数
labeled_bs = 2  # 每个 batch 中有标签数据的数量
unlabeled_bs = 4  # 每个 batch 中无标签数据的数量
# 保存 encoder 的路径
encoder_save_path = "phase1_monuseg.pth"
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomAffine


def sigmoid_rampup(current, rampup_length):
    """
    实现 Sigmoid Ramp-up 函数，用于动态调整损失权重。

    :param current: 当前的 epoch 数
    :param rampup_length: ramp-up 持续的长度（总的 epoch 数）
    :return: 当前的权重值（介于 0 到 1 之间）
    """
    if rampup_length == 0:
        return 1.0
    else:
        # 确保 current 在 [0, rampup_length] 区间
        current = max(0.0, min(current, rampup_length))
        phase = 1.0 - current / rampup_length
        return math.exp(-5.0 * phase * phase)

class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, preds, targets):
        # preds 和 targets 形状为 (N, C, H, W)，需要通过 softmax 和 one-hot 处理
        smooth = 1e-6  # 平滑项，避免分母为零
        preds = torch.softmax(preds, dim=1)  # 转为概率分布
        targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(preds * targets, dim=(2, 3))
        union = torch.sum(preds, dim=(2, 3)) + torch.sum(targets, dim=(2, 3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return 1 - torch.mean(iou)  # 返回 IoU 损失
import random
from torchvision.transforms import functional

class SynchronizedTransform:
    def __init__(self):
        self.transforms = [
            self.random_horizontal_flip,
            self.random_vertical_flip,
            self.random_rotation,
        ]

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask

    def random_horizontal_flip(self, image, mask, p=0.5):
        if random.random() < p:
            image = functional.hflip(image)
            mask = functional.hflip(mask)
        return image, mask

    def random_vertical_flip(self, image, mask, p=0.5):
        if random.random() < p:
            image = functional.vflip(image)
            mask = functional.vflip(mask)
        return image, mask

    def random_rotation(self, image, mask, degrees=30):
        angle = random.uniform(-degrees, degrees)
        image = functional.rotate(image, angle)
        mask = functional.rotate(mask, angle)
        return image, mask

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return lambda_max * sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    """
    更新 EMA 模型参数。
    
    :param model: 学生模型，用于获取最新参数
    :param ema_model: 教师模型，用于平滑参数
    :param alpha: 平滑因子，控制新旧参数的权重
    :param global_step: 当前训练步数，用于动态调整 alpha
    """
    # 动态调整 alpha（训练初期使用平均，后期切换为 EMA）
    alpha = min(1 - 1 / (global_step + 1), alpha)
    
    # 遍历学生模型与教师模型对应的参数
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # 公式：ema_param = alpha * ema_param + (1 - alpha) * param
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# ================== 自定义评价函数 ==================
def initialize_metrics(num_classes):
    """
    初始化评估指标，支持二分类和多类别分割任务。
    
    :param num_classes: 分割任务的类别数（包括背景）
    :return: MetricCollection 对象，包含 IoU 和 Dice 指标
    """
    # 判断任务类型
    task = 'binary' if num_classes == 2 else 'multiclass'
    # 创建评估指标集合
    metrics = MetricCollection({
        # IoU (Jaccard Index)
        "IOU_Jaccard": JaccardIndex(
            num_classes=num_classes,
            ignore_index=0,  # 忽略背景类别
            task=task,       # 根据任务类型切换
            average="micro"  # 对所有类别求整体 IoU
        ),
        # Dice 系数
        "Dice_Bal": Dice(
            num_classes=num_classes,
            ignore_index=0,                # 忽略背景类别
            average="macro" if num_classes > 2 else "micro"  # 二分类用 micro，多类别用 macro
        ),
    })
    return metrics


def calculate_metrics(true_mask, pred_mask, metrics):
    metrics_output = metrics(pred_mask, true_mask)
    dice = metrics_output["Dice_Bal"].item()
    iou = metrics_output["IOU_Jaccard"].item()
    return dice, iou

# ================== 数据增强与预处理 ==================
train_transform_img = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform_img = Compose([
    Resize((256, 256)),
    # RandomHorizontalFlip(p=0.5),
    # RandomVerticalFlip(p=0.5),
    # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # RandomAffine(degrees=10, translate=(0.1, 0.1)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform_mask = Compose([
    Resize((256, 256)),
    ToTensor(),
])

unlabeled_transform_img = Compose([
    Resize((256, 256)),
    # RandomHorizontalFlip(p=0.5),
    # RandomVerticalFlip(p=0.5),
    # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # RandomAffine(degrees=10, translate=(0.1, 0.1)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# ================== 模型定义 ==================
# 1.efficient unet框架
# 2.FPN多尺度融合特征
# 3.sam encoder unet结合的开源框架，encoder直接使用，参考nnunet搭自己的网络结构
# 4.试试其它的8张图像组合（至少两个）
# 5.新的数据集，monuseg,monusac,conic

# student_model = UNet(
#     spatial_dims=2,
#     in_channels=3,
#     out_channels=num_classes,
#     channels=(64, 128, 256, 512, 1024),
#     strides=(2, 2, 2, 2)
# ).cuda(3)

# teacher_model = UNet(
#     spatial_dims=2,
#     in_channels=3,
#     out_channels=num_classes,
#     channels=(64, 128, 256, 512, 1024),
#     strides=(2, 2, 2, 2)
# ).cuda(3)
def get_model(model_name, num_classes):
    if model_name == 'effunet':
        model = Unet(
            encoder_name='efficientnet-b2',
            encoder_weights="imagenet",
            dropout=0.5,
            in_channels=3,
            classes=num_classes,
            activation='sigmoid'
        )
    elif model_name == 'unet':
        model = Unet(
            encoder_weights=None,
            classes=num_classes
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
# 0用的unet,1用的unetplusplus,2用的是deeplabv3,5用的是deeplabv3plus
# efficientunet结果是0.45,0.5
# 
student_model = get_model("effunet",num_classes=2).cuda(3)
teacher_model = get_model("effunet",num_classes=2).cuda(3)

# 初始化教师模型权重
teacher_model.load_state_dict(student_model.state_dict())
teacher_model.eval()

# 优化器与调度器
optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=10, verbose=True)

# 损失函数
dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
ce_loss = torch.nn.CrossEntropyLoss()
metrics = initialize_metrics(num_classes=num_classes).cuda(3)

# ================== 数据集定义 ==================
class LabeledDataset(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, transform_img=None, transform_mask=None, augmentations=None, fold_filter=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.augmentations = augmentations  # 添加同步变换

        if fold_filter is not None:
            self.data = self.data[self.data["fold"].isin(fold_filter)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.loc[idx, "img_id"]
        img_path = os.path.join(self.image_dir, img_id + ".png")
        mask_path = os.path.join(self.mask_dir, img_id + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augmentations:
            image, mask = self.augmentations(image, mask)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        mask = (mask != 0).float()  # 二值化掩码
        return {"image": image, "mask": mask}


class UnlabeledDataset(Dataset):
    def __init__(self, image_dir, mask_dir, unlabeled_list, transform_img, transform_mask):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.unlabeled_list = unlabeled_list
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.unlabeled_list)

    def __getitem__(self, idx):
        img_name = self.unlabeled_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = self.transform_img(image)
        mask = self.transform_mask(mask)
        mask = (mask != 0).float()  # 二值化掩码
        # 添加空的 mask 键
        return {"image": image, "mask": mask}

def get_unlabeled_images(image_dir, csv_file):
    all_images = set(os.listdir(image_dir))
    labeled_images = set(pd.read_csv(csv_file)["img_id"] + ".png")
    return list(all_images - labeled_images)

# 自定义 Sampler
class SemiSupervisedSampler(Sampler):
    def __init__(self, len_labeled, len_unlabeled, labeled_bs, unlabeled_bs):
        self.len_labeled = len_labeled
        self.len_unlabeled = len_unlabeled
        self.labeled_bs = labeled_bs
        self.unlabeled_bs = unlabeled_bs

    def __iter__(self):
        # 随机打乱有标签和无标签数据的索引
        labeled_indices = np.random.permutation(self.len_labeled)
        unlabeled_indices = np.random.permutation(self.len_unlabeled)

        # 对无标签数据的索引加上偏移量
        unlabeled_indices += self.len_labeled

        # 每个 epoch 的批次数量
        num_batches = max(
            math.ceil(self.len_labeled / self.labeled_bs),
            math.ceil(self.len_unlabeled / self.unlabeled_bs)
        )

        batch_indices = []
        for i in range(num_batches):
            # 采样有标签数据的索引
            labeled_batch = labeled_indices[i * self.labeled_bs:(i + 1) * self.labeled_bs]
            # 采样无标签数据的索引
            unlabeled_batch = unlabeled_indices[i * self.unlabeled_bs:(i + 1) * self.unlabeled_bs]

            # 如果不足一个 batch，用重复采样补充
            if len(labeled_batch) < self.labeled_bs:
                labeled_batch = np.random.choice(labeled_indices, self.labeled_bs, replace=True)
            if len(unlabeled_batch) < self.unlabeled_bs:
                unlabeled_batch = np.random.choice(unlabeled_indices, self.unlabeled_bs, replace=True)

            # 合并有标签和无标签数据的索引
            batch_indices.extend(np.concatenate([labeled_batch, unlabeled_batch]))

        # 返回生成器，逐个返回索引
        return iter(batch_indices)

    def __len__(self):
        return self.len_labeled + self.len_unlabeled

    
# ================== 数据加载 ==================
synchronized_transform = SynchronizedTransform()

labeled_dataset = LabeledDataset(
    csv_file=csv_path,
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform_img=train_transform_img,
    transform_mask=val_transform_mask,
    augmentations=synchronized_transform,
    fold_filter=[1,2,3,4]
)

train_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)

unlabeled_images = get_unlabeled_images(image_dir, csv_path)
unlabeled_dataset = UnlabeledDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    unlabeled_list=unlabeled_images,
    transform_img=val_transform_img,
    transform_mask=val_transform_mask
)


semi_supervised_loader = DataLoader(
    labeled_dataset + unlabeled_dataset,
    sampler=SemiSupervisedSampler(len(labeled_dataset), len(unlabeled_dataset), labeled_bs, unlabeled_bs),
    batch_size=labeled_bs + unlabeled_bs,  # Sampler 管理 batch 的大小
    num_workers=4,
    pin_memory=True
)

test_dataset = LabeledDataset(
    csv_file=csv_path,
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform_img=val_transform_img,
    transform_mask=val_transform_mask,
    fold_filter=[-1]
)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ================== 半监督训练 ==================
def visualize_pseudo_labels(images_unsup, masks_unsup, preds_teacher, pseudo_labels, confidence, confidence_mask, masked_pseudo_labels,epoch, writer):
    """
    将伪标签和相关信息可视化到 TensorBoard，合并到一张图中。
    """
    # 获取未标注图像
    images_grid = torchvision.utils.make_grid(images_unsup[:4],normalize=True)
    masks_grid = torchvision.utils.make_grid(masks_unsup[:4].float())
    # 获取伪标签
    pseudo_labels_grid = torchvision.utils.make_grid(pseudo_labels[:4].unsqueeze(1).float())
    # 获取置信度掩码
    confidence_mask_grid = torchvision.utils.make_grid(confidence_mask[:4].unsqueeze(1).float())

    masked_pseudo_labels_grid = torchvision.utils.make_grid(masked_pseudo_labels[:4].float())

    # 合并为单张图像（纵向拼接）
    combined_grid = torch.cat([images_grid, masks_grid, pseudo_labels_grid, confidence_mask_grid, masked_pseudo_labels_grid], dim=1)  # 按通道拼接

    # 将合并后的图像写入 TensorBoard
    writer.add_image(f"Combined Visualization_1/Epoch_{epoch}", combined_grid, epoch)


best_iou = 0.0
best_dice = 0.0
best_epoch = -1

for epoch in range(num_epochs):
    student_model.train()
    epoch_loss = 0
    lambda_pgt = get_current_consistency_weight(epoch)
    
    # 看全监督到多少轮次开始收敛
    if epoch <= 100:
        confidence_threshold = 1.1
    else:
        confidence_threshold = confidence_threshold_3

    # 前30个epoch只用有标签数据，后三十个既有有标签也有无标签
    for batch_idx, batch in enumerate(tqdm(semi_supervised_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
        images_sup, masks_sup = batch["image"][:labeled_bs].cuda(3), batch["mask"][:labeled_bs].cuda(3)
        preds_sup = student_model(images_sup)

        loss_supervised = dice_loss(preds_sup, masks_sup) + ce_loss(preds_sup, masks_sup.long().squeeze(1))

        # 无监督训练部分
        images_unsup = batch["image"][labeled_bs:].cuda(3)
        masks_unsup = batch["mask"][labeled_bs:].cuda(3)
        preds_student = student_model(images_unsup)
        with torch.no_grad():
            preds_teacher = teacher_model(images_unsup)

        # 置信度计算
        confidence, pseudo_labels = torch.max(preds_teacher.softmax(dim=1), dim=1)  # 最大概率和伪标签类别
        confidence_mask = confidence > confidence_threshold  # 根据阈值生成掩码

        masked_pseudo_labels = (pseudo_labels * confidence_mask).unsqueeze(1)
        # 可视化伪标签
        if batch_idx == 0:  # 仅保存第一个批次的伪标签
            visualize_pseudo_labels(
                images_unsup, masks_unsup, preds_teacher, pseudo_labels, confidence, confidence_mask, masked_pseudo_labels, epoch, writer
            )

        # 只考虑高置信度
        preds_student = preds_student * confidence_mask.unsqueeze(1)
        preds_teacher = preds_teacher * confidence_mask.unsqueeze(1)

        # pgt损失 加上交叉熵损失
        pgt_loss = dice_loss(preds_student, masked_pseudo_labels)
        # 总损失
        total_loss = loss_supervised + lambda_pgt * pgt_loss

        # 优化学生模型
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 更新教师模型
        update_ema_variables(student_model, teacher_model, ema_decay, epoch)

        epoch_loss += total_loss.item()

        # 将训练损失写入 TensorBoard
        writer.add_scalar("Loss/Total_Loss", total_loss, epoch * (labeled_bs + unlabeled_bs) + batch_idx)
        writer.add_scalar("Loss/Supervised_Loss", loss_supervised, epoch * (labeled_bs + unlabeled_bs) + batch_idx)
        writer.add_scalar("Loss/Unsupervised_Loss(no weight)", pgt_loss, epoch * (labeled_bs + unlabeled_bs) + batch_idx)
        writer.add_scalar("Loss/Unsupervised_Loss(add weight)", lambda_pgt * pgt_loss, epoch * (labeled_bs + unlabeled_bs) + batch_idx)

    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

    # 测试集评估
    student_model.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            images = batch["image"].cuda(3)
            true_masks = batch["mask"].long().squeeze(1).cuda(3)
            preds = student_model(images)
            preds_binary = torch.argmax(preds, dim=1)

            dice, iou = calculate_metrics(true_masks, preds_binary, metrics)
            dice_scores.append(dice)
            iou_scores.append(iou)

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    writer.add_scalar("Metrics/Validation_Dice", avg_dice, epoch)
    writer.add_scalar("Metrics/Validation_IoU", avg_iou, epoch)
    print(f"Epoch {epoch + 1} Average Dice Score: {avg_dice:.4f}, Average IoU: {avg_iou:.4f}")

    # 保存最佳模型
    if avg_dice + avg_iou > best_dice + best_iou:
        best_epoch = epoch
        best_dice = avg_dice
        best_iou = avg_iou
        torch.save(student_model.encoder.state_dict(), encoder_save_path)
        print(f"Encoder saved at epoch {epoch + 1} with Dice: {best_dice:.4f}, IoU: {best_iou:.4f}")

# 输出最终结果
print(f"Training Complete. Best Dice Score: {best_dice:.4f}, Best IoU: {best_iou:.4f}")

#一定要记得加该行！！！
writer.close()
