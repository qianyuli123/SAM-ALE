from functools import partial
import torch
from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
from torchmetrics import MetricCollection, JaccardIndex, F1Score, Dice
import time
from losses import SAMLoss
import os
import numpy as np
from PIL import Image  # 用于保存为灰度图像
import datetime  # 用于获取当前时间
import pytz  # 确保安装 pytz 库：pip install pytz
class SamSeg(LightningModule):
    def __init__(
        self,
        cfg,
        sam_model: nn.Module,
        metrics: MetricCollection,
        num_classes: int,
        focal_cof: float = 20.0,
        dice_cof: float = 1.0,
        iou_cof: float = 1.0,
        ce_cof: float = 0.0,
        lr: float = 0.0001,
        weight_decay: float = 0.01,
        lr_steps: list = (10, 20),
        warmup_steps: int = 0,
        ignored_index=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["sam_model", "metrics"])
        self.model = sam_model
        self.num_classes = num_classes

        self.loss = SAMLoss(focal_cof, dice_cof, ce_cof, iou_cof)
        self.epoch_loss = 0.0
        if metrics is not None:
            self.train_metrics = metrics.clone(postfix='/train')
            self.valid_metrics = nn.ModuleList([
                metrics.clone(postfix='/val'),
                metrics.clone(postfix='/test')
            ])
            self.test_metrics = metrics.clone(prefix='final_test/')

        self.lr = lr
        self.ignored_index = ignored_index

        self.time_and_cnt = [0.0, 0]

        # 初始化验证集样本计数器
        self.val_sample_counts = [0, 0]
        
        self.idx = 1
        # 定义结果保存文件路径
        # 定义结果保存文件路径，文件名包含当前时间戳
        # 设置时区为 "Asia/Shanghai"（中国标准时间）
        timezone = pytz.timezone("Asia/Shanghai")
        current_time = datetime.datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        self.result_file = f"semi_sam_extra_{current_time}.txt"

        with open(self.result_file, "w") as f:
            f.write(f"Validation Results Log\n{'='*50}\n")

        # 在每个验证周期开始前重置样本计数器
        self.on_validation_epoch_start_callbacks()

        self.best_metric = float('-inf')  # 初始化最佳指标值
        self.best_epoch = -1  # 记录最佳模型的 epoch
        self.best_model_path = "total_model_CRAG_sam"  # 保存最佳模型的路径
        self.best_validation_mask_dir = f"semi_validation_masks_CRAG_sam_noextra_{current_time}"
        self.best_validation_gt_dir = "semi_validation_gt_CRAG_sam"
        self.best_train_mask_dir = f"dataset/CRAG_pseudo/mask"
        self.best_train_gt_dir = "semi_train_gt_CRAG_sam"
        os.makedirs(self.best_validation_mask_dir, exist_ok=True)

        # 缓存 mask_cls_pred，用于保存最佳模型的 mask
        self.validation_masks_cache = []  # 用于存储当前 epoch 的 mask_cls_pred
        self.validation_gt_cache = []
        self.train_masks_cache = []  # 用于存储当前 epoch 的 mask_cls_pred
        self.train_gt_cache = []
        self.best_metric_train = 0.0
        self.is_write_gt_train = False
        self.is_write_gt = False
    def on_validation_epoch_start_callbacks(self):
        """在每个验证周期开始前重置样本计数器"""
        self.val_sample_counts = [0, 0]

    def forward(self, images):
        # 使用 forward 进行推理/预测
        pred_masks, iou_predictions = self.model(images)

        # pred_masks 和 iou_predictions 是列表
        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)

        return pred_masks, iou_predictions

    def calc_loss(self, pred_masks, gt_masks, iou_predictions, ignored_masks):
        loss_dict = self.loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks)
        assert "loss" in loss_dict
        return loss_dict

    @torch.no_grad()
    def process_masks(self, gt_masks):
        # 处理忽略的掩码
        ignored_masks = gt_masks == 0
        ignored_masks = ignored_masks.unsqueeze(1).long()
        return gt_masks, ignored_masks

    def predict_mask(self, pred_masks, gt_masks, ignored_masks):
        # pred_masks 的形状：[batch_size, #classes, h, w]
        # 类别0总是用于忽略
        pred_masks = torch.argmax(pred_masks[:, 1:, ...], dim=1) + 1
        pred_masks = pred_masks * (1 - ignored_masks.squeeze(1))

        if self.ignored_index is not None:
            pred_masks[pred_masks == self.ignored_index] = 0
            gt_masks[gt_masks == self.ignored_index] = 0

        return pred_masks, gt_masks

    def training_step(self, batch, batch_idx):
        images, gt_masks = batch
        gt_masks, ignored_masks = self.process_masks(gt_masks)# 1,1536,1536

        pred_masks, iou_predictions = self(images)# 1,3,1536,1536;1,3,1
        losses = self.calc_loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks)

        self.log_losses(losses, "train")

        mask_cls_pred, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)# 2;0,2
        self.train_metrics.update(mask_cls_pred, gt_masks)

        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)

        return losses["loss"]

    def on_train_epoch_end(self):
        # 获取训练集的测试结果
        metrics = self.train_metrics.compute()
        # dice_score = 0
        # iou_score = 0

        # for metric_name, metric_value in  metrics.items():
        #     if "Dice_Bal/train" in metric_name:
        #         dice_score = metric_value
        #     elif "IOU_Jaccard/train" in metric_name:
        #         iou_score = metric_value
        # if dice_score > self.best_metric_train:
        #     self.best_metric_train = dice_score
        #     self.save_train_masks()
        # 保存训练集的测试结果到文件
        # with open(self.result_file, "a") as f:
        #     f.write(f"Epoch {self.current_epoch} Train Results:\n")
        #     for metric_name, metric_value in metrics.items():
        #         f.write(f"  {metric_name}: {metric_value:.4f}\n")
        #     f.write("=" * 50 + "\n")
        
        # 重置训练指标
        self.train_metrics.reset()


    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # print(dataloader_idx)
        images, gt_masks = batch
        gt_masks, ignored_masks = self.process_masks(gt_masks)

        prefix = get_prefix_from_val_id(dataloader_idx)
        metrics_idx = dataloader_idx if dataloader_idx is not None else 0

        pred_masks, iou_predictions = self(images)
        losses = self.calc_loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks)

        mask_cls_pred, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)

        if not self.trainer.sanity_checking and dataloader_idx <=1:
            self.log_losses(losses, prefix)
            self.valid_metrics[metrics_idx].update(mask_cls_pred, gt_masks)

            # 更新对应验证集的样本计数
            batch_size = images.size(0)
            if metrics_idx < len(self.val_sample_counts):
                self.val_sample_counts[metrics_idx] += batch_size
            else:
                # 如果有更多的验证集，动态扩展计数器
                while len(self.val_sample_counts) <= metrics_idx:
                    self.val_sample_counts.append(0)
                self.val_sample_counts[metrics_idx] += batch_size

        # 缓存验证集的 mask_cls_pred，仅对 dataloader_idx == 1
        if dataloader_idx == 1:
            self.validation_masks_cache.append((batch_idx, mask_cls_pred))
            self.validation_gt_cache.append((batch_idx, gt_masks))

        if dataloader_idx == 2:
            self.train_masks_cache.append((batch_idx, mask_cls_pred))
            self.train_gt_cache.append((batch_idx, gt_masks))

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            results = []
            for idx, valM in enumerate(self.valid_metrics):
                metrics_result = valM.compute()
                prefix = get_prefix_from_val_id(idx)
                sample_count = self.val_sample_counts[idx] if idx < len(self.val_sample_counts) else 0

                dice_score = 0
                iou_score = 0

                # 记录评估结果
                result_str = f"Dataset: {prefix}\n" \
                            f"Number of samples: {sample_count}\n" \
                            f"Metrics:\n"
                for metric_name, metric_value in metrics_result.items():
                    result_str += f"  {metric_name}: {metric_value:.4f}\n"
                    if "Dice_Bal/test" in metric_name:
                        dice_score = metric_value
                    elif "IOU_Jaccard/test" in metric_name:
                        iou_score = metric_value

                # 标识验证集和测试集
                if prefix == "val":
                    result_str += "Type: Validation Set\n"
                elif prefix == "test":
                    result_str += "Type: Test Set\n"
                else:
                    result_str += "Type: Unknown\n"

                results.append(result_str)

                # 如果是 dataloader_idx=1，记录 Dice 和 IoU 并保存最佳模型
                if idx == 1:  # 仅对测试集（idx=1）进行记录和模型保存
                    # 计算 Dice 和 IoU 的加权平均（也可以只用某一个指标）
                    metric_score = dice_score

                    # 保存最佳模型
                    if metric_score > self.best_metric:

                        self.best_metric = metric_score
                        self.best_epoch = self.current_epoch

                        # 保存模型
                        # best_model_path = f"{self.best_model_path}/best_model_epoch{self.current_epoch}.ckpt"
                        # self.trainer.save_checkpoint(best_model_path)
                        # print(f"Saved new best model at {best_model_path} with metric score: {metric_score:.4f}")

                        # 保存验证集 mask 和 gt
                        self.save_validation_masks()

                # 重置指标
                valM.reset()
            self.save_train_masks()
            self.train_masks_cache = []
            self.train_gt_cache = []
            # 清空缓存
            self.validation_masks_cache = []
            self.validation_gt_cache = []
            # 获取当前学习率
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']  # Accessing the lr from the first parameter group

            # 将结果和当前学习率写入文本文件
            with open(self.result_file, "a") as f:
                f.write(f"Epoch {self.current_epoch} Results:\n")
                for res in results:
                    f.write(res + "\n")
                f.write(f"Current Learning Rate: {current_lr:.6f}\n")  # 输出当前学习率
                f.write("="*50 + "\n")

            # self.lr_scheduler.step(val_loss)  # Update based on dice_score or any other metric

    def save_train_masks(self):
        """
        仅在最佳模型时保存验证集 mask_cls_pred。
        """
        # save_dir = os.path.join(self.best_validation_mask_dir, f"epoch_{self.current_epoch}")
        save_dir = self.best_train_mask_dir
        os.makedirs(save_dir, exist_ok=True)

        for batch_idx, mask_cls_pred in self.train_masks_cache:
            for i, mask in enumerate(mask_cls_pred):
                # 将 mask 转为 numpy 格式
                mask_np = mask.cpu().numpy().astype(np.uint8)
                mask_np = np.where(mask_np == 1, 0, mask_np)
                mask_np = np.where(mask_np == 2, 1, mask_np)
                # 保存为灰度图像
                save_path = os.path.join(save_dir, f"train_{self.idx}.png")
                Image.fromarray(mask_np).save(save_path)
                print(f"Saved train mask: {save_path}")
                self.idx = self.idx + 1
        self.idx = 1

        # save_dir = os.path.join(self.best_train_gt_dir, f"epoch_{self.current_epoch}")
        if not self.is_write_gt_train:
            save_dir = self.best_train_gt_dir
            os.makedirs(save_dir, exist_ok=True)

            for batch_idx, mask_cls_pred in self.train_gt_cache:
                for i, mask in enumerate(mask_cls_pred):
                    # 将 mask 转为 numpy 格式
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    mask_np = np.where(mask_np == 1, 0, mask_np)
                    mask_np = np.where(mask_np == 2, 1, mask_np)
                    # 保存为灰度图像
                    save_path = os.path.join(save_dir, f"train_{self.idx}.png")
                    Image.fromarray(mask_np).save(save_path)
                    print(f"Saved gt mask: {save_path}")
                    self.idx = self.idx + 1
            self.idx = 1
            self.is_write_gt_train = True
    def save_validation_masks(self):
        """
        仅在最佳模型时保存验证集 mask_cls_pred。
        """
        # save_dir = os.path.join(self.best_validation_mask_dir, f"epoch_{self.current_epoch}")
        save_dir = self.best_validation_mask_dir
        os.makedirs(save_dir, exist_ok=True)

        for batch_idx, mask_cls_pred in self.validation_masks_cache:
            for i, mask in enumerate(mask_cls_pred):
                # 将 mask 转为 numpy 格式
                mask_np = mask.cpu().numpy().astype(np.uint8)

                # 保存为灰度图像
                save_path = os.path.join(save_dir, f"test_{self.idx}.png")
                Image.fromarray(mask_np).save(save_path)
                print(f"Saved validation mask: {save_path}")
                self.idx = self.idx + 1
        self.idx = 1

        # save_dir = os.path.join(self.best_validation_gt_dir, f"epoch_{self.current_epoch}")
        # if not self.is_write_gt:
        #     save_dir = self.best_validation_gt_dir
        #     os.makedirs(save_dir, exist_ok=True)

        #     for batch_idx, mask_cls_pred in self.validation_gt_cache:
        #         for i, mask in enumerate(mask_cls_pred):
        #             # 将 mask 转为 numpy 格式
        #             mask_np = mask.cpu().numpy().astype(np.uint8)

        #             # 保存为灰度图像
        #             save_path = os.path.join(save_dir, f"test_{self.idx}.png")
        #             Image.fromarray(mask_np).save(save_path)
        #             print(f"Saved gt mask: {save_path}")
        #             self.idx = self.idx + 1
        #     self.idx = 1
        #     self.is_write_gt = True

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        images = batch
        # pred_masks, iou_predictions = self(images)
        with torch.no_grad():
            time_start = time.perf_counter()
            pred_masks, iou_predictions = self.model(images)
            time_predict = time.perf_counter() - time_start

        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)

        self.time_and_cnt[0] += time_predict
        self.time_and_cnt[1] += 1
        print("Average prediction time: %f" % (self.time_and_cnt[0] / self.time_and_cnt[1]))

        # pred_masks, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)

        # pred_masks 的形状：[batch_size, #classes, h, w]
        # 类别0总是用于忽略
        pred_masks = torch.argmax(pred_masks[:, 1:, ...], dim=1) + 1
        return pred_masks

    def log_losses(self, losses, prefix):
        if prefix == "train":
            for t in losses:
                self.log(f"Loss/{prefix}_{t}", losses[t], on_epoch=True, on_step=True, sync_dist=True)
        else:
            for t in losses:
                self.log(f"Loss/{prefix}_{t}", losses[t], on_epoch=True, on_step=False, sync_dist=True,
                         add_dataloader_idx=False)

    def configure_optimizers(self):
        # self.hparams 可用，因为我们调用了 self.save_hyperparameters()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)

        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            elif step < self.hparams.lr_steps[0]:
                return 1.0
            elif step < self.hparams.lr_steps[1]:
                return 0.5
            else:
                return 0.3

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=False)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }  # [optimizer], [scheduler]

def get_prefix_from_val_id(dataloader_idx):
    if dataloader_idx is None or dataloader_idx == 0:
        return "val"
    elif dataloader_idx == 1:
        return "test"
    elif dataloader_idx == 2:
        return "train_pseudo"
    else:
        raise NotImplementedError

