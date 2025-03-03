import os
# import numpy as np
# import cv2
# import csv
# from scipy.spatial.distance import directed_hausdorff
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor

# # 计算 Dice 系数
# def dice_coefficient(pred, true):
#     intersection = np.sum(pred * true)
#     return 2 * intersection / (np.sum(pred) + np.sum(true))

# # 计算 IoU
# def iou(pred, true):
#     intersection = np.sum(pred * true)
#     union = np.sum(pred) + np.sum(true) - intersection
#     return intersection / union if union != 0 else 0

# # 计算 Hausdorff 距离（降低分辨率计算以加速）
# def hausdorff_distance(pred, true):
#     pred_boundary = np.column_stack(np.where(pred > 0))
#     true_boundary = np.column_stack(np.where(true > 0))
#     if len(pred_boundary) == 0 or len(true_boundary) == 0:
#         return np.inf  # 如果边界点为空，返回无穷大
#     hd = directed_hausdorff(pred_boundary, true_boundary)[0]
#     return hd

# # 计算 ASSD
# def assd(pred, true):
#     pred_boundary = np.column_stack(np.where(pred > 0))
#     true_boundary = np.column_stack(np.where(true > 0))
#     if len(pred_boundary) == 0 or len(true_boundary) == 0:
#         return np.inf  # 避免无效计算
#     total_distance = np.mean([np.min(np.linalg.norm(true_boundary - p, axis=1)) for p in pred_boundary] +
#                              [np.min(np.linalg.norm(pred_boundary - t, axis=1)) for t in true_boundary])
#     return total_distance

# # 预加载图像到内存
# def load_images(pred_folder, true_folder, image_list):
#     pred_images, true_images = {}, {}
#     for image_name in tqdm(image_list, desc="Loading images"):
#         pred_path = os.path.join(pred_folder, image_name)
#         true_path = os.path.join(true_folder, image_name)
        
#         pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
#         true = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        
#         pred = cv2.resize(pred, (512, 512))  # 降低分辨率以加速
#         true = cv2.resize(true, (512, 512))

        
#         pred[pred > 0] = 1  # 将所有大于0的像素值设置为1
#         true[true > 0] = 1
#         # unique_classes_1, counts_1 = np.unique(pred, return_counts=True)

#         # unique_classes_2, counts_2 = np.unique(true, return_counts=True)
#         # print(unique_classes_1, counts_1)
#         # print(unique_classes_2, counts_2)

#         pred_images[image_name] = pred
#         true_images[image_name] = true
    
#     return pred_images, true_images

# # 计算单张图片的指标
# def process_image(image_name, pred_images, true_images):
#     pred = pred_images[image_name]
#     true = true_images[image_name]
    
#     dice = dice_coefficient(pred, true)
#     iou_value = iou(pred, true)
#     hd = hausdorff_distance(pred, true)
#     assd_value = assd(pred, true)
    
#     return [image_name, dice, iou_value, hd, assd_value]

# # 计算所有图片的指标
# def calculate_metrics(pred_folder, true_folder):
#     test_images = [f for f in os.listdir(pred_folder) if f.startswith('test') and f.endswith('.png')]
    
#     # 预加载图像到内存
#     pred_images, true_images = load_images(pred_folder, true_folder, test_images)

#     # 使用 ThreadPoolExecutor 加速
#     results = []
#     with ThreadPoolExecutor() as executor:
#         futures = list(tqdm(executor.map(lambda img: process_image(img, pred_images, true_images), test_images), 
#                             total=len(test_images), desc="Computing metrics"))
#         results.extend(futures)
    
#     return results

# # 计算平均值
# def calculate_average(metrics):
#     dice_avg = np.mean([m[1] for m in metrics])
#     iou_avg = np.mean([m[2] for m in metrics])
#     hd_avg = np.mean([m[3] for m in metrics])
#     assd_avg = np.mean([m[4] for m in metrics])
    
#     return ['Average', dice_avg, iou_avg, hd_avg, assd_avg]

# # 写入 CSV
# def write_to_csv(results, output_file):
#     avg = calculate_average(results)
#     with open(output_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Image', 'Dice', 'IoU', '95HD', 'ASSD'])
#         writer.writerows(results)
#         writer.writerow(avg)

# # 主函数
# def main(pred_folder, true_folder, output_file):
#     metrics = calculate_metrics(pred_folder, true_folder)
#     write_to_csv(metrics, output_file)

# # 运行
# true_folder = 'best_validation_gt_CRAG_sam/epoch_32'  # 预测 mask 文件夹路径
# pred_folder = 'best_validation_masks_CRAG_sam/epoch_32'  # 真实 mask 文件夹路径
# output_file = 'metrics_output.csv'  # 输出 CSV 文件路径

# main(pred_folder, true_folder, output_file)
import os
import numpy as np
import cv2
import csv
from medpy.metric.binary import dc,jc,hd95,asd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 使用 medpy 计算 Dice 系数
def dice_coefficient(pred, true):
    return dc(pred, true)

# 使用 medpy 计算 IoU
def iou(pred, true):
    return jc(pred, true)

# 使用 medpy 计算 Hausdorff 距离
def hausdorff_distance(pred, true):
    return hd95(pred, true)

# 使用 medpy 计算 ASSD
def assd(pred, true):
    return asd(pred, true)

# 预加载图像到内存
def load_images(pred_folder, true_folder, image_list):
    pred_images, true_images = {}, {}
    for image_name in tqdm(image_list, desc="Loading images"):
        pred_path = os.path.join(pred_folder, image_name)
        true_path = os.path.join(true_folder, image_name)
        
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        true = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        
        # 将 pred 中大于0的所有值设置为1
        pred[pred < 1] = 0
        true[true < 1] = 0

        pred[pred >= 1] = 1
        true[true >= 1] = 1

        pred = cv2.resize(pred, (512, 512))  
        true = cv2.resize(true, (512, 512))

        pred_images[image_name] = pred
        true_images[image_name] = true
    
    return pred_images, true_images

# 计算单张图片的指标
def process_image(image_name, pred_images, true_images):
    pred = pred_images[image_name]
    true = true_images[image_name]
    
    # print(pred.shape,true.shape)
    dice = dice_coefficient(pred, true)
    iou_value = iou(pred, true)
    hd = hausdorff_distance(pred, true)
    assd_value = assd(pred, true)
    
    return [image_name, dice, iou_value, hd, assd_value]

# 计算所有图片的指标
def calculate_metrics(pred_folder, true_folder):
    test_images = [f for f in os.listdir(pred_folder) if f.startswith('test') and f.endswith('.png')]
    
    # 预加载图像到内存
    pred_images, true_images = load_images(pred_folder, true_folder, test_images)

    # 计算每张图像的指标
    results = []
    for image_name in tqdm(test_images, desc="Computing metrics"):
        results.append(process_image(image_name, pred_images, true_images))
    
    return results

# 计算平均值
def calculate_average(metrics):
    dice_avg = np.mean([m[1] for m in metrics])
    iou_avg = np.mean([m[2] for m in metrics])
    hd_avg = np.mean([m[3] for m in metrics])
    assd_avg = np.mean([m[4] for m in metrics])
    
    return ['Average', dice_avg, iou_avg, hd_avg, assd_avg]

# 写入 CSV
def write_to_csv(results, output_file):
    avg = calculate_average(results)
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Dice', 'IoU', '95HD', 'ASSD'])
        writer.writerows(results)
        writer.writerow(avg)

# 主函数
def main(pred_folder, true_folder, output_file):
    metrics = calculate_metrics(pred_folder, true_folder)
    write_to_csv(metrics, output_file)

# 运行
pred_folder = 'best_validation_masks_CRAG_sam_noextra_20250221_102339'  # 预测 mask 文件夹路径
true_folder = 'best_validation_gt_CRAG_sam'  # 真实 mask 文件夹路径
output_file = 'metrics_output.csv'  # 输出 CSV 文件路径

main(pred_folder, true_folder, output_file)