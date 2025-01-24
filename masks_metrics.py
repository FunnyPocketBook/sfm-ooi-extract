import os
import cv2
import numpy as np
from glob import glob

def calculate_metrics(gt_path, gen_path):
    # Get list of files in gt_path
    gt_files = glob(os.path.join(gt_path, "*.png"))

    # Initialize metrics
    total_iou, total_dice, total_precision, total_recall, total_f1 = 0, 0, 0, 0, 0
    num_files = len(gt_files)

    for gt_file in gt_files:
        # Load ground truth mask
        gt_mask = cv2.imread(gt_file, cv2.IMREAD_COLOR)
        gt_mask = (gt_mask[:, :, 2] > 0).astype(np.uint8)  # Red channel as binary mask

        # Find the corresponding generated mask
        gen_file = os.path.join(gen_path, os.path.basename(gt_file))
        if not os.path.exists(gen_file):
            print(f"Warning: Generated mask not found for {gt_file}")
            num_files -= 1
            continue

        # Load generated mask
        gen_mask = cv2.imread(gen_file, cv2.IMREAD_GRAYSCALE)
        gen_mask = (gen_mask > 0).astype(np.uint8)  # Convert to binary mask

        # Calculate metrics
        intersection = np.logical_and(gt_mask, gen_mask).sum()
        union = np.logical_or(gt_mask, gen_mask).sum()
        gt_sum = gt_mask.sum()
        gen_sum = gen_mask.sum()

        # Avoid division by zero
        iou = intersection / union if union > 0 else 0
        dice = 2 * intersection / (gt_sum + gen_sum) if (gt_sum + gen_sum) > 0 else 0
        precision = intersection / gen_sum if gen_sum > 0 else 0
        recall = intersection / gt_sum if gt_sum > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Accumulate metrics
        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # Calculate averages
    if num_files > 0:
        avg_iou = total_iou / num_files
        avg_dice = total_dice / num_files
        avg_precision = total_precision / num_files
        avg_recall = total_recall / num_files
        avg_f1 = total_f1 / num_files
    else:
        avg_iou = avg_dice = avg_precision = avg_recall = avg_f1 = 0

    return {
        "IoU": avg_iou,
        "Dice Coefficient": avg_dice,
        "Precision": avg_precision,
        "Recall": avg_recall,
        "F1 Score": avg_f1,
    }

def run_metrics(object_name):
    gt_path = f"data/{object_name}/masks_gt"
    gen_path = f"output/{object_name}/exp1/mesh/mask"

    # Calculate metrics
    # Calculate metrics
    metrics = calculate_metrics(gt_path, gen_path)


    # Print metrics
    print("Average Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


    # Write metrics to a file
    output_file = f"metrics_results_{object_name}.txt"
    with open(output_file, "w") as f:
        f.write("Average Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"Metrics have been written to {output_file}")

run_metrics("kitchen")
run_metrics("bonsai")
run_metrics("garden")