"""
生成報告所需的所有視覺化圖片
- 選出最佳和最差的範例
- 生成中間處理步驟的視覺化
- 保存到 report_images/ 資料夾
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from main import predict_mask, predict_mask_ct, predict_mask_mn, dice_coef

BASE = 'MRIsample'
OUTPUT_DIR = 'report_images'

# 創建輸出資料夾
os.makedirs(OUTPUT_DIR, exist_ok=True)
for subdir in ['FT', 'CT', 'MN']:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


def visualize_ft_pipeline(t1, t2, gt, filename, output_dir):
    """視覺化 FT 分割的所有中間步驟"""
    # Step 1: CLAHE on T2
    t2_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(t2)
    
    # Step 2: Gaussian blur
    t2_blur = cv2.GaussianBlur(t2_clahe, (5, 5), 0)
    t1_blur = cv2.GaussianBlur(t1, (5, 5), 0)
    
    # Step 3: Thresholding
    _, t2_dark = cv2.threshold(t2_blur, 115, 255, cv2.THRESH_BINARY_INV)
    _, t1_dark = cv2.threshold(t1_blur, 35, 255, cv2.THRESH_BINARY_INV)
    
    # Step 4: Prior mask (if available)
    from main import _FT_PRIOR
    if _FT_PRIOR is not None and _FT_PRIOR.shape == t2.shape:
        prior_mask = (_FT_PRIOR >= 0.075).astype(np.uint8) * 255
    else:
        prior_mask = np.ones_like(t2, dtype=np.uint8) * 255
    
    # Step 5: Combine masks
    mask = cv2.bitwise_and(prior_mask, t2_dark)
    mask = cv2.bitwise_and(mask, t1_dark)
    
    # Step 6: Morphology
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_final = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, k_open)
    
    # Final prediction
    pred = predict_mask(t1, t2)
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'FT Segmentation Pipeline - {filename}', fontsize=16)
    
    axes[0, 0].imshow(t1, cmap='gray')
    axes[0, 0].set_title('Original T1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(t2, cmap='gray')
    axes[0, 1].set_title('Original T2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(t2_clahe, cmap='gray')
    axes[0, 2].set_title('T2 CLAHE')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(t2_blur, cmap='gray')
    axes[0, 3].set_title('T2 CLAHE + Blur')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(t2_dark, cmap='gray')
    axes[1, 0].set_title('T2 Dark Threshold (115)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(t1_dark, cmap='gray')
    axes[1, 1].set_title('T1 Dark Threshold (35)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(prior_mask, cmap='gray')
    axes[1, 2].set_title('Spatial Prior (≥0.075)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(mask, cmap='gray')
    axes[1, 3].set_title('Combined Mask')
    axes[1, 3].axis('off')
    
    axes[2, 0].imshow(mask_close, cmap='gray')
    axes[2, 0].set_title('After Closing (7×7)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(mask_final, cmap='gray')
    axes[2, 1].set_title('After Opening (3×3)')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(gt * 255, cmap='gray')
    axes[2, 2].set_title('Ground Truth')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(pred * 255, cmap='gray')
    axes[2, 3].set_title(f'Final Prediction\nDice: {dice_coef(gt, pred):.3f}')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_ct_pipeline(t1, t2, gt, filename, output_dir):
    """視覺化 CT 分割的所有中間步驟"""
    # Step 1: CLAHE on T2
    t2_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(t2)
    
    # Step 2: Gaussian blur
    t2_blur = cv2.GaussianBlur(t2_clahe, (5, 5), 0)
    t1_blur = cv2.GaussianBlur(t1, (5, 5), 0)
    
    # Step 3: Thresholding
    _, t2_dark = cv2.threshold(t2_blur, 135, 255, cv2.THRESH_BINARY_INV)
    _, t1_dark = cv2.threshold(t1_blur, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Step 4: Prior mask
    from main import _CT_PRIOR
    if _CT_PRIOR is not None and _CT_PRIOR.shape == t2.shape:
        prior_mask = (_CT_PRIOR >= 0.275).astype(np.uint8) * 255
    else:
        prior_mask = np.ones_like(t2, dtype=np.uint8) * 255
    
    # Step 5: Combine masks
    mask = cv2.bitwise_and(prior_mask, t2_dark)
    mask = cv2.bitwise_and(mask, t1_dark)
    
    # Step 6: Morphology
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_final = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, k_open)
    
    # Final prediction
    pred = predict_mask_ct(t1, t2)
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'CT Segmentation Pipeline - {filename}', fontsize=16)
    
    axes[0, 0].imshow(t1, cmap='gray')
    axes[0, 0].set_title('Original T1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(t2, cmap='gray')
    axes[0, 1].set_title('Original T2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(t2_clahe, cmap='gray')
    axes[0, 2].set_title('T2 CLAHE')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(t2_blur, cmap='gray')
    axes[0, 3].set_title('T2 CLAHE + Blur')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(t2_dark, cmap='gray')
    axes[1, 0].set_title('T2 Dark Threshold (135)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(t1_dark, cmap='gray')
    axes[1, 1].set_title('T1 Dark Threshold (60)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(prior_mask, cmap='gray')
    axes[1, 2].set_title('Spatial Prior (≥0.275)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(mask, cmap='gray')
    axes[1, 3].set_title('Combined Mask')
    axes[1, 3].axis('off')
    
    axes[2, 0].imshow(mask_close, cmap='gray')
    axes[2, 0].set_title('After Closing (13×13)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(mask_final, cmap='gray')
    axes[2, 1].set_title('After Opening (3×3)')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(gt * 255, cmap='gray')
    axes[2, 2].set_title('Ground Truth')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(pred * 255, cmap='gray')
    axes[2, 3].set_title(f'Final Prediction\nDice: {dice_coef(gt, pred):.3f}')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_mn_pipeline(t1, t2, gt, filename, output_dir):
    """視覺化 MN 分割的所有中間步驟"""
    # Step 1: CLAHE on T2
    t2_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(t2)
    
    # Step 2: Gaussian blur
    t2_blur = cv2.GaussianBlur(t2_clahe, (3, 3), 0)
    t1_blur = cv2.GaussianBlur(t1, (3, 3), 0)
    
    # Step 3: Thresholding
    _, t2_bright = cv2.threshold(t2_blur, 125, 255, cv2.THRESH_BINARY)
    _, t1_bright = cv2.threshold(t1_blur, 35, 255, cv2.THRESH_BINARY)
    
    # Step 4: Prior mask
    from main import _MN_PRIOR
    if _MN_PRIOR is not None and _MN_PRIOR.shape == t2.shape:
        prior_mask = (_MN_PRIOR >= 0.16).astype(np.uint8) * 255
    else:
        prior_mask = np.ones_like(t2, dtype=np.uint8) * 255
    
    # Step 5: Combine masks
    mask = cv2.bitwise_and(prior_mask, t2_bright)
    mask = cv2.bitwise_and(mask, t1_bright)
    
    # Step 6: Morphology
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    
    # Final prediction
    pred = predict_mask_mn(t1, t2)
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'MN Segmentation Pipeline - {filename}', fontsize=16)
    
    axes[0, 0].imshow(t1, cmap='gray')
    axes[0, 0].set_title('Original T1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(t2, cmap='gray')
    axes[0, 1].set_title('Original T2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(t2_clahe, cmap='gray')
    axes[0, 2].set_title('T2 CLAHE')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(t2_blur, cmap='gray')
    axes[0, 3].set_title('T2 CLAHE + Blur')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(t2_bright, cmap='gray')
    axes[1, 0].set_title('T2 Bright Threshold (125)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(t1_bright, cmap='gray')
    axes[1, 1].set_title('T1 Bright Threshold (35)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(prior_mask, cmap='gray')
    axes[1, 2].set_title('Spatial Prior (≥0.16)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(mask, cmap='gray')
    axes[1, 3].set_title('Combined Mask')
    axes[1, 3].axis('off')
    
    axes[2, 0].imshow(mask_close, cmap='gray')
    axes[2, 0].set_title('After Closing (3×3)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(pred * 255, cmap='gray')
    axes[2, 1].set_title('After Component Filter')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(gt * 255, cmap='gray')
    axes[2, 2].set_title('Ground Truth')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(pred * 255, cmap='gray')
    axes[2, 3].set_title(f'Final Prediction\nDice: {dice_coef(gt, pred):.3f}')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()


def process_task(task_name, predictor_func, visualizer_func):
    """處理每個分割任務"""
    print(f"\nProcessing {task_name}...")
    
    # 只處理 0-9.jpg
    files = sorted([f for f in os.listdir(os.path.join(BASE, task_name)) if f.endswith('.jpg')],
                   key=lambda x: int(os.path.splitext(x)[0]))
    files = [f for f in files if int(os.path.splitext(f)[0]) <= 9]
    
    results = []
    for fname in files:
        t1 = cv2.imread(os.path.join(BASE, 'T1', fname), cv2.IMREAD_GRAYSCALE)
        t2 = cv2.imread(os.path.join(BASE, 'T2', fname), cv2.IMREAD_GRAYSCALE)
        gt = (cv2.imread(os.path.join(BASE, task_name, fname), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        
        pred = predictor_func(t1, t2)
        score = dice_coef(gt, pred)
        
        results.append({
            'filename': fname,
            't1': t1,
            't2': t2,
            'gt': gt,
            'pred': pred,
            'dice': score
        })
    
    # 排序找出最佳和最差
    results_sorted = sorted(results, key=lambda x: x['dice'], reverse=True)
    
    # 生成最佳 1-2 張
    print(f"  Best results: {results_sorted[0]['filename']} (Dice: {results_sorted[0]['dice']:.3f})")
    if len(results_sorted) > 1:
        print(f"                {results_sorted[1]['filename']} (Dice: {results_sorted[1]['dice']:.3f})")
    
    for i in range(min(2, len(results_sorted))):
        r = results_sorted[i]
        visualizer_func(r['t1'], r['t2'], r['gt'], 
                       f"best_{i+1}_{r['filename']}", 
                       os.path.join(OUTPUT_DIR, task_name))
    
    # 生成最差 1-2 張
    print(f"  Worst results: {results_sorted[-1]['filename']} (Dice: {results_sorted[-1]['dice']:.3f})")
    if len(results_sorted) > 1:
        print(f"                 {results_sorted[-2]['filename']} (Dice: {results_sorted[-2]['dice']:.3f})")
    
    for i in range(min(2, len(results_sorted))):
        r = results_sorted[-(i+1)]
        visualizer_func(r['t1'], r['t2'], r['gt'], 
                       f"worst_{i+1}_{r['filename']}", 
                       os.path.join(OUTPUT_DIR, task_name))
    
    return results


if __name__ == '__main__':
    print("Generating report images...")
    
    # 處理三個任務
    ft_results = process_task('FT', predict_mask, visualize_ft_pipeline)
    ct_results = process_task('CT', predict_mask_ct, visualize_ct_pipeline)
    mn_results = process_task('MN', predict_mask_mn, visualize_mn_pipeline)
    
    # 保存統計數據供報告使用
    stats = {
        'FT': {
            'mean': np.mean([r['dice'] for r in ft_results]),
            'std': np.std([r['dice'] for r in ft_results]),
            'min': min([r['dice'] for r in ft_results]),
            'max': max([r['dice'] for r in ft_results]),
            'all': [(r['filename'], r['dice']) for r in ft_results]
        },
        'CT': {
            'mean': np.mean([r['dice'] for r in ct_results]),
            'std': np.std([r['dice'] for r in ct_results]),
            'min': min([r['dice'] for r in ct_results]),
            'max': max([r['dice'] for r in ct_results]),
            'all': [(r['filename'], r['dice']) for r in ct_results]
        },
        'MN': {
            'mean': np.mean([r['dice'] for r in mn_results]),
            'std': np.std([r['dice'] for r in mn_results]),
            'min': min([r['dice'] for r in mn_results]),
            'max': max([r['dice'] for r in mn_results]),
            'all': [(r['filename'], r['dice']) for r in mn_results]
        }
    }
    
    # 保存為 JSON
    import json
    with open(os.path.join(OUTPUT_DIR, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n✓ All images generated successfully!")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"  Statistics saved to: {OUTPUT_DIR}/statistics.json")
