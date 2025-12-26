import sys
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QGroupBox, QHBoxLayout, QVBoxLayout, QSpinBox, QTabWidget, QGridLayout,
    QFrame, QMessageBox
)
    
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import cv2
import numpy as np
import subprocess

SIZE = 350

def make_image_box(size=SIZE):
    lbl = QLabel()
    lbl.setFixedSize(size, size)
    lbl.setFrameStyle(QFrame.Box | QFrame.Plain)
    lbl.setLineWidth(3)
    lbl.setAlignment(Qt.AlignCenter)
    return lbl


def draw_mask(image, mask):
    """
    image: RGB uint8 (H, W, 3)
    mask: 0/1 或 bool (H, W)
    回傳：亮綠半透明 overlay 後的 RGB uint8
    """
    masked_image = image.copy()
    mask_bool = mask.astype(bool)

    # 將 mask 區域塗成亮綠色
    masked_image[mask_bool] = (0, 255, 0)

    # blend: 原圖 30% + 綠色 70%
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

def draw_predict_mask(base_img, gt_mask, pred_mask):
    """
    base_img: RGB uint8 (H, W, 3)
    gt_mask: 0/1 GT mask
    pred_mask: 0/1 預測 mask（畫紅線）
    """
    # Step 1: 先畫綠色透明 GT mask
    overlay = draw_mask(base_img, gt_mask)

    # Step 2: 再畫紅線框（thickness=1）
    mask_u8 = (pred_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (0, 0, 255), thickness=1)

    # 回到 RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def predict_mask(t1_img, t2_img, dist_limit=60, min_mn_area=5):
    """
    Advanced traditional-image-processing predictor returning three binary masks:
      -> pred_ct, pred_ft, pred_mn (each dtype uint8, values 0/1)

    Strategy summary (implements user-specified improvements):
      - Preprocessing: CLAHE (no global equalize), denoise with median + gaussian
      - FT: use inverted T1 + CLAHE + Otsu + opening; relax area/circularity lower bounds
      - MN: use morphological top-hat on T2 to highlight small bright blobs; spatially
            constrain MN candidates to be near FT centroids (within 100 px)
      - CT: convex hull of FT+MN points, then closing and restrict to ROI

    Inputs:
      t1_img, t2_img: grayscale uint8 numpy arrays (H, W)
    Returns:
      pred_ct, pred_ft, pred_mn: each is uint8 binary mask (0 or 1)
    """

    # --- input validation ---
    if t1_img is None or t2_img is None:
        raise ValueError('t1_img and t2_img must be provided')
    if t1_img.ndim != 2 or t2_img.ndim != 2:
        raise ValueError('inputs must be 2D grayscale images')

    H, W = t1_img.shape
    AREA_IMG = H * W

    # ---------------- Preprocessing ----------------
    # Median filter to remove impulse noise (small speckles)
    t1_med = cv2.medianBlur(t1_img, 5)
    t2_med = cv2.medianBlur(t2_img, 5)

    # CLAHE for both T1 and T2 (local contrast enhancement, avoids global equalize artifacts)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    t1_clahe = clahe.apply(t1_med)
    t2_clahe = clahe.apply(t2_med)

    # Slight Gaussian smoothing to stabilize thresholding
    t1_proc = cv2.GaussianBlur(t1_clahe, (3, 3), 0)
    t2_proc = cv2.GaussianBlur(t2_clahe, (3, 3), 0)

    # ---------------- ROI extraction to limit search space ----------------
    # Use a conservative Otsu on T1 to get wrist area; then take largest CC
    _, otsu = cv2.threshold(t1_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    roi_mask = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_close)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = np.zeros_like(roi_mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi, [c], -1, 255, thickness=-1)
    roi_bin = (roi > 0).astype(np.uint8)

    def _mask_in_roi(bin_img):
        return cv2.bitwise_and(bin_img, bin_img, mask=(roi_bin * 255))

    # ---------------- FT detection (stable: inverted T1 + CLAHE + Otsu + filtering) ----------------
    # Invert T1 so dark tendons become bright; then Otsu + morphological cleaning
    t1_inv = cv2.bitwise_not(t1_proc)
    _, th_inv = cv2.threshold(t1_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = _mask_in_roi(th_inv)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th_open = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel_open)
    th_clean = cv2.morphologyEx(th_open, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # Strict filtering as requested: Area > 15, Circularity > 0.3
    ft_mask = np.zeros_like(th_clean)
    ft_contours, _ = cv2.findContours(th_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ft_centroids = []
    for c in ft_contours:
        area = cv2.contourArea(c)
        if area < 15 or area > AREA_IMG * 0.06:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perim * perim)
        if circularity < 0.3:
            continue
        cv2.drawContours(ft_mask, [c], -1, 255, -1)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            ft_centroids.append((cx, cy))

    ft_mask = (ft_mask > 0).astype(np.uint8)

    # create ft_combined_mask (filled FT regions) for anatomical reference
    ft_combined_mask = (ft_mask > 0).astype(np.uint8) * 255
    ft_combined_mask = cv2.morphologyEx(ft_combined_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))

    # ---------------- MN detection (LoG blob detection + distance transform) ----------------
    # Use Laplacian of Gaussian to highlight small bright blobs on T2
    # Gaussian blur (sigma=2) then Laplacian
    t2_blur = cv2.GaussianBlur(t2_proc, (0, 0), 2)
    log = cv2.Laplacian(t2_blur, cv2.CV_32F, ksize=3)
    # For bright blobs, -log will have positive peaks; take negative
    log_inv = -log
    # normalize to 0-255
    log_norm = cv2.normalize(log_inv, None, 0, 255, cv2.NORM_MINMAX)
    log_u8 = log_norm.astype(np.uint8)

    # threshold with Otsu to get blob candidates
    _, mn_cand = cv2.threshold(log_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mn_cand = _mask_in_roi(mn_cand)
    mn_cand = cv2.morphologyEx(mn_cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

    # compute distance transform from FT regions: we want distance to nearest FT
    # distanceTransform computes distance to zero pixels, so invert ft_combined_mask
    if np.any(ft_combined_mask > 0):
        inv_ft = (ft_combined_mask == 0).astype(np.uint8) * 255
        dist_map = cv2.distanceTransform(inv_ft, cv2.DIST_L2, 5)
    else:
        # if no FT found, create large-distance map (so some candidates may pass)
        dist_map = np.full((H, W), np.max([H, W]), dtype=np.float32)

    # Filter candidates by area and distance to nearest FT contour
    mn_mask = np.zeros_like(mn_cand)
    mn_contours, _ = cv2.findContours(mn_cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in mn_contours:
        area = cv2.contourArea(c)
        if area < min_mn_area or area > AREA_IMG * 0.02:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # distance in pixels to nearest FT (from distance transform)
        d = float(dist_map[cy, cx])
        if d <= dist_limit:
            cv2.drawContours(mn_mask, [c], -1, 255, -1)

    mn_mask = (mn_mask > 0).astype(np.uint8)

    # ---------------- CT construction (convex hull of FT + MN points) ----------------
    # Combine all FT and MN contour points
    combined_pts = []
    for c in ft_contours:
        for p in c.reshape(-1, 2):
            combined_pts.append(p)
    mn_contours2, _ = cv2.findContours(mn_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in mn_contours2:
        for p in c.reshape(-1, 2):
            combined_pts.append(p)

    ct_mask = np.zeros((H, W), dtype=np.uint8)
    if len(combined_pts) > 0:
        all_pts = np.array(combined_pts).reshape(-1, 2)
        hull = cv2.convexHull(all_pts)
        cv2.drawContours(ct_mask, [hull], -1, 255, -1)
        # refine with closing to fill small gaps
        ct_mask = cv2.morphologyEx(ct_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

        # Adaptive dilation: if hull area is small, dilate more aggressively
        hull_area = cv2.contourArea(hull)
        if hull_area < 3000:
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            dil_iters = 3
        else:
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            dil_iters = 2
        ct_mask = cv2.dilate(ct_mask, dil_kernel, iterations=dil_iters)

        # ensure expansion does not go beyond wrist ROI
        ct_mask = cv2.bitwise_and(ct_mask, ct_mask, mask=(roi_bin * 255))

    # Finalize masks as binary 0/1 uint8
    pred_ft = (ft_mask > 0).astype(np.uint8)
    pred_mn = (mn_mask > 0).astype(np.uint8)
    pred_ct = (ct_mask > 0).astype(np.uint8)

    return pred_ct, pred_ft, pred_mn


def dice_coef(gt, pred):
    """
    gt, pred: 0/1 或 bool mask
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    s = gt.sum() + pred.sum()
    if s == 0:
        return 1.0
    return 2.0 * inter / s


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Segmentation Viewer")
        self.resize(1300, 700)

        # 影像列表
        self.t1_images = []
        self.t2_images = []

        # GT mask（已經 resize + binarize 過的 numpy）
        self.gt_masks = {
            "CT": [],
            "FT": [],
            "MN": [],
        }

        # 預測結果 mask（numpy, 0/1）
        self.pred_masks = {
            "CT": [],
            "FT": [],
            "MN": [],
        }

        # Dice per image
        self.dice_scores = {
            "CT": [],
            "FT": [],
            "MN": [],
        }

        self.idx = 0  # 當前第幾張（0-based）
        self.show_pred = False  # False: 顯示 GT mask; True: 顯示預測結果

        self.setup_ui()

    # ---------------- UI 佈局 ----------------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ========== 左邊：T1 / T2 ==========
        left_box = QGroupBox()
        left_layout = QVBoxLayout(left_box)

        left_layout.addWidget(QLabel("T1"))
        self.lbl_t1 = make_image_box()
        left_layout.addWidget(self.lbl_t1)

        left_layout.addWidget(QLabel("T2"))
        self.lbl_t2 = make_image_box()
        left_layout.addWidget(self.lbl_t2)
        left_layout.addStretch()

        # Load T1 + 左右切換
        btn_load_t1 = QPushButton("Load T1 folder")
        btn_prev = QPushButton("←")
        btn_next = QPushButton("→")

        btn_load_t1.clicked.connect(self.load_t1_folder)
        btn_prev.clicked.connect(self.prev_img)
        btn_next.clicked.connect(self.next_img)

        h1 = QHBoxLayout()
        h1.addWidget(btn_load_t1)
        h1.addStretch()
        h1.addWidget(btn_prev)
        h1.addWidget(btn_next)
        left_layout.addLayout(h1)

        # Load T2 + index
        btn_load_t2 = QPushButton("Load T2 folder")
        btn_load_t2.clicked.connect(self.load_t2_folder)

        self.spin_idx = QSpinBox()
        self.spin_idx.setMinimum(0)
        self.spin_idx.setMaximum(0)
        self.spin_idx.setValue(0)
        self.spin_idx.valueChanged.connect(self.go_index)
        
        self.lbl_filename = QLabel("")

        h2 = QHBoxLayout()
        h2.addWidget(btn_load_t2)
        h2.addStretch()
        h2.addWidget(self.spin_idx)
        h2.addWidget(self.lbl_filename)
        left_layout.addLayout(h2)

        # ========== 右邊：Tabs + CT/FT/MN ==========
        right_box = QGroupBox()
        right_layout = QVBoxLayout(right_box)

        self.tabs = QTabWidget()
        self.tab_t1 = QWidget()
        self.tab_t2 = QWidget()
        self.tabs.addTab(self.tab_t1, "T1")
        self.tabs.addTab(self.tab_t2, "T2")
        right_layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # 每個 tab 各有一組 CT/FT/MN 顯示框 + Dice label
        self.result_boxes = {"T1": {}, "T2": {}}
        self.dice_labels = {"T1": {}, "T2": {}}
        self.build_tab("T1", self.tab_t1)
        self.build_tab("T2", self.tab_t2)

        # 下方：Load mask 三顆按鈕 + Predict
        bottom_layout = QHBoxLayout()

        btn_ct_mask = QPushButton("Load CT Mask folder")
        btn_ft_mask = QPushButton("Load FT Mask folder")
        btn_mn_mask = QPushButton("Load MN Mask folder")
        btn_predict = QPushButton("Predict")
        btn_demo = QPushButton("Demo Mode")
        btn_open_overlays = QPushButton("Open Overlays")

        btn_ct_mask.clicked.connect(lambda: self.load_mask_folder("CT"))
        btn_ft_mask.clicked.connect(lambda: self.load_mask_folder("FT"))
        btn_mn_mask.clicked.connect(lambda: self.load_mask_folder("MN"))
        btn_predict.clicked.connect(self.predict_all)
        btn_demo.clicked.connect(self.run_demo_mode)
        btn_open_overlays.clicked.connect(self.open_overlays_folder)

        bottom_layout.addWidget(btn_ct_mask)
        bottom_layout.addWidget(btn_ft_mask)
        bottom_layout.addWidget(btn_mn_mask)
        bottom_layout.addWidget(btn_demo)
        bottom_layout.addWidget(btn_open_overlays)
        bottom_layout.addSpacing(40)
        bottom_layout.addWidget(btn_predict)
        bottom_layout.addStretch()

        right_layout.addLayout(bottom_layout)

        # 加到 main layout
        main_layout.addWidget(left_box, 1)
        main_layout.addWidget(right_box, 3)

    # tab 裡面 CT / FT / MN 的三個框
    def build_tab(self, tab_name: str, container: QWidget):
        layout = QVBoxLayout(container)
        grid = QGridLayout()
        grid.setHorizontalSpacing(80)

        titles = ["CT", "FT", "MN"]
        for col, key in enumerate(titles):
            lbl_title = QLabel(key)
            box = make_image_box()
            lbl_dice = QLabel("Dice coefficient:")

            self.result_boxes[tab_name][key] = box
            self.dice_labels[tab_name][key] = lbl_dice

            grid.addWidget(lbl_title, 0, col, alignment=Qt.AlignCenter)
            grid.addWidget(box, 1, col, alignment=Qt.AlignCenter)
            grid.addWidget(lbl_dice, 2, col, alignment=Qt.AlignCenter)

        layout.addLayout(grid)
        layout.addStretch()

    # ---------------- 共用：更新 spin 上限 ----------------
    def update_spin_range(self):
        lengths = [len(self.t1_images), len(self.t2_images)]
        for lst in self.gt_masks.values():
            lengths.append(len(lst))
        max_len = max(lengths) if lengths else 0
        if max_len <= 0:
            self.spin_idx.setMaximum(0)
        else:
            self.spin_idx.setMaximum(max_len - 1)

    # ---------------- 載入影像資料夾 ----------------
    def load_folder_images(self, folder): # 按照檔名排序
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]

        # 數字排序
        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        return files

    def load_t1_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select T1 Folder")
        if folder:
            self.t1_images = self.load_folder_images(folder)
            self.idx = 0
            self.spin_idx.setValue(0)
            self.update_spin_range()
            self.update_base_images()

    def load_t2_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select T2 Folder")
        if folder:
            self.t2_images = self.load_folder_images(folder)
            self.idx = 0
            self.spin_idx.setValue(0)
            self.update_spin_range()
            self.update_base_images()

    # ---------------- 載入 mask 資料夾 ----------------
    def load_mask_folder(self, kind: str):
        folder = QFileDialog.getExistingDirectory(self, f"Select {kind} Mask Folder")
        if not folder:
            return

        files = self.load_folder_images(folder)
        size = (SIZE, SIZE)
        masks = []

        for path in files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size)
            mask_bin = (img > 127).astype(np.uint8)
            masks.append(mask_bin)

        self.gt_masks[kind] = masks
        # reset 該 kind 的預測
        self.pred_masks[kind] = []
        self.dice_scores[kind] = []

        self.show_pred = False  # 新 mask 進來，先回到 GT 顯示
        self.update_spin_range()
        self.update_base_images()  # 裡面會順便呼叫 update_results()

    # ---------------- 切換 index ----------------
    def prev_img(self):
        if self.idx > 0:
            self.idx -= 1
            self.spin_idx.blockSignals(True)
            self.spin_idx.setValue(self.idx)
            self.spin_idx.blockSignals(False)
            self.update_base_images()

    def next_img(self):
        if self.idx < self.spin_idx.maximum():
            self.idx += 1
            self.spin_idx.blockSignals(True)
            self.spin_idx.setValue(self.idx)
            self.spin_idx.blockSignals(False)
            self.update_base_images()

    def go_index(self, value):
        self.idx = value
        self.update_base_images()
        
    def update_filename_label(self):
        """
        根據目前 tab + idx 顯示對應影像的檔名
        """
        tab_name = "T1" if self.tabs.currentIndex() == 0 else "T2"
        base_list = self.t1_images if tab_name == "T1" else self.t2_images

        if base_list and self.idx < len(base_list):
            path = base_list[self.idx]
            name = os.path.basename(path)
            self.lbl_filename.setText(name)
        else:
            self.lbl_filename.setText("")

    # ---------------- 更新左邊 T1/T2 display ----------------
    def update_base_images(self):
        size = SIZE

        # T1
        if self.t1_images and self.idx < len(self.t1_images):
            pix = QPixmap(self.t1_images[self.idx]).scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_t1.setPixmap(pix)
        else:
            self.lbl_t1.clear()

        # T2
        if self.t2_images and self.idx < len(self.t2_images):
            pix = QPixmap(self.t2_images[self.idx]).scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_t2.setPixmap(pix)
        else:
            self.lbl_t2.clear()

        # 右邊 CT/FT/MN 同步更新
        self.update_results()
        
        # 更新檔案名稱
        self.update_filename_label()

    def on_tab_changed(self, index):
        # 每次切換 T1 / T2，都重新依照目前 tab 更新右側顯示
        self.update_results()
        self.update_filename_label()
    
    # ---------------- 核心：更新 CT / FT / MN 顯示 ----------------
    def update_results(self):
        """
        依照目前 tab (T1 or T2)，將
        - GT mask 或 預測 mask 疊到對應的 T1/T2 影像上
        - 更新 Dice label
        """
        tab_name = "T1" if self.tabs.currentIndex() == 0 else "T2"
        base_list = self.t1_images if tab_name == "T1" else self.t2_images

        if not base_list or self.idx >= len(base_list):
            for kind in ["CT", "FT", "MN"]:
                self.result_boxes[tab_name][kind].clear()
                self.dice_labels[tab_name][kind].setText("Dice coefficient:")
            return

        base_path = base_list[self.idx]
        base_img = cv2.imread(base_path)
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        base_img = cv2.resize(base_img, (SIZE, SIZE))

        for kind in ["CT", "FT", "MN"]:
            box = self.result_boxes[tab_name][kind]
            dice_label = self.dice_labels[tab_name][kind]

            mask_to_use = None
            dice_text = "Dice coefficient:"

            if self.show_pred and self.pred_masks[kind]:
                if self.idx < len(self.pred_masks[kind]):
                    mask_to_use = self.pred_masks[kind][self.idx]
                    if self.idx < len(self.dice_scores[kind]):
                        dice_text = f"Dice coefficient: {self.dice_scores[kind][self.idx]:.3f}"
            
            elif self.gt_masks[kind]:
                if self.idx < len(self.gt_masks[kind]):
                    mask_to_use = self.gt_masks[kind][self.idx]
                    dice_text = "Dice coefficient: -"

            if mask_to_use is None:
                box.clear()
                dice_label.setText("Dice coefficient:")
                continue
            
            # 是否有預測
            if not self.show_pred:
                # 僅顯示 GT mask
                overlay_np = draw_mask(base_img, mask_to_use)

            else:
                # 同時顯示 GT + 預測紅線
                gt = self.gt_masks[kind][self.idx] if self.idx < len(self.gt_masks[kind]) else None
                pred = self.pred_masks[kind][self.idx] if self.idx < len(self.pred_masks[kind]) else None

                if gt is None or pred is None:
                    box.clear()
                    continue
                
                overlay_np = draw_predict_mask(base_img, gt, pred)

            h, w, ch = overlay_np.shape
            bytes_per_line = ch * w
            qimg = QImage(
                overlay_np.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            pix = QPixmap.fromImage(qimg)

            box.setPixmap(pix)
            dice_label.setText(dice_text)

    # ---------------- Predict：針對所有圖做預測 + Dice ----------------
    def predict_all(self):
        """
        針對每個 kind (CT/FT/MN) 的所有 GT mask：
        - 產生 pred mask 
        - 計算 Dice
        之後將 self.show_pred 設為 True，
        再呼叫 update_results() 顯示預測 overlay + Dice
        """
        size = (SIZE, SIZE)

        try:
            for kind in ["CT", "FT", "MN"]:
                gt_list = self.gt_masks[kind]
                self.pred_masks[kind] = []
                self.dice_scores[kind] = []

                for i, gt_mask in enumerate(gt_list):
                    # 讀取對應 T1 / T2
                    t1_path = self.t1_images[i]
                    t2_path = self.t2_images[i]

                    t1_img = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
                    t2_img = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)

                    t1_img = cv2.resize(t1_img, size)
                    t2_img = cv2.resize(t2_img, size)

                    # predict_mask now returns three masks: ct, ft, mn
                    pred_ct, pred_ft, pred_mn = predict_mask(t1_img, t2_img)
                    if kind == 'CT':
                        pred_bin = pred_ct
                    elif kind == 'FT':
                        pred_bin = pred_ft
                    else:
                        pred_bin = pred_mn

                    d = dice_coef(gt_mask, pred_bin)
                    self.pred_masks[kind].append(pred_bin)
                    self.dice_scores[kind].append(d)

        except NotImplementedError:
            QMessageBox.warning(
                self,
                "尚未完成作業",
                "predict_mask() 尚未實作。\n\n"
                "請依照 TODO 說明，\n"
                "使用 T1 / T2 影像設計分割方法後再執行 Predict。"
            )
            return  # 中斷

        self.show_pred = True
        self.update_results()

    # ---------------- Demo / Utilities ----------------
    def run_demo_mode(self):
        """Auto-load `MRIsample` (T1/T2 and CT/FT/MN masks) and run predict_all().

        This is for demo purposes so the GUI immediately displays results.
        """
        base = os.path.join(os.getcwd(), 'MRIsample')
        if not os.path.isdir(base):
            QMessageBox.warning(self, 'Demo Mode', f'MRIsample folder not found at {base}')
            return

        # Load T1/T2
        t1_folder = os.path.join(base, 'T1')
        t2_folder = os.path.join(base, 'T2')
        if os.path.isdir(t1_folder):
            self.t1_images = self.load_folder_images(t1_folder)
        if os.path.isdir(t2_folder):
            self.t2_images = self.load_folder_images(t2_folder)

        # Load masks
        for kind in ['CT', 'FT', 'MN']:
            mask_folder = os.path.join(base, kind)
            if os.path.isdir(mask_folder):
                # reuse load_mask_folder logic but without dialog
                files = self.load_folder_images(mask_folder)
                size = (SIZE, SIZE)
                masks = []
                for path in files:
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, size)
                    mask_bin = (img > 127).astype(np.uint8)
                    masks.append(mask_bin)
                self.gt_masks[kind] = masks

        # update UI state
        self.idx = 0
        self.spin_idx.setValue(0)
        self.update_spin_range()
        self.update_base_images()

        # Run predict_all to compute preds and dice
        self.predict_all()

    def open_overlays_folder(self):
        """Open overlays folder in OS file browser (cross-platform)."""
        path = os.path.join(os.getcwd(), 'overlays')
        if not os.path.isdir(path):
            QMessageBox.information(self, 'Open Overlays', f'No overlays directory at {path}')
            return
        try:
            if sys.platform == 'darwin':
                subprocess.call(['open', path])
            elif sys.platform == 'win32':
                os.startfile(path)
            else:
                subprocess.call(['xdg-open', path])
        except Exception as e:
            QMessageBox.warning(self, 'Open Overlays', f'Failed to open overlays: {e}')


# ---------------- main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
