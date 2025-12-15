## ESTE ARCHIVO LO USAMOS PARA HACER PRUEBAS RÁPIDAS DURANTE EL DESARROLLO. NO ES PARTE DEL TP FINAL.


import os
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

videos_path = BASE_DIR / "videos"
tiradas = sorted(p for p in videos_path.glob("tirada_*.mp4"))
frames_path = BASE_DIR / "frames"

# VIDEO 1 - FRAME 67
# VIDEO 2 - FRAME 59
# VIDEO 3 - FRAME 64
# VIDEO 4 - FRAME 47

tirada = tiradas[0]
frame_nbr = 69

video_frames_path = frames_path / tirada.stem
video_frames_path.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(tirada))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_number = 0
frames = []

while cap.isOpened():

    ret, frame = cap.read()

    if ret == True:

        frame = cv2.resize(frame, dsize=(int(width / 3), int(height / 3)))

        # cv2.imshow("Frame", frame)

        # cv2.imwrite(os.path.join(video_frames_path, f"frame_{frame_number}.jpg"), frame)
        frames.append(frame)

        frame_number += 1
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

### ============ INICIO =========== ###


def crop_image_by_percentage(
    image, y_start_pct=0, y_end_pct=70, x_start_pct=5, x_end_pct=97
):
    height, width = image.shape[:2]

    y_start = int(height * y_start_pct / 100)
    y_end = int(height * y_end_pct / 100)
    x_start = int(width * x_start_pct / 100)
    x_end = int(width * x_end_pct / 100)

    cropped_image = image[y_start:y_end, x_start:x_end]
    return cropped_image


def filter_background_by_hue(image, h_min=30, h_max=90, show_plot=False):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([h_min, 0, 0])
    upper_bound = np.array([h_max, 255, 255])
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    mask_neg = cv2.bitwise_not(mask)

    filtered_image = cv2.bitwise_and(image, image, mask=mask_neg)

    if show_plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Imagen original")
        ax[0].axis("off")

        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title(f"Mascara H entre {h_min} y {h_max}")
        ax[1].axis("off")

        ax[2].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        ax[2].set_title("Imagen filtrada")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show(block=False)

    return filtered_image


def filter_components_by_area(img, min_area=400, max_area=600, draw_boxes=False, image_for_boxes=None):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

    frames_with_five_objects = {}

    if draw_boxes and image_for_boxes is not None:
        img_boxes = image_for_boxes.copy()
    else:
        img_boxes = None

    for i in range(1, num_labels):
        area_componente = stats[i, cv2.CC_STAT_AREA]

        if min_area <= area_componente <= max_area:
            # Obtener coordenadas del bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            frames_with_five_objects[i] = (x, y, w, h, area_componente)

            if draw_boxes and img_boxes is not None:
                cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if draw_boxes and img_boxes is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
        plt.title("Bounding boxes de componentes detectados")
        plt.axis("off")
        plt.show()

    return frames_with_five_objects


frame = frames[frame_nbr]

frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10, 4))

# Mostrar imagen en escala de grises
plt.subplot(1, 2, 1)
plt.imshow(frame_gray, cmap='gray')
plt.title('Frame en escala de grises')
plt.axis('off')

# Mostrar histograma de intensidades
plt.subplot(1, 2, 2)
plt.hist(frame_gray.ravel(), bins=256, range=(0, 256), color='black')
plt.title('Histograma de intensidades')
plt.xlabel('Intensidad')
plt.ylabel('Cantidad de píxeles')

plt.tight_layout()
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
img_top_hat = cv2.morphologyEx(frame_gray, cv2.MORPH_TOPHAT, kernel)

_, img_thresh = cv2.threshold(img_top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

num_labels_hat, labels_hat, stats_hat, _ = cv2.connectedComponentsWithStats(
    img_top_hat, connectivity=8
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

ax1.imshow(img_top_hat, cmap='gray')
ax1.set_title('Imagen Top Hat')
ax1.axis('off')

ax2.imshow(img_thresh, cmap='gray')
ax2.set_title('Imagen Umbralizada')
ax2.axis('off')

plt.tight_layout()
plt.show()

frame_cortado = crop_image_by_percentage(frame)
masked_bgr = filter_background_by_hue(frame_cortado, show_plot=True)

img_gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)

_, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

frames_with_five_objects = filter_components_by_area(img_thresh, draw_boxes=True, image_for_boxes=frame_cortado)

h_orig, w_orig = frame.shape[:2]
offset_x = int(w_orig * 5 / 100)
offset_y = int(h_orig * 0 / 100)

dice_vis = []
dice_titles = []

dice_crops = []
dice_masks = []
dice_titles = []

for label_id, (x, y, w, h, area) in frames_with_five_objects.items():
    abs_x = x + offset_x
    abs_y = y + offset_y

    cropped_img = frame[abs_y : abs_y + h, abs_x : abs_x + w]
    img_hsv_crop = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])

    # Generamos una maskara binaria para los colores dentro del rango definido
    img_thresh_crop = cv2.inRange(img_hsv_crop, lower_white, upper_white)
    
    dice_crops.append(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    dice_masks.append(img_thresh_crop)
    dice_titles.append(f"Dado {label_id}")

    num_labels_crop, labels_crop, stats_crop, _ = cv2.connectedComponentsWithStats(
        img_thresh_crop, connectivity=8
    )
    
    pips = 0
    crop_vis = cropped_img.copy()

    for i in range(1, num_labels_crop):
        area_comp = stats_crop[i, cv2.CC_STAT_AREA]
        if 3 <= area_comp <= 22:
            pips += 1

            # Bounding box del pip (coordenadas relativas al crop)
            px = stats_crop[i, cv2.CC_STAT_LEFT]
            py = stats_crop[i, cv2.CC_STAT_TOP]
            pw = stats_crop[i, cv2.CC_STAT_WIDTH]
            ph = stats_crop[i, cv2.CC_STAT_HEIGHT]

            cv2.rectangle(crop_vis, (px, py), (px + pw, py + ph), (0, 0, 255), 1)

    print(f"Dado {label_id}: area={area}, pips={pips}")

    dice_vis.append(crop_vis)
    dice_titles.append(f"Dado {label_id} | pips={pips}")

n = len(dice_crops)
fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

for i in range(n):
    axes[0, i].imshow(dice_crops[i])
    axes[0, i].set_title(dice_titles[i])
    axes[0, i].axis('off')
    axes[1, i].imshow(dice_masks[i], cmap='gray')
    axes[1, i].set_title('Máscara pips')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show(block=False)

if len(dice_vis) > 0:
    n = len(dice_vis)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, dice_vis, dice_titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show(block=False)
