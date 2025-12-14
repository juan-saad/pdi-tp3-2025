from collections import deque
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


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


def filter_components_by_area(img, min_area=400, max_area=600):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

    frames_with_five_objects = {}

    for i in range(1, num_labels):
        area_componente = stats[i, cv2.CC_STAT_AREA]

        if min_area <= area_componente <= max_area:
            # Obtener coordenadas del bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            frames_with_five_objects[i] = (x, y, w, h, area_componente)

    return frames_with_five_objects


def check_stability(history, area_tolerance=10, required_frames=3):
    """
    Verifica si los dados se han detenido comparando N frames consecutivos.
    Condiciones:
    1. 5 dados detectados en los N frames.
    2. Mismos pips en cada dado (ordenados por posición X).
    3. Área similar (con un margen de tolerancia) en toda la ventana.
    """
    if len(history) != required_frames:
        return False

    if any(len(frame_dice) != 5 for frame_dice in history):
        return False

    ordered = [sorted(frame_dice, key=lambda d: d["x"]) for frame_dice in history]

    for i in range(5):
        pips_series = [ordered[t][i]["pips"] for t in range(required_frames)]
        if len(set(pips_series)) <= 0:
            return False

        area_series = [ordered[t][i]["area"] for t in range(required_frames)]
        if max(area_series) - min(area_series) > area_tolerance:
            return False

    return True


def process_frame(frame):
    """Procesa un frame y devuelve (frame_resized, detecciones_de_dados)."""
    frame_cortado = crop_image_by_percentage(frame)
    masked_bgr = filter_background_by_hue(frame_cortado)

    img_gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    frames_with_five_objects = filter_components_by_area(img_thresh)
    current_frame_dice = []

    if len(frames_with_five_objects) != 5:
        return current_frame_dice

    h_orig, w_orig = frame.shape[:2]
    offset_x = int(w_orig * 5 / 100)
    offset_y = int(h_orig * 0 / 100)

    for label_id, (x, y, w, h, area) in frames_with_five_objects.items():
        abs_x = x + offset_x
        abs_y = y + offset_y

        cropped_img = frame[abs_y : abs_y + h, abs_x : abs_x + w]
        img_hsv_crop = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])

        # Generamos una mascara binaria para los colores dentro del rango definido
        img_thresh_crop = cv2.inRange(img_hsv_crop, lower_white, upper_white)

        num_labels_crop, _, stats_crop, _ = cv2.connectedComponentsWithStats(
            img_thresh_crop, connectivity=8
        )

        pips = 0
        for i in range(1, num_labels_crop):
            area_comp = stats_crop[i, cv2.CC_STAT_AREA]
            if 3 <= area_comp <= 22:
                pips += 1

        current_frame_dice.append(
            {
                "label_id": label_id,
                "x": abs_x,
                "y": abs_y,
                "w": w,
                "h": h,
                "area": area,
                "pips": pips,
            }
        )

    return current_frame_dice


def draw_detections(img, dice):
    """Dibuja boxes y etiqueta (D# y pips) sobre una copia de img."""
    display = img.copy()
    for idx, d in enumerate(sorted(dice, key=lambda k: k["x"]), start=1):
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(
            display,
            f"D{idx}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            display,
            f"C:{d['pips']}",
            (x, y + h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return display


def main():
    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd()

    videos_path = base_dir / "videos"
    tiradas = sorted(p for p in videos_path.glob("tirada_*.mp4"))
    if not tiradas:
        raise FileNotFoundError(f"No se encontraron videos en: {videos_path}")

    tirada = tiradas[0]
    cap = cv2.VideoCapture(str(tirada))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {tirada}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    stable_frames = 3
    window = deque(maxlen=stable_frames)
    frame_number = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, dsize=(int(width / 3), int(height / 3)))
            current_frame_dice = process_frame(frame)

            window.append(
                {
                    "frame": frame.copy(),
                    "dice": current_frame_dice,
                    "frame_number": frame_number,
                }
            )

            # Mostrar el frame del medio cuando haya 3
            if len(window) < stable_frames:
                display = window[-1]["frame"]
                cv2.imshow("Detecciones en Video", display)
                frame_number += 1
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    frame_number = 0
                    break
                continue

            prev_item, mid_item, next_item = window[0], window[1], window[2]
            stable_center = check_stability(
                [prev_item["dice"], mid_item["dice"], next_item["dice"]],
                area_tolerance=10,
                required_frames=stable_frames,
            )

            display = mid_item["frame"].copy()

            if stable_center and len(mid_item["dice"]) == 5:
                display = draw_detections(display, mid_item["dice"])

                print(
                    f"!!! DADOS DETENIDOS DETECTADOS EN FRAME {mid_item['frame_number']} !!!"
                )

                print(
                    "Frames en la ventana (prev, mid, next):",
                    [item["frame_number"] for item in window],
                )

                # Mostrar el frame con boxes antes de pausar
                cv2.imshow("Detecciones en Video", display)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    frame_number = 0
                    break

                frame_number += 1
                continue

            cv2.imshow("Detecciones en Video", display)

            frame_number += 1
            if cv2.waitKey(25) & 0xFF == ord("q"):
                frame_number = 0
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
