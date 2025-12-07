import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def crop_image_by_percentage(
    image, y_start_pct=0, y_end_pct=70, x_start_pct=10, x_end_pct=97
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


def filter_components_by_area(img, min_area=400, max_area=500):
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
            frames_with_five_objects[i] = (x, y, w, h)

    print(
        f"Componentes con area entre {min_area} y {max_area}: {len(frames_with_five_objects)}"
    )
    return frames_with_five_objects


def show_bounding_boxes(img, stats, color=(0, 255, 0)):
    img_bboxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i in range(len(stats.keys())):
        x, y, w, h = stats[i + 1]
        cv2.rectangle(img_bboxes, (x, y), (x + w, y + h), color, 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_bboxes, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Boxes de Componentes Conectados")
    plt.axis("off")
    plt.show(block=False)


def main():
    try:
        BASE_DIR = Path(__file__).parent
    except NameError:
        BASE_DIR = Path.cwd()

    videos_path = BASE_DIR / "videos"
    tiradas = sorted(p for p in videos_path.glob("tirada_*.mp4"))


if __name__ == "__main__":
    main()


try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

videos_path = BASE_DIR / "videos"
tiradas = sorted(p for p in videos_path.glob("tirada_*.mp4"))
frames_path = BASE_DIR / "frames"

tirada = tiradas[0]

# Crear carpeta para este video (sin extensión)
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

        # frames.append(frame)

        # Cortar: ancho completo, solo la altura especificada
        frame_cortado = crop_image_by_percentage(frame)
        img_hsv = cv2.cvtColor(frame_cortado, cv2.COLOR_BGR2HSV)

        masked_bgr = filter_background_by_hue(frame_cortado)

        img_gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img_thresh, connectivity=8
        )

        frames_with_five_objects = filter_components_by_area(img_thresh)

        if len(frames_with_five_objects) == 5:
            # 1. Calcular el offset del recorte (basado en los defaults de tu función crop_image_by_percentage)
            # x_start_pct=10, y_start_pct=0
            h_orig, w_orig = frame.shape[:2]
            offset_x = int(w_orig * 10 / 100)
            offset_y = int(h_orig * 0 / 100)

            # 2. Iterar sobre los objetos detectados
            for label_id, (x, y, w, h) in frames_with_five_objects.items():
                # Ajustar coordenadas al frame original sumando el offset
                abs_x = x + offset_x
                abs_y = y + offset_y

                # Dibujar rectángulo en el 'frame' original
                cv2.rectangle(
                    frame, (abs_x, abs_y), (abs_x + w, abs_y + h), (0, 255, 0), 1
                )
                # Escribir el ID encima del rectángulo
                cv2.putText(
                    frame,
                    f"ID:{label_id}",
                    (abs_x, abs_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                
        cv2.imshow("Detecciones en Video", frame)

        frame_number += 1
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cap.release()
cv2.destroyAllWindows()
