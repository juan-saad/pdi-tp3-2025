import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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

        frames.append(frame)

        # cv2.imshow("Frame", frame)

        # cv2.imwrite(str(video_frames_path / f"frame_{frame_number}.jpg"), frame)

        # frame_number += 1
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


frame_70 = frames[69] # Usar frame 85 para análisis incorrecto

# Obtener dimensiones
height, width = frame_70.shape[:2]

# Cortar basado en porcentajes de altura
y_inicio_pct = 0
y_fin_pct = 70
x_inicio_pct = 10
x_fin_pct = 97

y_inicio = int(height * y_inicio_pct / 100)
y_fin = int(height * y_fin_pct / 100)
x_inicio = int(width * x_inicio_pct / 100)
x_fin = int(width * x_fin_pct / 100)

# Cortar: ancho completo, solo la altura especificada
frame_cortado = frame_70[y_inicio:y_fin, x_inicio:x_fin]
img_hsv = cv2.cvtColor(frame_cortado, cv2.COLOR_BGR2HSV)

# Mostrar sólo tonos H en el rango 30-90 (OpenCV H: 0-180)
h_min, h_max = np.array([30, 0, 0]), np.array([90, 255, 255])
mask = cv2.inRange(img_hsv, h_min, h_max)
mask_neg = cv2.bitwise_not(mask)

masked_bgr = cv2.bitwise_and(frame_cortado, frame_cortado, mask=mask_neg)

# Graficar canal H, máscara y resultado filtrado
fig2, ax2 = plt.subplots(1, 3, sharey=True, sharex=True)
ax2[0].imshow(img_hsv[:, :, 0], cmap="hsv")
ax2[0].set_title("Canal H")
ax2[0].axis("off")

ax2[1].imshow(mask_neg, cmap="gray")
ax2[1].set_title(f"Máscara {h_min}-{h_max}")
ax2[1].axis("off")

ax2[2].imshow(cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB))
ax2[2].set_title("Imagen filtrada por H")
ax2[2].axis("off")

plt.tight_layout()
plt.show(block=False)

img_gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
ax2[0].imshow(cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB))
ax2[0].set_title("Imagen filtrada por H")
ax2[0].axis("off")

ax2[1].imshow(img_gray, cmap="gray")
ax2[1].set_title("Imagen en grises")
ax2[1].axis("off")

ax2[2].imshow(img_thresh, cmap="gray")
ax2[2].set_title(f"Imagen umbralizada (Otsu)")
ax2[2].axis("off")

plt.tight_layout()
plt.show(block=False)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    img_thresh, connectivity=8
)

# # Convertir a BGR para poder dibujar puntos de color (rojo) sobre la imagen binaria
# img_centroids = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)

# # Iterar sobre los centroides (empezamos en 1 para ignorar el fondo, que es la etiqueta 0)
# for i in range(1, num_labels):
#     x, y = centroids[i]
#     # Dibujar un círculo rojo en la posición del centroide
#     # (imagen, centro, radio, color BGR, grosor -1 para relleno)
#     cv2.circle(img_centroids, (int(x), int(y)), 2, (0, 0, 255), -1)

# # Graficar el resultado
# plt.figure(figsize=(10, 8))
# plt.imshow(cv2.cvtColor(img_centroids, cv2.COLOR_BGR2RGB))
# plt.title(f"Centroides detectados: {num_labels - 1}")
# plt.axis("off")
# plt.show()

area_min = 400
area_max = 500

frames_with_five_objects = {}

# Convertir a BGR para dibujar rectángulos de color
img_bboxes = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
areas = stats[1:, cv2.CC_STAT_AREA] 

for i in range(1, num_labels):
    area_componente = stats[i, cv2.CC_STAT_AREA]
        
    if area_min <= area_componente <= area_max:
        # Obtener coordenadas del bounding box
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        frames_with_five_objects[i] = (x, y, w, h)
        
if len(frames_with_five_objects) == 5:
    for i in frames_with_five_objects:
        x, y, w, h = frames_with_five_objects[i]
        cv2.rectangle(img_bboxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
    # Mostrar la imagen resultante
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_bboxes, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Boxes filtrados por área")
    plt.axis("off")
    plt.show(block=False)