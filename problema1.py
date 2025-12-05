import os
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

cap.release()  # Libera el objeto 'cap', cerrando el archivo.
cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas.


frame_70 = frames[69]

# Mostrar el frame 70
plt.imshow(cv2.cvtColor(frame_70, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show(block=False)

# Obtener dimensiones
height, width = frame_70.shape[:2]

# Cortar basado en porcentajes de altura
y_inicio_pct = 0
y_fin_pct = 70

y_inicio = int(height * y_inicio_pct / 100)
y_fin = int(height * y_fin_pct / 100)

# Cortar: ancho completo, solo la altura especificada
frame_cortado = frame_70[y_inicio:y_fin, :]

B, G, R = cv2.split(frame_cortado)
