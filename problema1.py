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
