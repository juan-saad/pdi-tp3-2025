import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    try:
        BASE_DIR = Path(__file__).parent
    except NameError:
        BASE_DIR = Path.cwd()

    IMAGE_PATH = BASE_DIR / "imagenes" / "monedas.jpg"


if __name__ == "__main__":
    main()