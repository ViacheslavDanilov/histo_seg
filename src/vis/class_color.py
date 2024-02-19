import cv2
import numpy as np

from src.data.utils import CLASS_COLOR

if __name__ == '__main__':
    for class_name in CLASS_COLOR:
        img = np.zeros((224, 224, 3))
        img[:, :] = CLASS_COLOR[class_name]
        cv2.imwrite(f'{class_name}.png', img)
