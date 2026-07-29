import cv2
import numpy as np


def load_image(path, grayscale=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def resize_image(img, target_size=(128, 128)):
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def denoise_gaussian(img, kernel_size=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(img, kernel_size, sigma)


def denoise_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def histogram_equalization(img):
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return cv2.equalizeHist(img)


def clahe_equalization(img, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return clahe.apply(img)


def cloud_segmentation(img, cloud_class_mask=None, threshold=None):
    if cloud_class_mask is not None:
        cloud_pixels = cloud_class_mask > 0
        return cloud_pixels.astype(np.uint8) * 255

    if threshold is None:
        threshold = 128
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary


def preprocess_pipeline(img_path, target_size=(128, 128), use_clahe=True):
    img = load_image(img_path)
    img = resize_image(img, target_size)
    img = denoise_gaussian(img)
    if use_clahe:
        img = clahe_equalization(img)
    else:
        img = histogram_equalization(img)
    return img
