import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.measure import label, regionprops


def extract_hog_features(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    features = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def extract_glcm_features(img, distances=(1, 3, 5), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    glcm = graycomatrix(img, distances=distances, angles=angles, symmetric=True, normed=True)

    props = []
    for prop_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        prop_val = graycoprops(glcm, prop_name)
        props.append(prop_val.flatten())

    return np.concatenate(props)


def cloud_coverage(cloud_mask):
    return float(np.sum(cloud_mask > 0)) / float(cloud_mask.size)


def cloud_type_distribution(cloud_class_mask, num_classes=11):
    hist, _ = np.histogram(cloud_class_mask, bins=num_classes, range=(0, num_classes))
    return hist.astype(np.float32) / float(cloud_class_mask.size)


def cloud_cluster_stats(cloud_mask):
    labeled = label(cloud_mask > 0)
    if labeled.max() == 0:
        return np.array([0, 0, 0, 0])

    props = regionprops(labeled)
    areas = [p.area for p in props]
    if len(areas) == 0:
        return np.array([0, 0, 0, 0])

    return np.array([
        np.mean(areas),
        np.std(areas) if len(areas) > 1 else 0,
        np.max(areas),
        len(areas),
    ])


def extract_all_image_features(img, cloud_class_mask=None):
    features = {}

    features["hog"] = extract_hog_features(img)

    features["glcm"] = extract_glcm_features(img)

    if cloud_class_mask is not None:
        features["cloud_coverage"] = np.array([cloud_coverage(cloud_class_mask)])
        features["cloud_type_dist"] = cloud_type_distribution(cloud_class_mask)
        features["cloud_cluster"] = cloud_cluster_stats(cloud_class_mask)
    else:
        mask = cloud_segmentation_simple(img)
        features["cloud_coverage"] = np.array([cloud_coverage(mask)])
        features["cloud_type_dist"] = np.zeros(11)
        features["cloud_cluster"] = cloud_cluster_stats(mask)

    return np.concatenate([v for v in features.values()])


def cloud_segmentation_simple(img, threshold=None):
    if threshold is None:
        threshold = 128
    return (img > threshold).astype(np.uint8)
