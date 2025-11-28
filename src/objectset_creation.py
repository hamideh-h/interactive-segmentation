import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local, threshold_yen, gaussian
from skimage.measure import label, regionprops
import pickle
import math
from Object import Object

def load_data(file_name):
    """
    Load data from a file.

    Args:
    - file_name (str): Name of the file to load data from.

    Returns:
    - Data loaded from the file.
    """
    with open(file_name, "rb") as f:
        x = pickle.load(f)
    return x

def save_data(data, file_name):
    """
    Save data to a file.

    Args:
    - data: Data to be saved.
    - file_name (str): Name of the file to save data to.
    """
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def read_raw_files():
    """
    Read raw files, apply filters and thresholding, and extract objects.
    You might need to change this function completely based on your data

    Returns:
    - list_of_objects (list): List of extracted object properties.
    """
    # Load input files
    input_files = np.load('Workspace_data/input.npy', allow_pickle=True)

    # Parameters for thresholding
    threshold_std = 2.5
    isPlot = False  # Set to True if you want to plot images
    isPlotDifferentFilter = False  # Set to True if you want to plot images after applying different filters
    suggested_sigma_for_basiyan_filter = 7

    # Initialize list to store object information
    list_of_objects = []

    # Loop through input files
    for i, image in enumerate(input_files):
        image = np.array(image)  # Read the ser5p channel
        block_size = 8 * (image.shape[0] // 16) + 1

        # Apply different filters and thresholding
        blur_img = check_different_filter(threshold_std, block_size, image, isPlotDifferentFilter,
                                          suggested_sigma_for_basiyan_filter)
        local_thresh = threshold_local(blur_img, block_size, method='gaussian')
        mask = blur_img > threshold_std * local_thresh

        # Plot images if isPlot is True
        if isPlot:
            fig, axes = plt.subplots(2)
            axes[0].imshow(image, cmap='gray', vmin=np.percentile(image, 0.01), vmax=np.percentile(image, 99.99))
            axes[1].imshow(mask)

        # Label the mask and extract objects
        label_img = label(mask, connectivity=mask.ndim)
        list_of_objects.extend(extract_the_objects(label_img, image, blur_img))

    return list_of_objects

def check_different_filter(threshold_std, block_size, raw_image, isPlotDifferentFilter, suggested_sigma):
    """
    Apply different filters and thresholding techniques to the raw image.

    Args:
    - threshold_std (float): Standard deviation threshold for filtering.
    - block_size (int): Size of the block used for local thresholding.
    - raw_image (ndarray): The input raw image.
    - isPlotDifferentFilter (bool): Flag to indicate whether to plot the results of different filters.
    - suggested_sigma (int): Suggested sigma value for Gaussian filter.

    Returns:
    - img4 (ndarray): Image after applying Gaussian filter.
    """
    if isPlotDifferentFilter:
        # Plot images after applying different filters
        fig, axes = plt.subplots(3, 6)
        [axi.set_axis_off() for axi in axes.ravel()]
        plt.gray()

        # Apply Yen's thresholding
        axes[0][0].imshow(raw_image)
        axes[0][0].set_title('Filter 0')
        thresh = threshold_yen(raw_image)
        yen_binary = raw_image > thresh
        axes[1][0].imshow(yen_binary)

        # Apply Gaussian filter with sigma values 6, 8, 10, 12, and 14
        for i, sigma in enumerate([6, 8, 10, 12, 14], start=1):
            img = gaussian(raw_image, sigma=sigma)
            axes[0][i].imshow(img)
            local_thresh = threshold_local(img, block_size)
            binary_local = img > threshold_std * local_thresh
            axes[1][i].imshow(binary_local)
            axes[0][i].set_title('Filter ' + str(sigma))

        plt.show()

    # Apply Gaussian filter with suggested sigma value
    output_img = gaussian(raw_image, sigma=suggested_sigma)
    return output_img


def extract_the_objects(label_img, img, blur_img):
    """
    Extracting object properties from the
    labeled image and image properties.

    Args:
    - label_img (ndarray): Labeled image with segmented objects.
    - img (ndarray): Original image.
    - blur_img (ndarray): Blurred image.

    Returns:
    - list_of_objects (list): List of extracted object properties.
    """
    props = regionprops(label_img, img)

    # Filter out small objects and objects with small bounding boxes
    regionPropertiesValid = [x for x in props if
                             (x.area > 8 and x.bbox[2] - x.bbox[0] > 3 and x.bbox[3] - x.bbox[1] > 3)]

    nn = len(regionPropertiesValid)
    list_of_objects = []
    r = 30

    # Plot grayscale images
    plt.gray()

    # Loop through valid region properties
    for iter in range(nn):
        center = regionPropertiesValid[iter].centroid
        center = [math.floor(x) for x in center]
        area = regionPropertiesValid[iter].filled_area

        bbox = regionPropertiesValid[iter].bbox
        mean_intensity = regionPropertiesValid[iter].mean_intensity

        # Define bounding box dimensions
        low_dim0 = 0 if (center[0] - r < 0) else center[0] - r
        high_dim0 = img.shape[0] if (center[0] + r >= img.shape[0]) else center[0] + r
        low_dim1  = 0 if (center[1] - r < 0) else center[1] - r
        high_dim1 = img.shape[1] if (center[1] + r >= img.shape[1]) else center[1] + r

        # Continue if bounding box dimensions are too small
        if (high_dim0 - low_dim0 < (2*r)) or (high_dim1 - low_dim1 < (2*r)):
            continue

        # Extract mask and images within bounding box
        mask = label_img[low_dim0:high_dim0, low_dim1: high_dim1]
        bg_stack = img[low_dim0:high_dim0, low_dim1: high_dim1]
        blury = blur_img[low_dim0:high_dim0, low_dim1: high_dim1]
        solidity = regionPropertiesValid[iter].solidity
        elongation = regionPropertiesValid[iter].inertia_tensor_eigvals[0] / regionPropertiesValid[iter].inertia_tensor_eigvals[1]

        # Append object properties to the list
        list_of_objects.append(Object(bg_stack, mask, mean_intensity, area, center, elongation, solidity, blury))

    return list_of_objects


def interesting_objects():
    index= ["""add the index of objects you want to segment"""]
    return index

