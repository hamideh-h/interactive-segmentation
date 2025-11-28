import numpy as np
import cv2
from skimage.filters import threshold_otsu


class Object:
    def __init__(self, img, mask, intensity, area, center, elongation, solidity, blury):
        """
        Initialize an object with its properties.

        Args:
        - img (ndarray): Image of the object.
        - mask (ndarray): Mask of the object.
        - intensity (float): Mean intensity of the object.
        - area (int): Area of the object.
        - center (tuple): Center coordinates of the object.
        - elongation (float): Elongation of the object.
        - solidity (float): Solidity of the object.
        - blury (ndarray): Blurred image of the object.
        """
        self.image = img
        self.mask = mask
        self.intensity = intensity
        self.area = area
        self.center = center
        self.label = 0  # Label of the object
        self.dilation = np.zeros((5, self.mask.shape[0], self.mask.shape[1]), dtype=bool)  # Dilation of the object
        self.elongation = elongation
        self.solidity = solidity
        self.blury = blury
        self.degree_to_each_cluster = []  # Degree to each cluster
        self.features = []  # Features of the object
        self.num_of_neighbours = 0  # Number of neighbors
        self.distance_to_subspace_mean = 0  # Distance to subspace mean
        self.segmentation_algorithm = 1  # Segmentation algorithm
        self.correction_coef = 1  # Correction coefficient

        kernel_size = 3
        for i in range(5):
            # Dilate the mask progressively
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            tmp = cv2.dilate(self.mask.astype(np.uint8), kernel, iterations=i)
            thresh = threshold_otsu(tmp)
            self.dilation[i] = tmp > thresh
