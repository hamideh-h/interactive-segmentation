import easygui  # Importing necessary modules
import matplotlib.pyplot as plt
import math
import numpy as np
from skimage.measure import label, regionprops
from skimage.filters import threshold_yen, threshold_otsu
from skimage.morphology import dilation
import cv2
import easygui
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mahotas.labeled import bwperim
from skimage.filters import threshold_yen
from skimage.filters.thresholding import _cross_entropy
from objectset_creation import save_data
from FeatureSpace_Clustering import objects_per_cluster, borderline_objects



def denoising(img, kernel_size=3):
    """
    Perform flat denoising on the input image using averaging filter.

    Parameters:
    img (numpy.ndarray): Input image.
    kernel_size (int, optional): Size of the averaging filter kernel. Default is 3.

    Returns:
    numpy.ndarray: Denoised image.
    """
    # Apply averaging filter for flat denoising
    img_denoised = cv2.blur(img, (kernel_size, kernel_size))

    return img_denoised


def optimised_threshold_based_on_crossEntropy(img):
    thresholds = np.arange(np.min(img) + 1.3, np.max(img) - 1.3, 0.1)
    entropies = [_cross_entropy(img, t) for t in thresholds]
    thresh = thresholds[np.argmin(entropies)]
    return thresh


def riddler_calvard_threshold(img):
    # Convert image to grayscale

    # Initialize threshold value
    prev_thresh = 0
    curr_thresh = np.mean(img)

    # Iterate until the threshold value stabilizes
    while abs(curr_thresh - prev_thresh) > 0.5:
        prev_thresh = curr_thresh

        # Background and foreground pixel values
        bg_pixels = img[img <= curr_thresh]
        fg_pixels = img[img > curr_thresh]

        # Calculate mean intensities
        mean_bg = np.mean(bg_pixels)
        mean_fg = np.mean(fg_pixels)

        # Update threshold value
        curr_thresh = (mean_bg + mean_fg) / 2

    # Apply final threshold
    _, binary_img = cv2.threshold(img, curr_thresh, 255, cv2.THRESH_BINARY)

    return binary_img


def run_segmentation(interesting_object, file_name, n_clusters=2, toSave=0):
    """
    Perform segmentation on the provided interested_focus.

    Parameters:
    interested_focus (list): List of objects for segmentation.
    file_name (str): Name of the file for saving data.
    n_clusters (int, optional): Number of clusters for segmentation. Default is 2.
    toSave (int, optional): Option to save data. Default is 0.

    Returns:
    tuple: Tuple containing updated_objects (list) and settings_history (list).
    """
    cluster0 = objects_per_cluster(interesting_object, 0, isPlot=0)  # Retrieve objects in cluster 0
    cluster1 = objects_per_cluster(interesting_object, 1, isPlot=0)  # Retrieve objects in cluster 1
    segmentation_settings = []  # Initialize list to store segmentation settings
    settings_history = []  # Initialize list to store segmentation history
    border_cluster = borderline_objects(interesting_object, 2, isPlot=0)  # Find borderline objects
    updated_objects = []  # Initialize list to store updated objects

    # Iterate over clusters for segmentation
    for index, cluster in enumerate([cluster0, cluster1]):
        settings_history.append([-1, -1])  # Placeholder for initial settings
        random_index = np.random.randint(len(cluster), size=4)  # Select random indices
        set_of_images = [cluster[d].image for d in random_index]  # Extract images
        try_all_threshold_techniques_v2(set_of_images, toSave, index)  # Try threshold techniques
        plt.show()  # Display plots
        val1 = easygui.enterbox(
            "Enter your value: Enter 1 (Option 1), 2 (Option 2), 3 (Option 3), 4 (Option 4), 5 (Option 5)")  # User input

        # Loop until user selects consistent option
        while True:
            fig = plt.figure()
            random_index = np.random.randint(len(cluster), size=4)
            set_of_images = [cluster[d].image for d in random_index]
            try_all_threshold_techniques_v2(set_of_images, toSave, index)
            plt.show()
            val2 = easygui.enterbox("Enter your value: Enter 1 (Option 1), 2 (Option 2), 3 (Option 3), 4 (Option 4), 5 (Option 5)")

            if val1 == val2:
                settings_history.append([int(val1), 1])  # Append selected value to history
                break
            else:
                val1 = val2  # Update value


        settings_history = adjust_segmentation_v2(cluster, settings_history=settings_history, step=0.2, toSave=toSave,
                                                  title=str(index))  # Adjust segmentation
        val, coef = settings_history[-1]  # Retrieve latest settings
        segmentation_settings.append([int(val), coef])  # Store segmentation settings
        cluster = update_associated_setting(cluster, int(val), coef)  # Update associated settings
        updated_objects += second_segmentation(cluster)  # Perform second segmentation

    settings_history.append([-1, -1])  # Placeholder for final settings
    settings_history.append([1, 1])  # Placeholder for border settings
    settings_history = segmenting_borderline_objects(objects=border_cluster, settings=segmentation_settings,
                                                     settings_history=settings_history)  # Segment borderline objects
    val, coef = settings_history[-1]  # Retrieve latest settings
    border_cluster = update_associated_setting(border_cluster, int(val), coef)  # Update associated settings
    updated_objects += second_segmentation(border_cluster)  # Perform second segmentation

    # Save data if specified
    if toSave == 1:
        save_data(updated_objects, file_name)
        np.save('Workspace_data/settings.npy', settings_history)

    return updated_objects, settings_history  # Return updated objects and segmentation history


def second_segmentation(list_of_objects):
    """
    Perform a secondary segmentation on a list of objects objects.

    Parameters:
    list_of_focus (list): List of objects objects.

    Returns:
    list: List of objects objects with updated segmentation attributes.
    """
    nn = len(list_of_objects)  # Number of objects objects
    kernel_size = 3  # Kernel size for dilation

    for iter in range(nn):
        img = list_of_objects[iter].image  # Original image
        blur = list_of_objects[iter].blury  # Blurred image
        image_denoised = denoising(img)  # Denoise the image
        thresh_optimum = optimised_threshold_based_on_crossEntropy(image_denoised)  # Optimal threshold
        thresh_yen = threshold_yen(image_denoised)  # Yen's threshold

        # Extract segmentation parameters from objects object
        val = list_of_objects[iter].segmentation_algorithm
        coef = list_of_objects[iter].correction_coef

        # Determine threshold based on segmentation algorithm
        if val == 1:
            thresh = thresh_optimum
        elif val == 2:
            thresh = 2 * thresh_optimum
        elif val == 3:
            thresh = threshold_otsu(image_denoised)
        elif val == 4:
            thresh = threshold_yen(image_denoised)
        else:
            thresh = riddler_calvard_threshold(image_denoised)

        # Create mask based on threshold
        mask = image_denoised > coef * thresh

        # Label connected regions in the mask
        label_img = label(mask)

        # Extract properties of connected regions
        props = regionprops(label_img, img)

        # If regions are found, update objects object attributes
        if len(props) > 0:
            tmp = [p.filled_area for p in props]
            index = np.argmax(tmp)
            center = props[index].centroid
            center = [math.floor(x) for x in center]
            list_of_objects[iter].mask = mask
            list_of_objects[iter].center = center
            list_of_objects[iter].area = props[index].filled_area
            list_of_objects[iter].mean_intensity = props[index].mean_intensity
            list_of_objects[iter].solidity = props[index].solidity

        # Optional: Perform dilation (uncomment if needed)
        # for i in range(0, 5):
        #     kernel = np.ones((kernel_size, kernel_size), np.uint8)
        #     tmp = cv2.dilate(list_of_focus[iter].image, kernel, iterations=i)
        #     thresh = threshold_otsu(tmp)
        #     list_of_focus[iter].dilation[i] = tmp > thresh

    return list_of_objects



def try_all_threshold_techniques_v2(images, toSave=0, index=0, WEIGHT=1.7):
    """
    Try multiple thresholding techniques on a list of images.

    Parameters:
    images (list): List of images to process.
    toSave (int, optional): Flag to save the results. Default is 0.
    index (int, optional): Index for file naming. Default is 0.
    WEIGHT (float, optional): Weight factor. Default is 1.7.

    Returns:
    tuple: Tuple containing the figure and axes objects.
    """
    fig, axs = plt.subplots(len(images), 6)

    for i, image in enumerate(images):
        image_denoised = denoising(image)
        thresh_optimum = optimised_threshold_based_on_crossEntropy(image_denoised)
        thresh_yen = threshold_yen(image_denoised)
        thresh_otsu = threshold_otsu(image_denoised)
        thresh_riddler = riddler_calvard_threshold(image_denoised)

        # Generate masks using different thresholds
        mask1 = image_denoised > thresh_optimum
        mask2 = image_denoised > 2 * thresh_optimum
        mask3 = image_denoised > thresh_otsu
        mask4 = image_denoised > thresh_yen
        mask5  = image_denoised > thresh_riddler

        max_value = np.max(image)

        # Plot original image and images with different masks
        plot_image_with_mask(axs[i, 0], image, "original", max_value)
        plot_image_with_mask(axs[i, 1], image, "Option 1", max_value, mask1)
        plot_image_with_mask(axs[i, 2], image, "Option 2", max_value, mask2)
        plot_image_with_mask(axs[i, 3], image, "Option 3", max_value, mask3)
        plot_image_with_mask(axs[i, 4], image, "Option 4", max_value, mask4)
        plot_image_with_mask(axs[i, 5], image, "Option 5", max_value, mask5)

    if toSave:
        fig.savefig(f"Raw_Figs/all_segmentation_methods{index}.svg", format='svg')

    return fig, axs


def plot_image_with_mask(ax, image, title, max_value, mask=None, ):
    """
    Plot image with optional mask.

    Parameters:
    ax (AxesSubplot): Matplotlib axis object.
    image (numpy.ndarray): Image to plot.
    title (str): Title for the subplot.
    mask (numpy.ndarray, optional): Mask to overlay on the image. Default is None.
    max_value (float, optional): Maximum value for normalization. Default is None.
    """
    ax.imshow(image, cmap="gray", vmin=np.percentile(image, 0.01), vmax=np.percentile(image, 99.99))
    if mask is not None:
        ax.imshow(bwperim(mask) * max_value, cmap="gray", alpha=0.5)
    else:
        ax.imshow(image, cmap="gray", alpha=0.5)
    ax.set_title(title)
    ax.axis('off')


def adjust_segmentation_v2(focus_list, settings_history, step=0.1, toSave=0, title=''):
    """
    Adjust segmentation based on previous settings.

    Parameters:
    focus_list (list): List of objects.
    settings_history (list): History of segmentation settings.
    step (float, optional): Step size for adjustment. Default is 0.1.
    toSave (int, optional): Flag to save the results. Default is 0.
    title (str, optional): Title for file naming. Default is ''.

    Returns:
    list: Updated segmentation settings history.
    """
    val, coef = settings_history[-1]
    random_index = np.random.randint(len(focus_list), size=4)
    images = [focus_list[d].image for d in random_index]
    threshes = []
    denoised_images = []
    # Compute threshold for each image
    for image in images:
        image_denoised = denoising(image)
        denoised_images.append(image_denoised)
        thresh_optimum = optimised_threshold_based_on_crossEntropy(image_denoised)


        if val == 1:
            thresh = thresh_optimum
        elif val == 2:
            thresh = 2 * thresh_optimum
        elif val == 3:
            thresh = threshold_otsu(image_denoised)
        elif val == 4:
            thresh = threshold_yen(image_denoised)
        else:
            thresh = riddler_calvard_threshold(image_denoised)

        threshes.append(coef * thresh)

    # Call recursive adjustment function
    return adjust_segmentation_recursive_v2(focus_list, images, denoised_images, threshes, settings_history, step, toSave, title)


def adjust_segmentation_recursive_v2(focus_list, images, denoised_images, threshes, settings_history, step=0.1, toSave=0, title=''):
    """
    Recursively adjust segmentation based on different thresholding options.

    Parameters:
    focus_list (list): List of objects.
    images (list): List of original images.
    denoised_images (list): List of denoised images.
    threshes (list): List of thresholds for denoised images.
    settings_history (list): History of segmentation settings.
    step (float, optional): Step size for adjustment. Default is 0.1.
    toSave (int, optional): Flag to save the results. Default is 0.
    title (str, optional): Title for file naming. Default is ''.

    Returns:
    list: Updated segmentation settings history.
    """
    val, coef = settings_history[-1]
    fig, axs = plt.subplots(len(images), 6)
    for i in range(0, len(images)):
        image = images[i]
        denoised_image = denoised_images[i]
        thresh = threshes[i]
        mask0 = denoised_image > (1 - 2 * step) * thresh
        mask1 = denoised_image > (1 - step) * thresh
        mask2 = denoised_image > thresh
        mask3 = denoised_image > (1 + step) * thresh
        mask4 = denoised_image > (1 + 2 * step) * thresh

        max_value = np.max(image)
        # Plot original image and images with different masks
        plot_image_with_mask(axs[i][0], image, "original", max_value)
        plot_image_with_mask(axs[i][1], image, "1", max_value, mask0)
        plot_image_with_mask(axs[i][2], image, "2", max_value, mask1)
        plot_image_with_mask(axs[i][3], image, "3", max_value, mask2)
        plot_image_with_mask(axs[i][4], image, "4", max_value, mask3)
        plot_image_with_mask(axs[i][5], image, "5", max_value, mask4)
    plt.axis('off')
    plt.show()
    if toSave == 1:
        file_name = f"Raw_Figs/tune_segmentation_{title}_step_{step}.svg"
        fig.savefig(file_name, format="svg")
    myvar = easygui.enterbox("What is the best segmentation? Choose between (1 to 5). Choose 6 if all look the same.")
    if myvar == '6':
        return settings_history
    elif myvar in ['1', '2', '3', '4', '5']:
        adjustment_factor = {
            '1': (1 - 2 * step) * coef,
            '2': (1 - step) * coef,
            '3': coef,
            '4': (1 + step) * coef,
            '5': (1 + 2 * step) * coef
        }[myvar]
        settings_history.append([val, adjustment_factor])
        return adjust_segmentation_v2(focus_list=focus_list, step=step * 3 / 4, settings_history=settings_history, toSave=1)
    else:
        print("Invalid input. Segmentation unchanged.")
        return settings_history


def update_associated_setting(objects, segmentation_algorithm, correction_coef):
    """
    Update segmentation algorithm and correction coefficient for a list of objects.

    Parameters:
    objects (list): List of objects to update.
    segmentation_algorithm (int): Segmentation algorithm identifier.
    correction_coef (float): Correction coefficient value.

    Returns:
    list: Updated list of objects.
    """
    for obj in objects:
        obj.segmentation_algorithm = segmentation_algorithm
        obj.correction_coef = correction_coef
    return objects

def segmenting_borderline_objects(objects, settings, settings_history):
    """
    Segment borderline objects based on provided settings.

    Parameters:
    objects (list): List of objects.
    settings (list): List of segmentation settings.
    settings_history (list): History of segmentation settings.

    Returns:
    list: Updated settings history.
    """
    random_index = np.random.randint(len(objects), size=4)
    images = [objects[d].image for d in random_index]
    threshes = []
    denoised_images = []
    fig, axs = plt.subplots(len(images), 3)

    for i in range(len(images)):
        image = images[i]
        image_denoised = denoising(image)
        denoised_images.append(image_denoised)
        axs[i][0].imshow(image, cmap="gray",
                         vmin=np.percentile(image, 0.01),
                         vmax=np.percentile(image, 99.99))
        axs[i][0].axis('off')
        axs[i][1], thresh1 = draw_one_axis(axs[i][1], image, image_denoised, val=settings[0][0], coef=settings[0][1])
        axs[i][2], thresh2 = draw_one_axis(axs[i][2], image, image_denoised, val=settings[1][0], coef=settings[1][1])
        threshes.append([thresh1, thresh2])

    axs[0][0].set_title("original")
    axs[0][1].set_title("Option1")
    axs[0][2].set_title("Option2")
    plt.show()

    val1 = easygui.enterbox("Enter your value: Enter 1 (Option 1), 2 (Option 2)")
    thresh = [a[int(val1) - 1] for a in threshes]
    val = settings[int(val1) - 1][0]
    coef = settings[int(val1) - 1][1]
    settings_history.append([val, coef])
    settings_history = adjust_segmentation_recursive_v2(objects, images, denoised_images, threshes=thresh, settings_history=settings_history, toSave=0)
    return settings_history

def draw_one_axis(ax, image, image_denoised, val, coef):
    """
    Draw segmentation results on a single axis.

    Parameters:
    ax: Matplotlib axis to draw on.
    image (numpy.ndarray): Original image.
    image_denoised (numpy.ndarray): Denoised image.
    val (int): Segmentation algorithm identifier.
    coef (float): Correction coefficient.

    Returns:
    tuple: Tuple containing the axis and the threshold used for segmentation.
    """
    thresh_optimum = optimised_threshold_based_on_crossEntropy(image_denoised)
    if val == 1:
        thresh = thresh_optimum
    elif val == 2:
        thresh = 1.5 * thresh_optimum
    elif val == 3:
        thresh = threshold_yen(image_denoised)
    else:
        thresh = 2.5 * thresh_optimum
    mask = image_denoised > coef * thresh
    max_value = np.max(image)
    temp = bwperim(mask) * max_value
    ax.imshow(image + temp, cmap="gray",
              vmin=np.percentile(image, 0.01),
              vmax=np.percentile(image, 99.99))
    ax.axis('off')
    return ax, thresh

