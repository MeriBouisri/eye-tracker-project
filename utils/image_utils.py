import numpy as np
import cv2

# ========== IMAGE PROCESSING FUNCTIONS ========== #

def apply_single_scale_retinex(image, sigma, ksize = 0):
    """
    Enhance the image using the single scale retinex algorithm.
    Code adapted from: https://santhalakshminarayana.github.io/blog/retinex-image-enhancement

    Parameters
    ----------
    image : np.ndarray
        The image to enhance.

    sigma : float
        The standard deviation of the Gaussian kernel used to blur the image.

    ksize : int, optional, default = 0
        The size of the Gaussian kernel used to blur the image.

    Returns
    ----------
    enhanced_image : np.ndarray
        The image after applying the single scale retinex algorithm.
    """
    if ksize == 0:
        ksize = int(((sigma - 0.8)/0.15) + 2.0)

    kernel = cv2.getGaussianKernel(ksize, sigma)
    gaussian_blur = cv2.filter2D(image, -1, np.outer(kernel, kernel))

    # add epsilon to avoid log(0)
    epsilon = 1e-8
    img_log = np.log10(image + epsilon)
    img_blur_log = np.log10(gaussian_blur + epsilon)

    return img_log - img_blur_log


# ========== FORM TRANSFORMATION FUNCTIONS ========== #

def crop_frame(frame, upper_left_corner, lower_right_corner):
    """
    Crop the frame to the area enclosed within the upper left and lower right corners of a rectangle.

    Parameters
    ----------
    frame : np.ndarray
        The frame to crop.
    
    upper_left_corner : tuple[int, int]
        The (x, y) coordinates of the upper left corner of the rectangle to crop to.
        The upper left corner is made up of the minimum x and y values of the rectangle.

    lower_right_corner : tuple[int, int]
        The (x, y) coordinates of the lower right corner of the rectangle to crop to.
        The lower right corner is made up of the maximum x and y values of the rectangle.

    Returns
    ----------
    cropped_frame : ndarray
        The section of the input frame cropped to the area enclosed inside the rectangle defined by the upper left and lower right corners.
        The output frame is not a copy of the input frame, such as any changes made to the output frame will also affect the input frame.

    Notes
    ----------
    If the rectangle defined by the upper left and lower right corners extends beyond the frame's dimensions, 
    the rectangle is clipped to the frame's dimensions.

    See Also
    ----------
    get_min_enclosing_rectangle : Get the upper left corner and lower right corner of the minimum enclosing rectangle of a set of points.
    draw_min_enclosing_rectangle : Draw the minimum enclosing rectangle of a set of points on a frame.
    """

    min_height, min_width = 0, 0
    max_height, max_width, _ = frame.shape

    min_column, min_row = np.clip(upper_left_corner, min_height, max_height)
    max_column, max_row = np.clip(lower_right_corner, min_width, max_width)

    return frame[min_row:max_row, min_column:max_column]

def rotate_frame(frame, angle, rotation_axis=None, scale=1):
    """
    Rotate a frame by a given angle around a given axis.

    Parameters
    ----------
    frame : np.ndarray
        The frame to rotate.

    angle : float
        The angle in degrees by which to rotate the frame.

    rotation_axis : tuple[int, int], optional, default = None
        The (x, y) coordinates of the point around which to rotate the frame. 
        If None, the frame is rotated around its center.

    scale : int, optional, default = 1
        The scale factor to apply to the frame after rotation. 
        The default is to not scale the frame.

    Returns
    ----------
    rotated_frame : ndarray
        A copy of the input frame, rotated by the given angle around the point specified by the rotation_axis.
    """

    if rotation_axis is None:
        rotation_axis = np.array(frame.shape[:2]) / 2

    rotation_matrix = cv2.getRotationMatrix2D(rotation_axis, angle, scale)
    return cv2.warpAffine(frame, rotation_matrix, frame.shape[:2][::-1])

