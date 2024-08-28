import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from exceptions import *  # Ensure this module is correctly defined or installed.

MINIMUM_MATCH_POINTS = 15
CONFIDENCE_THRESH = 10  # Confidence threshold for homography computation.

def plot_interactive_keypoints(image, keypoints, title):
    fig = go.Figure(data=[go.Scatter(x=[kp[0] for kp in keypoints], y=[kp[1] for kp in keypoints],
                                     mode='markers', marker=dict(size=5, color='purple'))])
    fig.update_layout(title=title, xaxis_title='X Coordinate', yaxis_title='Y Coordinate',
                      width=800, height=600, autosize=False)
    fig.update_xaxes(range=[0, image.shape[1]])
    fig.update_yaxes(range=[0, image.shape[0]], autorange="reversed")  # Image coordinates are reversed in y-axis
    fig.show()

def draw_keypoints(vis, keypoints, color):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(vis, (int(x), int(y)), 5, color, -1)

# def visualize_transformation(img_a, img_b, h_mat):
#     """Overlay the boundary of img_b onto img_a using the computed homography matrix."""
#     corners_img_b = get_corners_as_array(img_b)
#     transformed_corners = transform_with_homography(h_mat, corners_img_b)

#     # Draw the transformed polygon on img_a
#     img_a_with_overlay = img_a.copy()
#     if len(img_a_with_overlay.shape) == 2:  # Ensure it's a color image
#         img_a_with_overlay = cv2.cvtColor(img_a_with_overlay, cv2.COLOR_GRAY2BGR)

#     pts = np.int32(transformed_corners).reshape((-1, 1, 2))
#     img_a_with_overlay = cv2.polylines(img_a_with_overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

#     plt.figure(figsize=(10, 10))
#     plt.imshow(img_a_with_overlay)
#     plt.title('Transformation Visualization')
#     plt.axis('off')
#     plt.show()

def get_matches(img_a_gray, img_b_gray, num_keypoints=1000, threshold=0.8):
    '''Function to get matched keypoints from two images using ORB

    Args:
        img_a_gray (numpy array): of shape (H, W) representing grayscale image A
        img_b_gray (numpy array): of shape (H, W) representing grayscale image B
        num_keypoints (int): number of points to be matched (default=100)
        threshold (float): can be used to filter strong matches only. Lower the value, stronger the requirements and hence fewer matches.
    Returns:
        match_points_a (numpy array): of shape (n, 2) representing x,y pixel coordinates of image A keypoints
        match_points_b (numpy array): of shape (n, 2) representing x,y pixel coordianted of matched keypoints in image B
    '''
    
    orb = cv2.ORB_create(nfeatures=num_keypoints)
    kp_a, desc_a = orb.detectAndCompute(img_a_gray, None)
    kp_b, desc_b = orb.detectAndCompute(img_b_gray, None)
    
    dis_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_list = dis_matcher.knnMatch(desc_a, desc_b, k=2)

    good_matches_list = []
    for match_1, match_2 in matches_list:
        if match_1.distance < threshold * match_2.distance:
            good_matches_list.append(match_1)

    plot_interactive_keypoints(img_a_gray, [kp_a[m.queryIdx].pt for m in good_matches_list], "Keypoints in Image A")
    plot_interactive_keypoints(img_b_gray, [kp_b[m.trainIdx].pt for m in good_matches_list], "Keypoints in Image B")

    img_kp_a = cv2.drawKeypoints(img_a_gray, kp_a, None, color=(0, 255, 0), flags=0)
    img_kp_b = cv2.drawKeypoints(img_b_gray, kp_b, None, color=(0, 255, 0), flags=0)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img_kp_a, cmap='gray')
    plt.title('Keypoints in Image A')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(img_kp_b, cmap='gray')
    plt.title('Keypoints in Image B')
    plt.axis('off')

    plt.show()

    # Visualization of keypoints and matches
    img_matches = cv2.drawMatches(img_a_gray, kp_a, img_b_gray, kp_b, good_matches_list, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('Key Points and Matches')
    plt.axis('off')
    plt.show()

    if len(good_matches_list) < MINIMUM_MATCH_POINTS:
        raise NotEnoughMatchPointsError(len(good_matches_list), MINIMUM_MATCH_POINTS)
    
    return np.array([kp_a[m.queryIdx].pt for m in good_matches_list]), np.array([kp_b[m.trainIdx].pt for m in good_matches_list])

def calculate_homography(points_img_a, points_img_b):
    points_a_and_b = np.concatenate((points_img_a, points_img_b), axis=1)
    A = []
    for u, v, x, y in points_a_and_b:
        A.extend([[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                  [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
    A = np.array(A)
    _, _, v_t = np.linalg.svd(A)
    h_mat = v_t[-1, :].reshape(3, 3)
    print("Homography Matrix:\n", h_mat)
    return h_mat


def transform_with_homography(h_mat, points_array):
    ones_col = np.ones((points_array.shape[0], 1))
    points_array_homog = np.concatenate((points_array, ones_col), axis=1)
    transformed_points = np.matmul(h_mat, points_array_homog.T)
    transformed_points /= transformed_points[2, :]  # Normalize
    return transformed_points[:2, :].T

def compute_outliers(h_mat, points_img_a, points_img_b, threshold=3):
    points_img_b_hat = transform_with_homography(h_mat, points_img_b)
    errors = np.linalg.norm(points_img_b_hat - points_img_a, axis=1)
    outliers = errors > threshold
    print("Outliers count:", np.sum(outliers))
    return np.sum(outliers)

def compute_homography_ransac(matches_a, matches_b):
    num_all_matches = matches_a.shape[0]
    SAMPLE_SIZE = 4
    SUCCESS_PROB = 0.995
    min_iterations = int(np.log(1 - SUCCESS_PROB) / np.log(1 - pow(0.5, SAMPLE_SIZE)))

    lowest_outliers_count = num_all_matches
    best_h_mat = None

    for i in range(min_iterations):
        rand_indices = np.random.choice(num_all_matches, SAMPLE_SIZE, replace=False)
        sampled_matches_a = matches_a[rand_indices]
        sampled_matches_b = matches_b[rand_indices]
        h_mat = calculate_homography(sampled_matches_a, sampled_matches_b)
        outliers_count = compute_outliers(h_mat, matches_a, matches_b)
        if outliers_count < lowest_outliers_count:
            best_h_mat = h_mat
            lowest_outliers_count = outliers_count

    best_confidence_obtained = (1 - lowest_outliers_count / num_all_matches) * 100
    print(f"Best confidence obtained: {best_confidence_obtained}%")
    if best_confidence_obtained < CONFIDENCE_THRESH:
        raise MatchesNotConfident(best_confidence_obtained, CONFIDENCE_THRESH)
    return best_h_mat

# def get_corners_as_array(img):
#     """Get the corner points of an image."""
#     h, w = img.shape[:2]
#     return np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1, 1, 2)


def get_corners_as_array(img_height, img_width):
    """Function to extract the corner points of an image from its width and height and arrange it in the form
        of a numpy array.
        
        The 4 corners are arranged as follows:
        corners = [top_left_x, top_left_y;
                   top_right_x, top_right_y;
                   bottom_right_x, bottom_right_y;
                   bottom_left_x, bottom_left_y]

    Args:
        img_height (str): height of the image
        img_width (str): width of the image
    
    Returns:
        corner_points_array (numpy array): of shape (4,2) representing for corners with x,y pixel coordinates
    """
    corners_array = np.array([[0, 0],
                            [img_width - 1, 0],
                            [img_width - 1, img_height - 1],
                            [0, img_height - 1]])
    return corners_array


def get_crop_points_horz(img_a_h, transfmd_corners_img_b):
    """Function to find the pixel corners in the horizontally stitched images to crop and remove the
        black space around.
    
    Args:
        img_a_h (int): the height of the pivot image that is image A
        transfmd_corners_img_b (numpy array): of shape (n, 2) representing the transformed corners of image B
            The corners need to be in the following sequence:
            corners = [top_left_x, top_left_y;
                   top_right_x, top_right_y;
                   bottom_right_x, bottom_right_y;
                   bottom_left_x, bottom_left_y]
    Returns:
        x_start (int): the x pixel-cordinate to start the crop on the stitched image
        y_start (int): the x pixel-cordinate to start the crop on the stitched image
        x_end (int): the x pixel-cordinate to end the crop on the stitched image
        y_end (int): the y pixel-cordinate to end the crop on the stitched image
    """
    # the four transformed corners of image B
    top_lft_x_hat, top_lft_y_hat = transfmd_corners_img_b[0, :]
    top_rht_x_hat, top_rht_y_hat = transfmd_corners_img_b[1, :]
    btm_rht_x_hat, btm_rht_y_hat = transfmd_corners_img_b[2, :]
    btm_lft_x_hat, btm_lft_y_hat = transfmd_corners_img_b[3, :]

    # initialize the crop points
    # since image A (on the left side) is used as pivot, x_start will always be zero
    x_start, y_start, x_end, y_end = (0, None, None, None)

    if (top_lft_y_hat > 0) and (top_lft_y_hat > top_rht_y_hat):
        y_start = top_lft_y_hat
    elif (top_rht_y_hat > 0) and (top_rht_y_hat > top_lft_y_hat):
        y_start = top_rht_y_hat
    else:
        y_start = 0
        
    if (btm_lft_y_hat < img_a_h - 1) and (btm_lft_y_hat < btm_rht_y_hat):
        y_end = btm_lft_y_hat
    elif (btm_rht_y_hat < img_a_h - 1) and (btm_rht_y_hat < btm_lft_y_hat):
        y_end = btm_rht_y_hat
    else:
        y_end = img_a_h - 1

    if (top_rht_x_hat < btm_rht_x_hat):
        x_end = top_rht_x_hat
    else:
        x_end = btm_rht_x_hat
    
    return int(x_start), int(y_start), int(x_end), int(y_end)


def get_crop_points_vert(img_a_w, transfmd_corners_img_b):
    """Function to find the pixel corners in the vertically stitched images to crop and remove the
        black space around.
    
    Args:
        img_a_h (int): the width of the pivot image that is image A
        transfmd_corners_img_b (numpy array): of shape (n, 2) representing the transformed corners of image B
            The corners need to be in the following sequence:
            corners = [top_left_x, top_left_y;
                   top_right_x, top_right_y;
                   bottom_right_x, bottom_right_y;
                   bottom_left_x, bottom_left_y]
    Returns:
        x_start (int): the x pixel-cordinate to start the crop on the stitched image
        y_start (int): the x pixel-cordinate to start the crop on the stitched image
        x_end (int): the x pixel-cordinate to end the crop on the stitched image
        y_end (int): the y pixel-cordinate to end the crop on the stitched image
    """
    # the four transformed corners of image B
    top_lft_x_hat, top_lft_y_hat = transfmd_corners_img_b[0, :]
    top_rht_x_hat, top_rht_y_hat = transfmd_corners_img_b[1, :]
    btm_rht_x_hat, btm_rht_y_hat = transfmd_corners_img_b[2, :]
    btm_lft_x_hat, btm_lft_y_hat = transfmd_corners_img_b[3, :]

    # initialize the crop points
    # since image A (on the top) is used as pivot, y_start will always be zero
    x_start, y_start, x_end, y_end = (None, 0, None, None)

    if (top_lft_x_hat > 0) and (top_lft_x_hat > btm_lft_x_hat):
        x_start = top_lft_x_hat
    elif (btm_lft_x_hat > 0) and (btm_lft_x_hat > top_lft_x_hat):
        x_start = btm_lft_x_hat
    else:
        x_start = 0
        
    if (top_rht_x_hat < img_a_w - 1) and (top_rht_x_hat < btm_rht_x_hat):
        x_end = top_rht_x_hat
    elif (btm_rht_x_hat < img_a_w - 1) and (btm_rht_x_hat < top_rht_x_hat):
        x_end = btm_rht_x_hat
    else:
        x_end = img_a_w - 1

    if (btm_lft_y_hat < btm_rht_y_hat):
        y_end = btm_lft_y_hat
    else:
        y_end = btm_rht_y_hat
    
    return int(x_start), int(y_start), int(x_end), int(y_end)


def get_crop_points(h_mat, img_a, img_b, stitch_direc):
    """Function to find the pixel corners to crop the stitched image such that the black space 
        in the stitched image is removed.
        The black space could be because either image B is not of the same dimensions as image A
        or image B is skewed after homographic transformation.
        Example: 
                  (Horizontal stitching)
                ____________                     _________________
                |           |                    |                |
                |           |__________          |                |
                |           |         /          |       A        |
                |     A     |   B    /           |________________|
                |           |       /                |          | 
                |           |______/                 |    B     |
                |___________|                        |          |
                                                     |__________|  <-imagine slant bottom edge
        
        This function returns the corner points to obtain the maximum area inside A and B combined and making
        sure the edges are straight (i.e horizontal and veritcal). 

    Args:
        h_mat (numpy array): of shape (3, 3) representing the homography from image B to image A
        img_a (numpy array): of shape (h, w, c) representing image A
        img_b (numpy array): of shape (h, w, c) representing image B
        stitch_direc (int): 0 when stitching vertically and 1 when stitching horizontally

    Returns:
        x_start (int): the x pixel-cordinate to start the crop on the stitched image
        y_start (int): the x pixel-cordinate to start the crop on the stitched image
        x_end (int): the x pixel-cordinate to end the crop on the stitched image
        y_end (int): the y pixel-cordinate to end the crop on the stitched image          
    """
    img_a_h, img_a_w, _ = img_a.shape
    img_b_h, img_b_w, _ = img_b.shape

    orig_corners_img_b = get_corners_as_array(img_b_h, img_b_w)
                
    transfmd_corners_img_b = transform_with_homography(h_mat, orig_corners_img_b)

    if stitch_direc == 1:
        x_start, y_start, x_end, y_end = get_crop_points_horz(img_a_w, transfmd_corners_img_b)
    # initialize the crop points
    x_start = None
    x_end = None
    y_start = None
    y_end = None

    if stitch_direc == 1: # 1 is horizontal
        x_start, y_start, x_end, y_end = get_crop_points_horz(img_a_h, transfmd_corners_img_b)
    else: # when stitching images in the vertical direction
        x_start, y_start, x_end, y_end = get_crop_points_vert(img_a_w, transfmd_corners_img_b)
    return x_start, y_start, x_end, y_end


def stitch_image_pair(img_a, img_b, stitch_direc):
    print("inside stich image pair function")

    """Function to stitch image B to image A in the mentioned direction

    Args:
        img_a (numpy array): of shape (H, W, C) with opencv representation of image A (i.e C: B,G,R)
        img_b (numpy array): of shape (H, W, C) with opencv representation of image B (i.e C: B,G,R)
        stitch_direc (int): 0 for vertical and 1 for horizontal stitching

    Returns:
        stitched_image (numpy array): stitched image with maximum content of image A and image B after cropping
            to remove the black space 
    """

    print("inside stich image pair function")
    img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    matches_a, matches_b = get_matches(img_a_gray, img_b_gray, num_keypoints=1000, threshold=0.8)

    print("matching done")

    h_mat = compute_homography_ransac(matches_a, matches_b)
    # visualize_transformation(img_a, img_b, h_mat)

    print("homography done")

    if stitch_direc == 0:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1], img_a.shape[0] + img_b.shape[0]))
        canvas[0:img_a.shape[0], :, :] = img_a[:, :, :]
        x_start, y_start, x_end, y_end = get_crop_points(h_mat, img_a, img_b, 0)
    else:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1] + img_b.shape[1], img_a.shape[0]))
        canvas[:, 0:img_a.shape[1], :] = img_a[:, :, :]
        x_start, y_start, x_end, y_end = get_crop_points(h_mat, img_a, img_b, 1)
    
    stitched_img = canvas[y_start:y_end,x_start:x_end,:]
    return stitched_img


def check_imgfile_validity(folder, filenames):
    """Function to check if the files in the given path are valid image files.
    
    Args:
        folder (str): path containing the image files
        filenames (list): a list of image filenames

    Returns:
        valid_files (bool): True if all the files are valid image files else False
        msg (str): Message that has to be displayed as error
    """
    for file in filenames:
        full_file_path = os.path.join(folder, file)
        regex = "([^\\s]+(\\.(?i:(jpe?g|png)))$)"
        p = re.compile(regex)

        if not os.path.isfile(full_file_path):
            return False, "File not found: " + full_file_path
        if not (re.search(p, file)):
            return False, "Invalid image file: " + file
    return True, None
