import cv2
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/sequences/00/'
IDX = 000000
MATCHES_NUM = 20
RATIO = 0.75
MATCHES_NORM = cv2.NORM_L2
PASSED = "PASSED"
FAILED = "FAILED"
ALG = cv2.SIFT_create()


# == Ex1 == #
def read_images(idx):
    """
    :param idx: Images's index in the Kitti dataset
    :return: left and right cameras photos
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    return img1, img2


def feature_detection_and_description(img1, img2, alg):
    """
    Computes images key points and their's descriptors
    :param alg: Feature detecting and description algorithm
    :return: images key points and descriptors
    """
    img1_kpts, img1_dsc = alg.detectAndCompute(img1, None)
    img2_kpts, img2_dsc = alg.detectAndCompute(img2, None)
    return np.array(img1_kpts), np.array(img1_dsc), np.array(img2_kpts), np.array(img2_dsc)


def bf_matching(metric, img1_dsc, img2_dsc, crossCheck=True, sort=True):
    """
    Find Matches between two images descriptors
    :param metric: distance function for computes distance between two descriptors
    :param img1_dsc: image 1 descriptors
    :param img2_dsc: image 2 descriptors
    :return: array of matches
    """
    bf = cv2.BFMatcher(metric, crossCheck=crossCheck)
    matches = bf.match(img1_dsc, img2_dsc)
    if sort:
        # Sort the matches from the best match to the worst - where best means it has the lowest distance
        matches = sorted(matches, key=lambda x: x.distance)
    return matches


def knn_flann_matching(img1_dsc, img2_dsc):
    """
    Find Matches between two images descriptors
    :param img1_dsc: image 1 descriptors
    :param img2_dsc: image 2 descriptors
    :return: Array of matches
    """
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(img1_dsc, img2_dsc, k=2)
    return matches


def knn_matching(metric, img1_dsc, img2_dsc, crossCheck=True, sort=True):
    """
    Find Matches between two images descriptors
    :param metric: distance function for computes distance between two descriptors
    :param img1_dsc: image 1 descriptors
    :param img2_dsc: image 2 descriptors
    :return: array of matches
    """
    bf = cv2.BFMatcher(metric)
    matches = bf.knnMatch(img1_dsc, img2_dsc, 2)
    return matches


def matching(img1_dsc, img2_dsc): # Todo: we left those options for further checkings
    # matches = knn_matching(MATCHES_NORM, img1_dsc, img2_dsc)
    matches = knn_flann_matching(img1_dsc, img2_dsc)
    matches, _ = significance_test(matches, RATIO)  # notice does not return np array
    # matches = bf_matching(MATCHES_NORM, img1_dsc, img2_dsc)
    return np.array(matches)


def detect_and_match(img1, img2):
    # Detects the image key points and compute their descriptors
    img1_kpts, img1_dsc, img2_kpts, img2_dsc = feature_detection_and_description(img1, img2, ALG)

    # Matches between the images and plots the matching
    matches = matching(img1_dsc, img2_dsc)
    return img1_kpts, img1_dsc, img2_kpts, img2_dsc, matches


def read_cameras():
    with open(DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
        l1 = [float(i) for i in l1]
        m1 = np.array(l1).reshape(3, 4)
        l2 = [float(i) for i in l2]
        m2 = np.array(l2).reshape(3, 4)
        k = m1[:, :3]
        m1 = np.linalg.inv(k) @ m1
        m2 = np.linalg.inv(k) @ m2
        return k, m1, m2


def draw_matches(img1, img1_kpts, img2, img2_kpts, matches):
    """
    Draws a line between matches in img1 and img2
    """
    result = cv2.drawMatches(img1, img1_kpts, img2, img2_kpts, matches, None, flags=2)
    # Display the best matching points
    plt.rcParams['figure.figsize'] = [20.0, 10.0]
    plt.title(f'Best Matching Points. Num matches:{len(matches)}')
    plt.imshow(result)
    plt.show()


def significance_test(matches, ratio):
    """
    Applying the significance test - rejects all matches that their distance ratio between the 1st and 2nd
    nearest neighbors is lower than predetermined RATIO.
    In practice:
        1. Finds 2 best matches in img2 for each descriptor in img1.
        2. Rejects all the matches that are not passing the test.
        3. Shows the new matches - that passed the test.
    :param ratio : The ratio threshold between the best and the second match
    :param rand: Boolean value that determines whether to choose part of the matches randomly or the best ones.
    """

    # Apply ratio test
    pass_test = []
    failed_test = []
    for first, second in matches:
        if first.distance < ratio * second.distance:
            pass_test.append(first)
        else:
            failed_test.append(first)

    return pass_test, failed_test


# == Ex2 == #

def axes_of_matches(match, img1_kpts, img2_kpts):
    """
    Returns (x,y) values for each point from the two in the match object
    """
    img1_idx = match.queryIdx
    img2_idx = match.trainIdx
    x1, y1 = img1_kpts[img1_idx].pt
    x2, y2 = img2_kpts[img2_idx].pt
    return x1, y1, x2, y2


def rectified_stereo_pattern_rej(img1_kpts, img2_kpts, matches):  # Todo: Efficiency for further exercises
    """
    Apply the rectified stereo pattern rejection on image1 key points and image 2 key points
    :return: List of Inliers and outliers indexes of the 2 images
    """
    inliers_matches_idx, outliers_matches_idx = [], []

    num_matches = len(matches)
    for i in range(num_matches):
        _, y1, _, y2 = axes_of_matches(matches[i], img1_kpts, img2_kpts)
        if abs(y2 - y1) < 1:
            inliers_matches_idx.append(i)
        else:
            outliers_matches_idx.append(i)

    return inliers_matches_idx, outliers_matches_idx


def get_matches_coor(matches, img1_kpts, img2_kpts):
    """
    Returns 2 numpy arrays of matches coordinates in img1 and img2 accordingly
    """
    img1_matches_coor, img2_matches_coor = [], []
    for match in matches:
        img1_idx, img2_idx = match.queryIdx, match.trainIdx
        img1_matches_coor.append(img1_kpts[img1_idx].pt)
        img2_matches_coor.append(img2_kpts[img2_idx].pt)

    return np.array(img1_matches_coor), np.array(img2_matches_coor)


def get_inliers_and_outliers_coor_for_rec(inliers_matches, outliers_matches, img1_kpts, img2_kpts):
    img1_inliers, img2_inliers = get_matches_coor(inliers_matches, img1_kpts, img2_kpts)
    img1_outliers, img2_outliers = get_matches_coor(outliers_matches, img1_kpts, img2_kpts)
    return img1_inliers, img2_inliers, img1_outliers, img2_outliers


def linear_least_square(l_cam_mat, r_cam_mat, kp1_xy, kp2_xy):
    """
    Linear least square procedure.
    :param l_cam_mat: Left camera matrix
    :param r_cam_mat: Right camera matrix
    :param kp1_xy: (x,y) for key point 1
    :param kp2_xy: (x,y) for key point 2
    :return: Solution for the equation Ax = 0
    """
    # Compute the matrix A
    mat = np.array([kp1_xy[0] * l_cam_mat[2] - l_cam_mat[0],
                    kp1_xy[1] * l_cam_mat[2] - l_cam_mat[1],
                    kp2_xy[0] * r_cam_mat[2] - r_cam_mat[0],
                    kp2_xy[1] * r_cam_mat[2] - r_cam_mat[1]])

    # Calculate A's SVD
    u, s, vh = np.linalg.svd(mat, compute_uv=True)

    # Last column of V is the result as a numpy object
    return vh[-1]


def triangulate(l_mat, r_mat, kp1_xy_lst, kp2_xy_lst):  # Todo : Efficiency in numpy and array create
    """
    Apply triangulation procedure
    :param l_mat: Left camera matrix
    :param r_mat: Right camera matrix
    :return: List of 3d points in the world
    """
    kp_num = len(kp1_xy_lst)
    res = []
    for i in range(kp_num):
        p4d = linear_least_square(l_mat, r_mat, kp1_xy_lst[i], kp2_xy_lst[i])
        p3d = p4d[:3] / p4d[3]
        res.append(p3d)
    return np.array(res)


def draw_triangulations(p3d_pts, cv_p3d_pts):
    """
    Draws the 3d points triangulations (Our and open-cv)
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    rows, cols = 1, 2
    fig.suptitle(f'Open-cv and our Triangulations compare')

    # Our results
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    ax.set_title("Our Result")
    ax.scatter3D(0, 0, 0, c='red', s=40)  # Camera
    ax.scatter3D(p3d_pts[:, 0], p3d_pts[:, 1], p3d_pts[:, 2])
    ax.set_xlim3d(-20, 10)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(-100, 200)

    # Cv results
    ax = fig.add_subplot(rows, cols, 2, projection='3d')
    ax.set_title("Open-cv result")
    ax.scatter3D(0, 0, 0, c='red', s=40)  # Camera
    ax.scatter3D(cv_p3d_pts[:, 0], cv_p3d_pts[:, 1], cv_p3d_pts[:, 2])
    ax.set_xlim3d(-20, 10)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(-100, 200)

    fig.savefig(f"VAN_ex/triangulations plot.png")
    plt.close(fig)

# == Ex3 ==
