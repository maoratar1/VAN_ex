import cv2
import numpy as np


DATA_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/sequences/00/'
IDX = 000000
MATCHES_NUM = 20
RATIO = 0.35
MATCHES_NORM = cv2.NORM_L2
PASSED = "PASSED"
FAILED = "FAILED"


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
    :return: images keypoints and descriptors
    """
    img1_kpts, img1_dsc = alg.detectAndCompute(img1, None)
    img2_kpts, img2_dsc = alg.detectAndCompute(img2, None)
    return img1_kpts, img1_dsc, img2_kpts, img2_dsc


def matching(metric, img1_dsc, img2_dsc):
    """
    Find Matches between two images descriptors
    :param metric: distance function for computes distance between two descriptors
    :param img1_dsc: image 1 descriptors
    :param img2_dsc: image 2 descriptors
    :return: List of matches
    """
    bf = cv2.BFMatcher(metric, crossCheck=True)
    matches = bf.match(img1_dsc, img2_dsc)
    # Sort the matches from the best match to the worst - where best means it has the lowest distance
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def read_detect_and_match(img1, img2):
    # Detects the image key points and compute their descriptors
    alg = cv2.AKAZE_create()
    img1_kpts, img1_dsc, img2_kpts, img2_dsc = feature_detection_and_description(img1, img2, alg)
    # draw_keypts(img1, img1_kpts, img2, img2_kpts)

    # Matches between the images and plots the matching
    matches = matching(MATCHES_NORM, img1_dsc, img2_dsc)
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
