import random
import cv2
import matplotlib.pyplot as plt

# == Constants == #
DATA_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/sequences/00/'
IDX = 000000
NUM_MATCHES = 20
RATIO = 0.9
MATCHES_NORM = cv2.NORM_L2

# == Functions == #
def read_images(idx):
    """
    :param idx: Images's index in the Kitti dataset
    :return: left and right cameras photos
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    return img1, img2


def show_images(img1, img2):
    """
    Plots images
    """
    fx, plots = plt.subplots(1, 2, figsize=(20, 10))

    plt.suptitle('Main title')
    plots[0].set_title("Left Image")
    plots[0].imshow(img1, cmap='gray')

    plots[1].set_title("Right Image")
    plots[1].imshow(img2, cmap='gray')
    plt.show()


def feature_detection_and_description(img1, img2, alg):
    """
    Computes images key points and their's descriptors
    :param alg: Feature detecting and description algorithm
    :return: images keypoints and descriptors
    """
    img1_keypoints, img1_desc = alg.detectAndCompute(img1, None)
    img2_keypoints, img2_desc = alg.detectAndCompute(img2, None)
    return img1_keypoints, img1_desc, img2_keypoints, img2_desc


def draw_keypts(img1, img1_keypoints, img2, img2_keypoints):
    img1_with_kpts = cv2.drawKeypoints(img1, img1_keypoints, None, color=(0, 255, 0), flags=0)
    img2_with_kpts = cv2.drawKeypoints(img2, img2_keypoints, None, color=(0, 255, 0), flags=0)
    show_images(img1_with_kpts, img2_with_kpts)


def print_descriptor(img1_first_desc, img2_first_desc):
    """
    Print descriptors values
    """
    print("Image 1 first descriptor:\n", img1_first_desc)
    print("Image 2 first descriptor:\n", img2_first_desc)


def matching(metric, img1_desc, img2_desc):
    """
    Find Matches between two images descriptors
    :param metric: distance function for computes distance between two descriptors
    :param img1_desc: image 1 descriptors
    :param img2_desc: image 2 descriptors
    :return: List of matches
    """
    bf = cv2.BFMatcher(metric, crossCheck=True)
    matches = bf.match(img1_desc, img2_desc)
    # Sort the matches from the best match to the worst - where best means it has the lowest distance
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def draw_matches(img1, img1_keypoints, img2, img2_keypoints, matches):
    """
    Draws a line between matches in img1 and img2
    """
    result = cv2.drawMatches(img1, img1_keypoints, img2, img2_keypoints, matches, None, flags=2)
    # Display the best matching points
    plt.rcParams['figure.figsize'] = [20.0, 10.0]
    plt.title(f'Best Matching Points. Num matches:{len(matches)}')
    plt.imshow(result)
    plt.show()


def significance_test(img1, img1_desc, img1_keypoints, img2, img2_desc, img2_keypoints):
    """
    Applying the significance test - rejects all matches that their distance ratio between the 1st and 2nd
    nearest neighbors is lower than predetermined RATIO.
    In practice:
        1. Finds 2 best matches in img2 for each descriptor in img1.
        2. Rejects all the matches that are not passing the test.
        3. Shows the new matches - that passed the test.

    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(img1_desc, img2_desc, k=2)
    rand_matches = random.choices(matches, k=NUM_MATCHES)

    # Apply ratio test
    pass_test = []
    for m, n in rand_matches:
        if m.distance < RATIO * n.distance:
            pass_test.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, img1_keypoints, img2, img2_keypoints, pass_test, None, flags=2)

    plt.title(f'Points that pass the significance test(Ratio: {RATIO}).'
              f' Num matches:{len(pass_test)}/{len(rand_matches)}')
    plt.imshow(img3)
    plt.show()


if __name__ == '__main__':
    # Reads the two images
    img1, img2 = read_images(IDX)

    # Detects the image key points and compute their descriptors
    alg = cv2.AKAZE_create()
    img1_keypoints, img1_desc, img2_keypoints, img2_desc = feature_detection_and_description(img1, img2, alg)

    # Matches between the images and plots the matching
    matches = matching(MATCHES_NORM, img1_desc, img2_desc)
    rand_matches = random.choices(matches, k=NUM_MATCHES)
    draw_matches(img1, img1_keypoints, img2, img2_keypoints, rand_matches)

    # Apply the significance test
    significance_test(img1, img1_desc, img1_keypoints, img2, img2_desc, img2_keypoints)
