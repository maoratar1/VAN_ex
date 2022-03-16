import random
import cv2
import matplotlib.pyplot as plt

# == Constants == #
DATA_PATH = r'/Users/maoratar/opt/anaconda3/envs/Van_Ex1/VAN_ex/dataset/sequences/00/'
IDX = 000000
MATCHES_NUM = 20
RATIO = 0.7
MATCHES_NORM = cv2.NORM_L2
PASSED = "PASSED"
FAILED = "FAILED"


# == Functions == #

    # == Main functions == #

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
    img1_kpts, img1_dsc = alg.detectAndCompute(img1, None)
    img2_kpts, img2_dsc = alg.detectAndCompute(img2, None)
    return img1_kpts, img1_dsc, img2_kpts, img2_dsc


def draw_keypts(img1, img1_kpts, img2, img2_kpts):
    img1_with_kpts = cv2.drawKeypoints(img1, img1_kpts, None, color=(0, 255, 0), flags=0)
    img2_with_kpts = cv2.drawKeypoints(img2, img2_kpts, None, color=(0, 255, 0), flags=0)
    show_images(img1_with_kpts, img2_with_kpts)


def print_descriptor(img1_first_desc, img2_first_desc):
    """
    Print descriptors values
    """
    print("Image 1 first descriptor:\n", img1_first_desc)
    print("Image 2 first descriptor:\n", img2_first_desc)


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


def significance_test(img1_dsc, img2_dsc, ratio, rand):
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
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(img1_dsc, img2_dsc, k=2)
    if rand:
        matches = random.choices(matches, k=MATCHES_NUM)
    else:
        matches = matches[:MATCHES_NUM]

    # Apply ratio test
    pass_test = []
    failed_test = []
    for first, second in matches:
        if first.distance < ratio * second.distance:
            pass_test.append([first])
        else:
            failed_test.append([first])

    return pass_test, failed_test


def draw_matches_of_sig_test(img1, img1_kpts, img2, img2_kpts, test_pts, matches_num, test_desc):
    """
    Plots the matches that pass/failed the significance test
    :param test_pts: points that failed / passed the significance test
    """

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, img1_kpts, img2, img2_kpts, test_pts, None, flags=2)

    plt.title(f'Points that {test_desc} the significance test (Ratio: {RATIO}).'
              f'\nNum matches:{len(test_pts)}/{matches_num}')
    plt.imshow(img3)
    plt.show()


def print_num_matches_diff_ratio(img1_dsc, img2_dsc):
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("Matches amount that passed/failed the significance test with the ratio:")
    print(f"Ratio\tpassed\tfailed")
    for ratio in ratios:
        pass_test, fail_test = significance_test(img1_dsc, img2_dsc, ratio, rand=True)
        print(f"{ratio}\t\t{len(pass_test)}\t\t{len(fail_test)}")


if __name__ == '__main__':
    # Reads the two images
    img1, img2 = read_images(IDX)

    # Detects the image key points and compute their descriptors
    alg = cv2.AKAZE_create()
    img1_kpts, img1_dsc, img2_kpts, img2_dsc = feature_detection_and_description(img1, img2, alg)
    draw_keypts(img1, img1_kpts, img2, img2_kpts)

    # Matches between the images and plots the matching
    matches = matching(MATCHES_NORM, img1_dsc, img2_dsc)
    rand_matches = random.choices(matches, k=MATCHES_NUM)
    draw_matches(img1, img1_kpts, img2, img2_kpts, rand_matches)

    # Apply the significance test
    pass_test, fail_test = significance_test(img1_dsc, img2_dsc, RATIO, rand=True)
    # Plots the pts that passed the test
    draw_matches_of_sig_test(img1, img1_kpts, img2, img2_kpts, pass_test, MATCHES_NUM, PASSED)
    # Plots the pts that failed the test
    draw_matches_of_sig_test(img1, img1_kpts, img2, img2_kpts, fail_test, MATCHES_NUM, FAILED)

