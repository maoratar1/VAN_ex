import pickle
import random

import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.patches import ConnectionPatch

import utills
import pandas as pd

TRACK_NAME = "TRACK"
FRAME_NAME = "FRAME"

DB_TRACK_PATH = r'VAN_ex/Track frames.csv'
DB_FRAME_PATH = r'VAN_ex/Frame tracks.csv'
STAT_TRACK_FRAME_PATH = r'VAN_ex/Track and frames stat.csv'


def compute_kpt_and_coor_whole_movie():
    """
    Compute the transformation of two consequence left images in the whole movie
    :return: Array which each row contains 2 arrays [frame0_features, frame1_features]
     frame0_features contains Feature objects that match to frame1_features, and they are sharing
     indexes. Meaning that ith feature at frame0_features are match to ith frame1_features
    """

    prev_and_next_frame_features = []

    # Find matches in pair0 with rectified test
    left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = \
                                                                          utills.read_and_rec_match(utills.IDX)
    print("Pair 0")
    inliers_percentage = []
    frame1_features = []
    for i in range(1, utills.MOVIE_LEN):
        if i % 100 == 0:
            print("Pair ", i)
        left1_kpts, left1_dsc, right1_kpts, pair1_matches, pair1_rec_matches_idx = \
                                                                     utills.read_and_rec_match(utills.IDX + i)

        frame0_features, frame1_features = compute_kpts_and_coor(left0_kpts, left0_dsc, right0_kpts,
                                                                       pair0_matches, pair0_rec_matches_idx,
                                                                       left1_kpts, left1_dsc, right1_kpts,
                                                                       pair1_matches, pair1_rec_matches_idx)

        inliers_percentage.append(100 * len(frame0_features) / len(pair0_matches))

        left0_kpts, left0_dsc, right0_kpts, pair0_matches, pair0_rec_matches_idx = left1_kpts, left1_dsc, \
                                                                                   right1_kpts, pair1_matches, \
                                                                                   pair1_rec_matches_idx
        prev_and_next_frame_features.append([frame0_features, frame1_features])

    inliers_percentage.append(100 * len(frame1_features) / len(pair0_matches))

    return prev_and_next_frame_features, inliers_percentage


def compute_kpts_and_coor(left0_kpts, left0_dsc, right0_kpts,
                          pair0_matches, pair0_rec_matches_idx,
                          left1_kpts, left1_dsc, right1_kpts,
                          pair1_matches, pair1_rec_matches_idx):
    """
   Compute the transformation T between left 0 and left1 images
   :return: Numpy array of Feature object from frame0 and frame1 that passed the consensus match
   """

    # Find matches between left0 and left1
    left0_left1_matches = utills.matching(left0_dsc, left1_dsc)

    # Find key pts that match in all 4 images
    rec0_dic = utills.create_rec_dic(pair0_matches, pair0_rec_matches_idx)  # dict of {left kpt idx: pair rec id}
    rec1_dic = utills.create_rec_dic(pair1_matches, pair1_rec_matches_idx)
    q_pair0_idx, q_pair1_idx, q_left0_left1_idx = utills.find_kpts_in_all_4_rec(left0_left1_matches, rec0_dic, rec1_dic)

    # Get frame 0 Feature objects (which passed the rec test)
    frame0_features = utills.get_feature_obj(pair0_matches[q_pair0_idx], left0_kpts, right0_kpts)

    # Here we take only their coordinates
    left0_matches_coor = utills.get_features_left_coor(frame0_features)
    right0_matches_coor = utills.get_features_right_coor(frame0_features)

    # Frame 0 triangulation
    pair0_p3d_pts = utills.triangulate(utills.K @ utills.M1, utills.K @ utills.M2, left0_matches_coor,
                                       right0_matches_coor)

    # Get frame 1 Feature objects  (which passed the rec test)
    frame1_features = utills.get_feature_obj(pair1_matches[q_pair1_idx], left1_kpts, right1_kpts)

    # Notice that frame0_features and frame1_features are sharing indexes

    # Here we take only their coordinates
    left1_matches_coor = utills.get_features_left_coor(frame1_features)
    right1_matches_coor = utills.get_features_right_coor(frame1_features)

    # Finds Feature's indexes of frame0_features and frame1_features that passed the consensus match
    # with using Ransac method
    _, max_supp_group_idx = utills.online_est_pnp_ransac(utills.PNP_NUM_PTS, pair0_p3d_pts,
                                                                          utills.M1, left0_matches_coor,
                                                                          utills.M2, right0_matches_coor,
                                                                          left1_matches_coor,
                                                                          utills.M2, right1_matches_coor,
                                                                          utills.K, acc=utills.SUPP_ERR)

    return frame0_features[max_supp_group_idx], frame1_features[max_supp_group_idx]


# === Missions === #
def mission2(db):
    """
    At this mission we compute the following Tracking statistics:
    1. Total tracks number
    2. Frames number
    3. Max track length
    4. Min track length
    5. Mean track length
    6. Tracks number in average frame
    :param db: Data base
    """

    total_tracks_num = db.get_num_tracks()
    frames_number = db.get_num_frames()
    max_track_len, min_track_len, mean_track_len = compute_track_stat(db.get_tracks())
    mean_frames_track_len = compute_mean_frames_track(db.get_frames_idxes())
    data = {'Statistics': [total_tracks_num, frames_number, max_track_len, min_track_len, mean_track_len,
                           mean_frames_track_len]}

    df = pd.DataFrame(data, index=['Total tracks number', 'Frames number', 'Max track length', 'Min track length',
                                   'Mean track length', 'Tracks number in average frame'])
    df.to_csv(STAT_TRACK_FRAME_PATH)


def compute_track_stat(tracks):
    """
    Compute the following Tracking statistics:
    1. Max track length
    2. Min track length
    3. Mean track length
    :param db: Data base
    """
    min_track_len, max_track_len, mean_track_len = utills.MOVIE_LEN, 0, 0

    for track in tracks:
        track_len = track.get_track_len()
        max_track_len = max(max_track_len, track_len)
        min_track_len = min(min_track_len, track_len)
        mean_track_len += track_len

    mean_track_len = mean_track_len / len(tracks)
    return max_track_len, min_track_len, format(mean_track_len, ".2f")


def compute_mean_frames_track(frames):
    """
    Compute the following Tracking statistics:
    1. Tracks number in average frame
    :param frames: List of frames object
    """
    mean_frames_track = 0
    for frame in frames:
        mean_frames_track += frame.get_tracks_num()

    mean_frames_track = mean_frames_track / len(frames)
    return mean_frames_track


def mission3(db, track_len=10):
    """
    At this mission we randomize a track with length of track_len and plots the track on frames
    :param db: Data base
    :return:
    """
    track = get_rand_track(track_len, db.get_tracks())
    plot_track(track)


def get_rand_track(track_len, tracks):
    """
    Randomize a track with length of track_len
    """
    track = None
    found = False
    while not found:
        idx = random.randint(0, len(tracks) - 1)
        if tracks[idx].get_track_len() == track_len:
            track = tracks[idx]
            found = True

    return track


def plot_track(track):
    """
    Plots a track all over it's frames with a line connecting the track
    :param track:
    :return:
    """
    track_len = track.get_track_len()
    track_id = track.get_id()
    frames_coor = track.get_frames_features_dict()

    # rows, cols = len(frames_coor) // 3, 3
    rows, cols = track_len, 2
    crop_size = 100

    fig, ax_arr = plt.subplots(rows, cols)
    fig.set_figwidth(4)
    fig.set_figheight(12)

    fig.suptitle(f"Track(id:{track_id}) with len {track_len}\n Image cropped to {crop_size}X{crop_size}")
    plt.rcParams["font.size"] = "10"
    plt.subplots_adjust(wspace=-0.55, hspace=0.1)

    ax_arr[0, 0].set_title("Left")
    ax_arr[0, 1].set_title("Right")

    prev_l_coor, prev_left, prev_l_ax = None, None, None
    prev_r_coor, prev_right, prev_r_ax = None, None, None

    i = 0

    for frame in frames_coor:
        feature = frames_coor[frame]

        l_coor = feature.get_left_coor()
        r_coor = feature.get_right_coor()

        left_img, right_img = utills.read_images(utills.IDX + frame)
        cropped_left_img, l_coor = crop_image(l_coor, left_img, crop_size)
        cropped_right_img, r_coor = crop_image(r_coor, right_img, crop_size)

        ax_arr[i, 0].axes.xaxis.set_visible(False)
        ax_arr[i, 0].axes.yaxis.set_label_text(f"frame {frame}")
        ax_arr[i, 0].set_yticklabels([])
        ax_arr[i, 0].imshow(cropped_left_img, cmap='gray')
        ax_arr[i, 0].scatter(l_coor[0], l_coor[1], c='red', s=10)

        ax_arr[i, 1].axes.xaxis.set_visible(False)
        ax_arr[i, 1].set_yticklabels([])
        ax_arr[i, 1].imshow(cropped_right_img, cmap='gray')
        ax_arr[i, 1].scatter(r_coor[0], r_coor[1], c='red', s=10)

        if i == 0:
            prev_l_coor, prev_left, prev_l_ax = l_coor, cropped_left_img, ax_arr[i, 0]
            prev_r_coor, prev_right, prev_r_ax = r_coor, cropped_right_img, ax_arr[i, 1]

        left0_left1_line = ConnectionPatch(xyA=(prev_l_coor[0], prev_l_coor[1]),
                                           xyB=(l_coor[0], l_coor[1]),
                                           coordsA="data", coordsB="data",
                                           axesA=prev_l_ax, axesB=ax_arr[i, 0], color="cyan")

        right0_right1_line = ConnectionPatch(xyA=(prev_r_coor[0], prev_r_coor[1]),
                                             xyB=(r_coor[0], r_coor[1]),
                                             coordsA="data", coordsB="data",
                                             axesA=prev_r_ax, axesB=ax_arr[i, 1], color="cyan")

        ax_arr[i, 0].add_artist(left0_left1_line)
        ax_arr[i, 1].add_artist(right0_right1_line)

        prev_l_coor, prev_left, prev_l_ax = l_coor, cropped_left_img, ax_arr[i, 0]
        prev_r_coor, prev_right, prev_r_ax = r_coor, cropped_right_img, ax_arr[i, 1]

        i += 1

    fig.savefig("VAN_ex/track.png")
    plt.close(fig)


def crop_image(coor, img, crop_size):
    """
    Crops image "img" to size of "crop_size" X "crop_size" around the coordinates "coor"
    :return: Cropped image
    """
    r_x = int(min(utills.IMAGE_WIDTH, coor[0] + crop_size))
    l_x = int(max(0, coor[0] - crop_size))
    u_y = int(max(0, coor[1] - crop_size))
    d_y = int(min(utills.IMAGE_HEIGHT, coor[1] + crop_size))

    return img[u_y: d_y, l_x: r_x], [crop_size, crop_size]


def mission4(db):
    """
    At this mission we compute the outgoing track for each frame and plots them as a graph
    :param db: Data base
    """
    outgoing_tracks_num_per_frame = connectivity_graph(db, utills.MOVIE_LEN)
    plot_connectivity_graph(outgoing_tracks_num_per_frame, utills.MOVIE_LEN)


def connectivity_graph(db, frames_num):
    """
    Computes the outgoing track for each frame
    :param db: Database
    :param frames_num: Last frame to apply this function on
    """
    outgoing_tracks_num_per_frame = []
    tracks = db.get_tracks()
    frames = db.get_frames_idxes()

    for i in range(frames_num):
        frame = frames[i]
        val = 0

        for track_id in frame.get_tracks_ids():
            track = tracks[track_id]
            if frame.get_id() + 1 in track.get_frames_features_dict():  # Means it continues to the next frame
                val += 1

        outgoing_tracks_num_per_frame.append(val)

    return outgoing_tracks_num_per_frame


def plot_connectivity_graph(vals_per_frame, frames_num):
    """
    Plots the connectivity graph
    :param vals_per_frame:
    :param frames_num:
    :return:
    """
    x = range(len(vals_per_frame))

    f = plt.figure()
    f.set_figwidth(14)
    f.set_figheight(7)

    # plotting the points
    plt.plot(x, vals_per_frame)

    plt.xlabel('frames')
    plt.ylabel('Outgoing tracks')
    plt.title(f'Connectivity for {frames_num} frames')

    plt.savefig("VAN_ex/Connectivity.png")


def mission5(db):
    """
    At this mission we plot the inliers percentage in each frame
    :param db: Database
    """
    plot_inliers_percentage_graph(db.get_inliers_percentage_per_frame())


def plot_inliers_percentage_graph(inliers_percentage):
    """
    Plots the inliers percentage in each frame
    :param inliers_percentage:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_title("Inliers percentage per frame")
    plt.plot(inliers_percentage)
    plt.ylabel('Percentage')
    plt.xlabel('Frame')

    fig.savefig("VAN_ex/Inliers percentage per frame graph.png")
    plt.close(fig)


def mission6(db):
    """
    At this mission we compute a histogram of tracks length's
    :param db: Database
    """
    track_lengths = []
    for track in db.get_tracks():
        track_lengths.append(track.get_track_len())

    plot_hist(track_lengths)


def plot_hist(track_lengths):
    """
    Plot the histogram
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    hist_track_lengths, _, _ = plt.hist(track_lengths)
    # plt.clf()

    ax.set_title("Track length histogram")
    plt.plot(hist_track_lengths)
    plt.ylabel('Track #')
    plt.xlabel('Track lengths')

    fig.savefig("VAN_ex/Track length histogram.png")
    plt.close(fig)


def mission7(db, frame_idx_triangulate, track_len=10):
    """
    At this mission we:
    1. Randomize a track with len of track_len
    2. Triangulate a point from the "frame_idx_triangulate" frame in the track
    3. Projects it to all frames in the track
    4. Computes their reprojection error on each frame
    5. Finally, plot the results
    :param db: Database
    :param frame_idx_triangulate: frame's index to triangulate from
    """
    track = get_rand_track(track_len, db.get_tracks())
    track_frames_ids = track.get_frames_ids()

    ground_truth_trans = utills.get_ground_truth_transformations()[track_frames_ids[0]: track_frames_ids[-1] + 1]

    left_locations = track.get_left_locations_in_all_frames()
    right_locations = track.get_right_locations_in_all_frames()

    last_left_img_coor = left_locations[frame_idx_triangulate]
    last_right_img_coor = right_locations[frame_idx_triangulate]

    last_left_trans = ground_truth_trans[frame_idx_triangulate]
    last_left_proj_mat = utills.K @ last_left_trans
    last_right_proj_mat = utills.K @ utills.compose_transformations(last_left_trans, utills.M2)
    p34 = utills.triangulate(last_left_proj_mat, last_right_proj_mat, [last_left_img_coor], [last_right_img_coor])

    left_projections = []
    right_projections = []

    for trans in ground_truth_trans:
        left_proj_cam = utills.K @ trans
        right_proj_cam = utills.K @ utills.compose_transformations(trans, utills.M2)

        left_proj = utills.project(p34, left_proj_cam)[0]
        right_proj = utills.project(p34, right_proj_cam)[0]

        left_projections.append(left_proj)
        right_projections.append(right_proj)

    left_proj_dist = utills.compute_reprojection_euclidean_dist(np.array(left_projections), np.array(left_locations))
    right_proj_dist = utills.compute_reprojection_euclidean_dist(np.array(right_projections), np.array(right_locations))
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2

    plot_reprojection_error_graph(total_proj_dist, frame_idx_triangulate)


def plot_reprojection_error_graph(total_proj_dist, frame_idx_triangulate):
    """
    Plots reprojection error
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    frame_title = "Last" if frame_idx_triangulate == -1 else "First"

    ax.set_title(f"Re projection error from {frame_title} frame")
    plt.scatter(range(len(total_proj_dist)), total_proj_dist)
    plt.ylabel('Error')
    plt.xlabel('Frames')

    fig.savefig(f"VAN_ex/Re projection error graph for {frame_title} frame.png")
    plt.close(fig)