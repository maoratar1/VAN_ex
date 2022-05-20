import gtsam
import numpy as np
from gtsam import symbol
import tqdm

from DataDirectory import Data
from utils import utills

LAND_MARK_SYM = "q"
CAMERA_SYM = "c"
P3D_MAX_DIST = 80


class BundleWindow:

    BUNDLE_WINDOW_LST = []

    def __init__(self, first_key_frame, second_key_frame, frames_in_window):  # Todo: add Rotation
        self.__first_key_frame = first_key_frame
        self.__second_key_frame = second_key_frame
        self.__bundle_len = 1 + second_key_frame - first_key_frame
        self.__frames_in_window = frames_in_window
        self.__computed_tracks = set()

        self.__optimizer = None
        self.__initial_estimate = gtsam.Values()
        self.__optimized_values = None
        self.__camera_sym = set()  # {CAMERA_SYM + frame index : gtsam symbol} for example {"q1": symbol(q1, place)}
        self.__landmark_sym = set()

        self.__graph = gtsam.NonlinearFactorGraph()

    def create_factor_graph(self):
        #Todo: 1. Consider change this such that it will get the all tracks in the frame in the bundle and run all over them
        # instead of receive a track from each frame
        # 2. Check the change of getting the relative trans array instead of the relatiive to first
        gtsam_calib_mat = utills.create_gtsam_calib_cam_mat(utills.K)

        first_frame = self.__frames_in_window[0]

        # Compute first frame extrinsic matrix that takes a point at the camera coordinates and map it to
        # the world coordinates where "world" here means frame 0 of the whole movie coordinates
        first_frame_cam_to_world_ex_mat = utills.convert_ex_cam_to_cam_to_world(first_frame.get_ex_cam_mat())  # first cam -> world

        # For each frame - create initial estimation for it's pose
        cur_cam_pose = None
        for i, frame in enumerate(self.__frames_in_window):

            # Create camera symbol and update values dictionary
            left_pose_sym = symbol(CAMERA_SYM, frame.get_id())
            self.__camera_sym.add(left_pose_sym)

            # Initialize constraints for first pose
            if i == 0:
                # sigmas array: first 3 for angles second 3 for location
                # sigmas = np.array([10 ** -3, 10 ** -3, 10 ** -3, 10 ** -2, 10 ** -2, 10 ** -2])
                sigmas = np.array([(3 * np.pi / 180)**2] * 3 + [1.0, 0.3, 1.0])
                pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)  # todo: check choice of diagonal
                # Initial pose
                factor = gtsam.PriorFactorPose3(left_pose_sym, gtsam.Pose3(), pose_uncertainty)
                self.__graph.add(factor)

            # Compute transformation of : (world - > cur cam) * (first cam -> world) = first cam -> cur cam
            camera_relate_to_first_frame_trans = utills.compose_transformations(first_frame_cam_to_world_ex_mat, frame.get_ex_cam_mat())

            # Convert this transformation to: cur cam -> first cam
            cur_cam_pose = utills.convert_ex_cam_to_cam_to_world(camera_relate_to_first_frame_trans)
            self.__initial_estimate.insert(left_pose_sym, gtsam.Pose3(cur_cam_pose))

        gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)

        # For each track create measurements factors
        tracks_in_frame = Data.DB.get_tracks_at_frame(first_frame.get_id())

        for track in tracks_in_frame:
            # Check that this track has bot been computed yet and that it's length is satisfied
            if track.get_id() in self.__computed_tracks or track.get_last_frame_ind() < self.__second_key_frame:
                continue

            # Create a gtsam object for the last frame for making the projection at the function "add_factors"
            gtsam_last_cam = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)
            self.add_factors(track, self.__first_key_frame, self.__second_key_frame, gtsam_last_cam, gtsam_calib_mat)  # Todo: as before

            self.__computed_tracks.add(track.get_id())


    # def create_factor_graph(self):
    #     # Todo: 1. Check this with rel trans
    #     # instead of receive a track from each frame
    #     # 2. Check the change of getting the relative trans array instead of the relatiive to first
    #     gtsam_calib_mat = utills.create_gtsam_calib_cam_mat(utills.K)
    #
    #     first_frame = self.__frames_in_window[0]
    #
    #     # Compute first frame extrinsic matrix that takes a point at the camera coordinates and map it to
    #     # the world coordinates where "world" here means frame 0 of the whole movie coordinates
    #     rel_trans = Data.DB.get_relative_cam_trans()[self.__first_key_frame: self.__second_key_frame + 1]
    #     cams_rel_to_bundle_first_cam = utills.convert_trans_from_rel_to_global(rel_trans)
    #
    #     # For each frame - create initial estimation for it's pose
    #     cur_cam_pose = None
    #     for i, frame in enumerate(self.__frames_in_window):
    #
    #         # Create camera symbol and update values dictionary
    #         left_pose_sym = symbol(CAMERA_SYM, frame.get_id())
    #         self.__camera_sym.add(left_pose_sym)
    #
    #         # Initialize constraints for first pose
    #         if i == 0:
    #             # sigmas array: first 3 for angles second 3 for location
    #             # sigmas = np.array([10 ** -3, 10 ** -3, 10 ** -3, 10 ** -2, 10 ** -2, 10 ** -2])
    #             sigmas = np.array([(3 * np.pi / 180) ** 2] * 3 + [1.0, 0.3, 1.0])
    #             pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)  # todo: check choice of diagonal
    #             # Initial pose
    #             factor = gtsam.PriorFactorPose3(left_pose_sym, gtsam.Pose3(), pose_uncertainty)
    #             self.__graph.add(factor)
    #
    #         # Convert this transformation to: cur cam -> first cam
    #         cur_cam_pose = utills.convert_ex_cam_to_cam_to_world(cams_rel_to_bundle_first_cam[i])
    #         self.__initial_estimate.insert(left_pose_sym, gtsam.Pose3(cur_cam_pose))
    #
    #     gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)
    #
    #     # For each track create measurements factors
    #     tracks_in_frame = Data.DB.get_tracks_at_frame(first_frame.get_id())
    #
    #     for track in tracks_in_frame:
    #         # Check that this track has bot been computed yet and that it's length is satisfied
    #         if track.get_id() in self.__computed_tracks or track.get_last_frame_ind() < self.__second_key_frame:
    #             continue
    #
    #         # Create a gtsam object for the last frame for making the projection at the function "add_factors"
    #         gtsam_last_cam = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)
    #         self.add_factors(track, self.__first_key_frame, self.__second_key_frame, gtsam_last_cam,
    #                          gtsam_calib_mat)  # Todo: as before
    #
    #         self.__computed_tracks.add(track.get_id())


    def add_factors(self, track, first_frame_ind, last_frame_ind, gtsam_frame_to_triangulate_from, gtsam_calib_mat, frame_idx_triangulate=-1):
        """
        At this mission we:
        1. Randomize a track with len of track_len
        2. Triangulate a point from the "frame_idx_triangulate" frame in the track
        3. Projects it to all frames_in_window in the track
        4. Computes their re projection error on each frame
        5. Finally, plot the results
        """

        frames_in_track = Data.DB.get_frames()[first_frame_ind: last_frame_ind + 1]

        # Track's locations in frames_in_window
        left_locations = track.get_left_locations_in_specific_frames(first_frame_ind, last_frame_ind)
        right_locations = track.get_right_locations_in_specific_frames(first_frame_ind, last_frame_ind)

        # Track's location at the Last frame for triangulations
        last_left_img_loc = left_locations[frame_idx_triangulate]
        last_right_img_loc = right_locations[frame_idx_triangulate]

        # Create Measures of last frame for the triangulation
        measure_xl, measure_xr, measure_y = last_left_img_loc[0], last_right_img_loc[0], last_left_img_loc[1]
        gtsam_stereo_point2_for_triangulation = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

        # Triangulation from last frame
        gtsam_p3d = gtsam_frame_to_triangulate_from.backproject(gtsam_stereo_point2_for_triangulation)

        # Ignore tracks that their 3d point is far  Todo: Maybe to add it with more than 100 - last check wasnt good
        # if utils.euclidean_dist_3d(gtsam_p3d, gtsam_frame_to_triangulate_from.pose().translation()) >= 100:
        #     return

        # Add landmark symbol to "values" dictionary
        p3d_sym = symbol(LAND_MARK_SYM, track.get_id())
        self.__landmark_sym.add(p3d_sym)
        self.__initial_estimate.insert(p3d_sym, gtsam_p3d)

        for i, frame in enumerate(frames_in_track):

            # Measurement values
            measure_xl, measure_xr, measure_y = left_locations[i][0], right_locations[i][0], left_locations[i][1]
            gtsam_measurement_pt2 = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

            # Factor creation
            projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
            factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                                 symbol(CAMERA_SYM, frame.get_id()), p3d_sym, gtsam_calib_mat)

            # Add factor to the graph
            self.__graph.add(factor)

    def optimize(self):
        self.__optimizer = gtsam.LevenbergMarquardtOptimizer(self.__graph, self.__initial_estimate)
        self.__optimized_values = self.__optimizer.optimize()

    def update_optimization(self, values):
        self.__initial_estimate = values

    def graph_error(self, optimized=True):
        if not optimized:
            error = self.__graph.error(self.__initial_estimate)
        else:
            error = self.__graph.error(self.__optimized_values)

        return np.log(error)  # Todo: here we returns the reprojection error probably

    def get_initial_estimate_values(self):
        return self.__initial_estimate

    def get_optimized_values(self):
        return self.__optimized_values

    def get_cameras_symbols_lst(self):
        return self.__camera_sym

    def get_landmarks_symbols_lst(self):
        return self.__landmark_sym

    def get_optimized_cameras_poses(self):
        cameras_poses = []
        for camera_sym in self.__camera_sym:
            cam_pose = self.__optimized_values.atPose3(camera_sym)
            cameras_poses.append(cam_pose)

        return cameras_poses

    def get_optimized_cameras_p3d_version2(self):
        cameras_poses = []
        for camera_sym in self.__camera_sym:
            cam_pose = self.__optimized_values.atPose3(camera_sym)
            cameras_poses.append([cam_pose.x(), cam_pose.y(), cam_pose.z()])

        return cameras_poses

    def get_optimized_cameras_p3d(self):
        cam_pose = self.__optimized_values.atPose3(symbol(CAMERA_SYM, self.__second_key_frame))
        return cam_pose

    def get_optimized_landmarks_p3d(self):
        landmarks = []
        for landmark_sym in self.__landmark_sym:
            landmark = self.__optimized_values.atPoint3(landmark_sym)
            landmarks.append(landmark)

        return landmarks

    def get_initial_estimate_cameras_p3dversion2(self):
        cameras_poses = []
        for camera_sym in self.__camera_sym:
            cam_pose = self.__initial_estimate.atPose3(camera_sym)
            cameras_poses.append([cam_pose.x(), cam_pose.y(), cam_pose.z()])

        return cameras_poses

    def get_initial_estimate_cameras_p3d(self):
        cam_pose = self.__initial_estimate.atPose3(symbol(CAMERA_SYM, self.__second_key_frame)).inverse()
        return cam_pose

    def get_initial_estimate_landmarks_p3d(self):
        landmarks = []
        for landmark_sym in self.__landmark_sym:
            landmark = self.__initial_estimate.atPoint3(landmark_sym)
            landmarks.append(landmark)

        return landmarks

    def get_key_frames(self):
        return self.__first_key_frame, self.__second_key_frame

    # Todo: Left for further checking

    # def create_factor_graph_all_tracks(self):
    #     gtsam_calib_mat = ex5.create_gtsam_calib_cam_mat(utils.K)
    #     # # Create gtsam objects for each frame
    #
    #     first_frame = self.__frames_in_window[0]
    #     # first_frame_global_cam_to_world_trans = gtsam.Pose3(first_frame.get_ex_cam_mat()).inverse()  # Trans from camera to world
    #
    #     first_frame_cam_to_world_ex_mat = ex5.convert_ex_cam_to_cam_to_world(first_frame.get_ex_cam_mat())  # first cam -> world
    #
    #     cur_cam_pose = None
    #     # Add projections factors for each frame - "camera"
    #     for i, frame in enumerate(self.__frames_in_window):
    #
    #         # Create camera symbol and update values dictionary
    #         left_pose_sym = symbol(CAMERA_SYM, frame.get_id())
    #         self.__camera_sym.add(left_pose_sym)
    #
    #         # Initialize constraints for first pose
    #         if i == 0:
    #             # sigmas array: first 3 for location second 3 for angles
    #             pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=np.array(
    #                 [10 ** -3, 10 ** -3, 10 ** -3, 10 ** -2, 10 ** -2, 10 ** -2]))  # todo: check choice of diagonal
    #             # Initial pose
    #             factor = gtsam.PriorFactorPose3(left_pose_sym, gtsam.Pose3(), pose_uncertainty)
    #             self.__graph.add(factor)
    #
    #         # camera_relate_to_first_frame_trans = first_frame_global_cam_to_world_trans.between(gtsam.Pose3(frame.get_ex_cam_mat()).inverse())
    #
    #         # Compute transformation of : (world - > cur cam) * (first cam -> world) = first cam -> cur cam
    #         camera_relate_to_first_frame_trans = utils.compose_transformations(first_frame_cam_to_world_ex_mat, frame.get_ex_cam_mat())
    #
    #         cur_cam_pose = ex5.convert_ex_cam_to_cam_to_world(camera_relate_to_first_frame_trans)  # cur cam -> first cam
    #         self.__initial_estimate.insert(left_pose_sym, gtsam.Pose3(cur_cam_pose))
    #
    #     gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)
    #     # For each track create measurements factors
    #
    #     for i, frame in enumerate(self.__frames_in_window):
    #
    #         tracks_in_frame = self.db.get_tracks_at_frame(frame.get_id())
    #
    #         for track in tracks_in_frame:
    #             if track.get_id() in self.__computed_tracks or track.get_last_frame_ind() < self.__second_key_frame + i:
    #                 continue
    #
    #             gtsam_last_cam = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)
    #             self.add_factors(track, frame.get_id(), self.__second_key_frame, gtsam_last_cam, gtsam_calib_mat)  # Todo: as before
    #
    #             self.__computed_tracks.add(track.get_id())
