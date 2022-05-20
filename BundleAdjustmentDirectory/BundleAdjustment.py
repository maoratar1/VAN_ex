import concurrent.futures

import numpy as np
import tqdm
from DataDirectory import Data
from utils import utills
from BundleWindow import BundleWindow
import gtsam


ITERATIVE_METHOD = "ITERATIVE"
MULTI_PROCESSED = "MULTI PROCESSED"

class BundleAdjustment:

    def __init__(self):
        print("Creating bundle windows")

        # Create a bundle windows
        # self.__key_frames = self.choose_key_frames_by_median_track_len(Data.DB.get_frames())
        self.__key_frames = self.get_loaded_key_frames()
        self.__bundles_lst = self.create_bundle_windows(self.__key_frames)
        self.__num_bundles = len(self.__bundles_lst)

    def create_bundle_windows(self, key_frames):
        bundle_window_lst = []
        frames = Data.DB.get_frames()

        for i in range(1, len(key_frames)):
            first_key = key_frames[i - 1]
            second_key = key_frames[i]
            bundle_window = BundleWindow(first_key, second_key, frames[first_key: second_key + 1])
            bundle_window_lst.append(bundle_window)

        return bundle_window_lst

    def choose_key_frames_by_time(self, frames):
        key_frames = []
        print("Choosing key frames")
        for i in tqdm.tqdm(range(len(frames))):
            if i % 5 == 0:
                key_frames.append(i)

        return key_frames

    def choose_key_frames_by_median_track_len(self, frames):
        """
        Track len median
        """
        key_frames = [0]
        n = len(frames)
        print("Choosing key frames")
        while key_frames[-1] < n - 1:
            last_key_frame = key_frames[-1]
            frame = frames[last_key_frame]
            tracks = Data.DB.get_tracks_at_frame(frame.get_id())

            tracks_lens = []
            for track in tracks:
                tracks_lens.append(track.get_track_len())

            tracks_lens.sort()

            # new_key_frame = max(min(tracks_lens[len(tracks_lens) // 2], 20), 5) + last_key_frame
            new_key_frame = tracks_lens[len(tracks_lens) // 2] + last_key_frame
            key_frames.append(min(new_key_frame, n - 1))

        return key_frames

    def solve(self, method=ITERATIVE_METHOD, workers_num=5):
        gtsam_cameras_rel_to_bundle, all_landmarks_rel_to_bundle = None, None

        print("For all bundles - Create factor graph and optimize:")
        if method is MULTI_PROCESSED:
            gtsam_cameras_rel_to_bundle, all_landmarks_rel_to_bundle = self.bundle_adjustment_multi_processed(
                workers_num)

        elif method is ITERATIVE_METHOD:
            gtsam_cameras_rel_to_bundle, all_landmarks_rel_to_bundle = self.bundle_adjustment_iterative()

        gtsam_cameras_rel_to_world = utills.gtsam_left_cameras_relative_trans(gtsam_cameras_rel_to_bundle)
        landmarks_rel_to_world = self.compute_landmarks_in_relate_first_movie_camera(gtsam_cameras_rel_to_world,
                                                                                all_landmarks_rel_to_bundle)

        return gtsam_cameras_rel_to_world, landmarks_rel_to_world

    def bundle_adjustment_multi_processed(self, workers_num):
        cameras = [gtsam.Pose3()]
        landmarks = []

        with concurrent.futures.ProcessPoolExecutor(workers_num) as executor:
            results = list(executor.map(self.solve_bundle_window, tqdm.tqdm(range(self.__num_bundles))))  # [[Cameras, Landmarks], ...]

        for res in results:
            cameras.append(res[0])
            landmarks.append(res[1])

        return np.array(cameras), landmarks

    def create_and_solve_bundle(self, first_key, second_key, frames):
        bundle_window = BundleWindow(first_key, second_key, frames[first_key: second_key + 1])
        bundle_window.create_factor_graph()
        bundle_window.optimize()
        return bundle_window.get_optimized_cameras_p3d(), bundle_window.get_optimized_landmarks_p3d()

    def solve_bundle_window(self, bundle_num):
        bundle = self.__bundles_lst[bundle_num]
        bundle.create_factor_graph()
        bundle.optimize()
        return bundle.get_optimized_cameras_p3d(), bundle.get_optimized_landmarks_p3d()

    def bundle_adjustment_iterative(self):
        cameras = [gtsam.Pose3()]
        landmarks = []

        for i in tqdm.tqdm(range(len(self.__bundles_lst))):
            self.__bundles_lst[i].create_factor_graph()
            self.__bundles_lst[i].optimize()
            cameras.append(self.__bundles_lst[i].get_optimized_cameras_p3d())
            landmarks.append(self.__bundles_lst[i].get_optimized_landmarks_p3d())  # Todo: check if list of numpy arrays is ok

        return np.array(cameras), landmarks

    def compute_landmarks_in_relate_first_movie_camera(self, cameras, landmarks):
        relative_to_first_cam_landmarks = []
        for bundle_camera, bundle_landmarks in zip(cameras, landmarks):
            bundle_relative_to_first_cam_landmarks = self.compute_landmarks_in_relate_first_camera(bundle_camera,
                                                                                              bundle_landmarks)
            relative_to_first_cam_landmarks += bundle_relative_to_first_cam_landmarks

        return np.array(relative_to_first_cam_landmarks)

    def compute_landmarks_in_relate_first_camera(self, first_cam_in_bundle_rel_to_first_cam_in_movie, bundle_landmarks):
        relative_to_first_cam_landmarks = []
        for landmark in bundle_landmarks:
            relative_to_first_cam_landmarks.append(
                first_cam_in_bundle_rel_to_first_cam_in_movie.transformFrom(landmark))

        return relative_to_first_cam_landmarks

    def get_key_frames(self):
        return self.__key_frames

    def get_bundles_lst(self):
        return self.__bundles_lst

    def get_loaded_key_frames(self):
        return [0, 4, 9, 16, 25, 36, 47, 57, 69, 79, 91, 100, 108, 114, 120, 128, 136, 146, 154, 157, 162, 169,
                      178, 186, 196, 207, 213, 219, 226, 232, 239, 245, 251, 257, 263, 269, 273, 279, 284, 288, 292,
                      299, 304, 311, 319, 327, 335, 344, 357, 370, 376, 383, 390, 400, 410, 416, 423, 429, 436, 444,
                      455, 464, 475, 490, 531, 543, 549, 555, 562, 569, 574, 579, 584, 589, 594, 599, 604, 611, 618,
                      624, 630, 636, 641, 647, 654, 662, 672, 685, 700, 705, 710, 717, 726, 733, 739, 747, 754, 763,
                      773, 783, 793, 801, 808, 815, 822, 828, 832, 836, 840, 844, 848, 852, 857, 863, 869, 876, 882,
                      889, 896, 902, 908, 914, 922, 927, 932, 936, 943, 948, 953, 958, 963, 967, 972, 977, 982, 986,
                      990, 996, 1002, 1008, 1014, 1019, 1024, 1028, 1033, 1038, 1043, 1051, 1057, 1070, 1080, 1086,
                      1093, 1103, 1111, 1118, 1125, 1130, 1136, 1141, 1147, 1156, 1164, 1173, 1178, 1188, 1197, 1208,
                      1220, 1225, 1232, 1241, 1248, 1255, 1261, 1267, 1272, 1279, 1288, 1297, 1304, 1314, 1323, 1337,
                      1351, 1361, 1366, 1373, 1379, 1384, 1390, 1395, 1399, 1403, 1407, 1410, 1414, 1418, 1422, 1426,
                      1431, 1435, 1439, 1443, 1447, 1451, 1455, 1459, 1463, 1467, 1472, 1477, 1484, 1492, 1498, 1503,
                      1507, 1512, 1518, 1525, 1536, 1546, 1554, 1560, 1567, 1576, 1584, 1591, 1597, 1603, 1610, 1617,
                      1625, 1631, 1638, 1649, 1659, 1668, 1674, 1680, 1687, 1693, 1698, 1703, 1708, 1713, 1720, 1726,
                      1733, 1739, 1746, 1753, 1760, 1766, 1772, 1778, 1785, 1792, 1798, 1803, 1807, 1813, 1818, 1823,
                      1828, 1833, 1838, 1844, 1850, 1854, 1858, 1862, 1867, 1874, 1882, 1891, 1900, 1907, 1915, 1924,
                      1932, 1939, 1946, 1952, 1957, 1962, 1967, 1973, 1978, 1983, 1989, 1992, 1996, 2001, 2006, 2012,
                      2018, 2023, 2029, 2036, 2043, 2052, 2063, 2070, 2076, 2085, 2094, 2102, 2110, 2116, 2124, 2133,
                      2140, 2145, 2150, 2155, 2161, 2168, 2178, 2188, 2202, 2215, 2222, 2228, 2235, 2241, 2248, 2253,
                      2258, 2263, 2272, 2280, 2287, 2294, 2300, 2307, 2313, 2319, 2324, 2329, 2333, 2340, 2348, 2354,
                      2361, 2374, 2388, 2395, 2405, 2415, 2422, 2428, 2436, 2443, 2448, 2453, 2459, 2466, 2474, 2479,
                      2487, 2493, 2499, 2507, 2515, 2521, 2528, 2534, 2542, 2551, 2560, 2569, 2574, 2578, 2584, 2591,
                      2596, 2601, 2607, 2613, 2621, 2629, 2637, 2644, 2648, 2653, 2661, 2670, 2680, 2687, 2692, 2696,
                      2702, 2706, 2711, 2717, 2723, 2727, 2732, 2737, 2743, 2748, 2755, 2762, 2769, 2774, 2778, 2783,
                      2788, 2793, 2799, 2806, 2815, 2822, 2828, 2834, 2842, 2851, 2856, 2863, 2869, 2874, 2879, 2885,
                      2894, 2899, 2904, 2910, 2917, 2924, 2931, 2936, 2943, 2950, 2956, 2963, 2971, 2979, 2984, 2990,
                      2995, 3000, 3008, 3014, 3022, 3029, 3035, 3042, 3049, 3057, 3063, 3069, 3082, 3090, 3095, 3101,
                      3105, 3109, 3115, 3120, 3127, 3133, 3139, 3145, 3150, 3155, 3161, 3165, 3169, 3174, 3179, 3183,
                      3186, 3189, 3193, 3196, 3199, 3202, 3206, 3212, 3218, 3223, 3230, 3236, 3243, 3252, 3259, 3264,
                      3269, 3274, 3278, 3284, 3293, 3300, 3309, 3324, 3338, 3346, 3356, 3370, 3379, 3387, 3392, 3401,
                      3410, 3417, 3423, 3429, 3436, 3441, 3449]
