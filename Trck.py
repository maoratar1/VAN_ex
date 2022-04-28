import bisect
import numpy as np


class Track:

    def __init__(self, track_id):
        self.__id = track_id
        self.__frames_locations = dict()  # {Frame id: feature} contains the track Feature object in each frame
        self.__frames_ids = []

    def get_id(self):
        """
        Returns track id
        """
        return self.__id

    def add_feature(self, frame_id, feature):
        """
        Add a feature to frame with "frame_id" index
        :return:
        """
        self.__frames_locations[frame_id] = feature
        bisect.insort(self.__frames_ids, frame_id)

    def get_frames_idxes(self):
        """
        Returns the frames indexes
        """
        return np.array(list(self.__frames_locations.keys())).astype(int)

    def get_feature_at_frame(self, frame_id):
        """
        Returns the tracks feature in frame with index frame_id
        """
        return self.__frames_locations[frame_id]

    def get_left_locations_in_all_frames(self):
        """
        Returns the locations at all the left frames
        """
        left_locations = []
        frame_ids = self.__frames_ids

        for frame in frame_ids:
            feature = self.__frames_locations[frame]
            left_locations.append(feature.get_left_coor())

        return left_locations

    def get_right_locations_in_all_frames(self):
        """
        Returns the locations at all the right frames
        """
        right_locations = []
        frame_ids = self.__frames_ids

        for frame in frame_ids:
            feature = self.__frames_locations[frame]
            right_locations.append(feature.get_right_coor())

        return right_locations

        # return self.__frames_ids  # todo: add it after running the db again

    def get_frames_features_dict(self):
        """
        Returns the dictionatry of {frame id: feature}
        :return:
        """
        return self.__frames_locations

    def get_track_len(self):
        """
        Returns the track length
        """
        return len(self.__frames_locations)