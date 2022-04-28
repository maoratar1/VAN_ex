import numpy as np


class Frame:
    """
    This class represents a frame
    """

    def __init__(self, id):
        self.__id = id
        self.__tracks_ids = []  # Track's ids in this frame

    def add_track(self, track_id):
        """
        Adds track to frame
        """
        self.__tracks_ids.append(track_id)

    def get_id(self):
        """
        Returns frame's id
        """
        return self.__id

    def get_tracks_ids(self):
        """
        Returns track's ids list
        """
        return np.array(self.__tracks_ids).astype(int)

    def get_tracks_num(self):
        """
        Return tracks number at this frame
        """
        return len(self.__tracks_ids)