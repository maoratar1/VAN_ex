

class Features:
    """
    This class represents feature in frame
    """

    def __init__(self, kpt_left, kpt_right, x_left, x_right, y):
        self.__kpt_left = kpt_left
        self.__kpt_right = kpt_right
        self.__x_left = x_left
        self.__x_right = x_right
        self.__y = y

    def get_left_coor(self):
        """
        Return coordinates at the left image
        """
        return self.__x_left, self.__y

    def get_right_coor(self):
        """
        Returns coordinates at the right image
        """
        return self.__x_right, self.__y

    def get_left_kpt(self):
        """
        Returns key point in the left image (from the list of key points that found in left image)
        """
        return self.__kpt_left

    def get_right_kpt(self):
        """
        Returns key point in the right image (from the list of key points that found in right image)
        :return:
        """
        return self.__kpt_right

