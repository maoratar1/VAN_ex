from DataDirectory import KittiData

# Load or create Kitti's DataDirectory
print("Trying to load an existing object of Kitti's data")
try:
    KITTI = KittiData.load(KittiData.LOADED_KITTI_DATA)
    print(f"\tKitti's data object exists and was loaded from the path: {KittiData.LOADED_KITTI_DATA}")
except:
    print("\tKitti's data did not created yed, Let's create it now:")
    KITTI = KittiData.KittiData()
    KittiData.save(KittiData.LOADED_KITTI_DATA, KITTI)
    print(f"\tKitti's data object created and it saved at : {KittiData.LOADED_KITTI_DATA}")

print("")

from utils import utills
from DataBaseDirectory import DataBase

# Load or create DataDirectory base
print("Trying to load an existing object of Data base")
try:
    DB = DataBase.load(DataBase.LOADED_DB_PATH)
    print(f"\tData base object exists and was loaded from the path: {DataBase.LOADED_DB_PATH}")
except:
    print("\tData base has not been created yet, Let's create it now:")
    consecutive_frame_features, inliers_percentage, global_trans, relative_to_first_trans = utills.find_features_in_consecutive_frames_whole_movie()
    DB = DataBase.DataBase(consecutive_frame_features, inliers_percentage, global_trans, relative_to_first_trans)
    DataBase.save(DataBase.LOADED_DB_PATH, DB)
    print(f"\tDataDirectory base object created and it saved at : {DataBase.LOADED_DB_PATH}")
