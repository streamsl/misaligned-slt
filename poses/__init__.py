# -- Pose Preprocessing (CoSign Inspired) -----------------------------------
# Define keypoint groups based on COCO-WholeBody (133 points: body 0-16, left foot 17-19, right foot 20-22, face 23-90, left hand 91-111, right hand 112-132)
# Upper body (9): nose(0), left eye(1), right eye(2), left shoulder(5), right shoulder(6), left elbow(7), right elbow(8), left wrist(9), right wrist(10)
# Mouth (8): inner lips approx 60-67 in face (face starts at 23, so 23+60-67 = 83-90, but adjust to 8)
# Face lower/cheek (18): contour approx 23+0 to 23+16 (17 points), plus nose
BODY_IDS = [0, 1, 2, 5, 6, 7, 8, 9, 10]  # 9 points
LEFT_HAND_IDS = list(range(91, 112))  # 21 points
RIGHT_HAND_IDS = list(range(112, 133))  # 21 points
MOUTH_IDS = list(range(83, 91))  # Inner mouth 8 points
FACE_IDS = list(range(23, 40)) + [53]  # First 18 as cheek/lower approx
ALL_SELECTED_IDS = BODY_IDS + LEFT_HAND_IDS + RIGHT_HAND_IDS + MOUTH_IDS + FACE_IDS  # Total 9+21+21+8+18=77
NUM_KEYPOINTS = len(ALL_SELECTED_IDS)
KPS_MODULES = {
    'body': {'kps_ids': BODY_IDS, 'kps_rel_range': (0, 9)},
    'left_hand': {'kps_ids': LEFT_HAND_IDS, 'kps_rel_range': (9, 30)},
    'right_hand': {'kps_ids': RIGHT_HAND_IDS, 'kps_rel_range': (30, 51)},
    'mouth': {'kps_ids': MOUTH_IDS, 'kps_rel_range': (51, 59)},
    'face': {'kps_ids': FACE_IDS, 'kps_rel_range': (59, 77)},
}
CONF_THRESHOLD = 0.5  # From supp: Keypoints with conf > 0.5 considered valid
WIDTH, HEIGHT = 1920, 1080
FPS = 25.0

from .preprocessing import *
from .augmentation import *
