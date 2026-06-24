# -- Uni-Sign 69-keypoint selection ------------------------------------------
# COCO-WholeBody-133 -> Uni-Sign 69 selection (ZechengLi19/Uni-Sign datasets.py load_part_kp), in the
# fixed part order [body 9 | left 21 | right 21 | face_all 18] that backbones.unisign.UniSignPoseEncoder
# splits back out. These indices + the crop_scale in preprocessing.normalize_keypoints_unisign MUST stay
# byte-faithful or the released pose-only checkpoints (csl_daily/how2sign/openasl) see out-of-distribution
# input and the pretraining is wasted. (133-kp layout: body 0-16, feet 17-22, face 23-90, L hand 91-111,
# R hand 112-132.)
UNISIGN_BODY_IDX = [0] + list(range(3, 11))                                  # 9  (nose + shoulders..wrists)
UNISIGN_LEFT_IDX = list(range(91, 112))                                      # 21 (left hand)
UNISIGN_RIGHT_IDX = list(range(112, 133))                                    # 21 (right hand)
UNISIGN_FACE_IDX = list(range(23, 40))[::2] + list(range(83, 91)) + [53]     # 18 (9 contour + 8 mouth + nose)
# The 69 raw-133 indices in part order — for selecting the raw subset (e.g. visualize.py skeleton overlay).
UNISIGN_SELECTED_IDS = UNISIGN_BODY_IDX + UNISIGN_LEFT_IDX + UNISIGN_RIGHT_IDX + UNISIGN_FACE_IDX
UNISIGN_NUM_KP = len(UNISIGN_SELECTED_IDS)

from .preprocessing import *
from .augmentation import *
from .pose_io import *