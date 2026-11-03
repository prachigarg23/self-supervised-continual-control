import os
### Task parameters

DATA_DIR = os.path.expanduser('~/aloha_datasets')

TASK_CONFIGS = {
    'aloha_wear_shoe':{
        'dataset_dir': DATA_DIR + '/aloha_wear_shoe',
        'num_episodes': 50,
        'episode_len': 1000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_pick_cubes':{
        'dataset_dir': DATA_DIR + '/aloha_pick_cubes',
        'num_episodes': 53,
        'episode_len': 800,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_rings':{
        'dataset_dir': DATA_DIR + '/aloha_rings',
        'num_episodes': 60,
        'episode_len': 1200,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_caterpillar':{
        'dataset_dir': DATA_DIR + '/aloha_caterpillar',
        'num_episodes': 80,
        'episode_len': 1400,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_search_bag':{
        'dataset_dir': DATA_DIR + '/aloha_search_bag',
        'num_episodes': 40,
        'episode_len': 2000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_basic_pick_place':{
        'dataset_dir': DATA_DIR + '/aloha_basic_pick_place',
        'num_episodes': 30,
        'episode_len': 800,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_laundry_in_basket':{
        'dataset_dir': DATA_DIR + '/aloha_laundry_in_basket',
        'num_episodes': 10,
        'episode_len': 1200,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_transfer_cube_myhuman_makeup':{
        'dataset_dir': DATA_DIR + '/aloha_transfer_cube_myhuman_makeup',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_pick_X_insert_socket':{
        'dataset_dir': DATA_DIR + '/aloha_pick_X_insert_socket',
        'num_episodes': 40,
        'episode_len': 800,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_pick_X_insert_socket_static_L':{
        'dataset_dir': DATA_DIR + '/aloha_pick_X_insert_socket_static_L',
        'num_episodes': 20,
        'episode_len': 800,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_pick_X_insert_socket_static_R':{
        'dataset_dir': DATA_DIR + '/aloha_pick_X_insert_socket_static_R',
        'num_episodes': 20,
        'episode_len': 800,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_caterpillar_task2':{
        'dataset_dir': DATA_DIR + '/aloha_caterpillar_task2',
        'num_episodes': 30,
        'episode_len': 800,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_caterpillar_task2_right_arm':{
        'dataset_dir': DATA_DIR + '/caterpillar_Task2_right_arm',
        'num_episodes': 60,
        'episode_len': 960,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_caterpillar_task3_simple':{
        'dataset_dir': DATA_DIR + '/aloha_caterpillar_task3_simple',
        'num_episodes': 30,
        'episode_len': 960,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_pick_R':{
        'dataset_dir': DATA_DIR + '/aloha_pick_R',
        'num_episodes': 30,
        'episode_len': 800,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    # March 19
    'aloha_pick_R_new':{
        'dataset_dir': DATA_DIR + '/aloha_pick_R_new',
        'num_episodes': 30,
        'episode_len': 800,
        # 'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'ee_camera_calibration':{
        'dataset_dir': DATA_DIR + '/ee_camera_calibration',
        'num_episodes': 30,
        'episode_len': 1600,
        'camera_names': ['cam_high']
    },
}

### ALOHA fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
