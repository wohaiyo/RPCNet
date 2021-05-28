# Config file for define some parameters
import argparse
import numpy as np

def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Facade Segmentation")
    parser.add_argument("--data-dir", type=str, default='/media/ilab/Storage 2/PycharmProjects/Repetitive_pattern/data/repetitive_pattern/',
                        help="Training and evaluation facade dataset.")
    parser.add_argument("--eval-dir", type=str, default='/media/ilab/Storage 2/PycharmProjects/Repetitive_pattern/data/repetitive_pattern/val',
                        help="Training and evaluation facade dataset.")
    parser.add_argument("--dataset-size", type=int, default=65,
                        help="Number of training images.")
    parser.add_argument("--dataset-num-classes", type=int, default=4, #
                        help="Number of classes to predict (including background).")
    parser.add_argument("--ignore-label", type=int, default=0,
                        help="The index of the label to ignore during the training.")

    parser.add_argument("--batch-size", type=int, default=2,    # 2, 8
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float,
                        default=1e-4, help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--image-height", type=int, default=512,
                        help="Image height and width of image.")
    parser.add_argument("--image-width", type=int, default=512,
                        help="Image height and width of image.")
    parser.add_argument("--freeze-bn", action="store_false",
                        help="Whether to freezes the running means and variances during the training.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                        help="Regularisation parameter for L2-loss.")

    parser.add_argument("--restore-from", type=str, default='data/pre_trained_model/resnet_v1_50.ckpt',
                        help="Where restore model parameters from.")
    parser.add_argument("--summary-interval", type=int, default=60,
                        help="Save summaries and training image every often.")
    parser.add_argument("--max-snapshot-num", type=int, default=5,
                        help="The maximal snapshot number to save.")
    parser.add_argument("--num-steps", type=int, default=6000,
                        help="Number of training steps.")
    parser.add_argument("--start-save-step", type=int, default=6000,
                        help="The step to start save checkpoint.")
    parser.add_argument("--save-step-every", type=int, default=2000,
                        help="Save checkpoint every often.")

    parser.add_argument("--random-crop-pad", action="store_false",
                        help="Whether to randomly crop and pad the inputs during the training.")
    parser.add_argument("--random-mirror", action="store_false",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_false",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--min-scale", type=float, default=0.5,
                        help="The minimal scale of the inputs during the training.")
    parser.add_argument("--max-scale", type=float, default=2.0,
                        help="The maximal scale of the inputs during the training.")

    parser.add_argument("--save-dir", type=str, default='saves/artdeco3_folds/set5/RPCNet/',
                        help="The Snapshot save path.")
    parser.add_argument("--log-dir", type=str, default='logs/artdeco3_folds/set5/RPCNet/',
                        help="The Logs save path.")

    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="The number of gradient accumulation.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")

    return parser.parse_args()
args = get_arguments()

# --------------- Hyperparameters ---------------------------

# Dataset description
DATA_DIR = args.data_dir
TRAIN_DATA_DIR = DATA_DIR + 'train'
TRAIN_DATA_LIST = DATA_DIR + 'train.txt'

EVAL_DIR = args.eval_dir

DATASET_SIZE = args.dataset_size

DATASET_NUM_CLASSESS = args.dataset_num_classes

IGNORE_LABEL = args.ignore_label

# Hyper parameters
BATCH_SIZE = args.batch_size

LEARNING_RATE = args.learning_rate

IMAGE_HEIGHT = args.image_height

IMAGE_WIDTH = args.image_width

FREEZE_BN = args.freeze_bn

POWER = args.power

WEIGHT_DECAY = args.weight_decay

# Paras
PRE_TRAINED_MODEL = args.restore_from

SUMMARY_INTERVAL = args.summary_interval

MAX_SNAPSHOT_NUM = args.max_snapshot_num

NUM_STEPS = args.num_steps

START_SAVE_STEP = args.start_save_step

SAVE_STEP_EVERY = args.save_step_every

# Data pre-process
RANDOM_CROP_PAD = args.random_crop_pad

RANDOM_MIRROR = args.random_mirror

RANDOM_SCALE = args.random_scale

MIN_SCALE = args.min_scale

MAX_SCALE = args.max_scale

# Create path to save
SAVE_DIR = args.save_dir

LOG_DIR = args.log_dir

#
GRADIENT_ACCUMULATION = args.gradient_accumulation

GPU = args.gpu

IMG_MEAN = np.array([103.94, 116.78, 123.68], dtype=np.float32)     # B G R

if 'artdeco' in DATA_DIR:
    WINDOW_CHANNEL = 4
    WALL_CHANNEL = 5
else:
    WINDOW_CHANNEL = 1
    WALL_CHANNEL = 2
BALCONY_CHANNEL = 3



