from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()

# Set the class name
__C.YOLO.CLASSES = "./data/classes/voc2012.names"
__C.YOLO.ANCHORS = "./data/anchors/basline_anchors.txt"
__C.YOLO.ANCHORS_TINY = "./data/anchors/basline_tiny_anchors.txt"
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.STRIDES_TINY = [16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5
__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.IS_TINY = True

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = "./data/dataset/voc2012.txt"
__C.TRAIN.ANNO_PATH = "./data/dataset/voc2012/"
__C.TRAIN.BATCH_SIZE = 4
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE = [416]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LR_INIT = 1e-4
__C.TRAIN.LR_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 1
__C.TRAIN.EPOCHS = 136

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = "./data/dataset/voc2007_test.txt"
__C.TEST.ANNO_PATH = "./data/dataset/voc2007_test/"
__C.TEST.BATCH_SIZE = 2
__C.TEST.INPUT_SIZE = 416
__C.TEST.DATA_AUG = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD = 0.01
__C.TEST.IOU_THRESHOLD = 0.45
__C.TEST.WEIGHT_FILE = "./yolov3"
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = "./data/detection/"
__C.TEST.SHOW_LABEL = True
