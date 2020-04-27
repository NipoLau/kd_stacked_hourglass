from yacs.config import CfgNode as CN

_C = CN()

_C.WORKERS = 4
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 100

_C.MODEL = CN()
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.NAME = 'posenet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 16
_C.MODEL.IMAGE_SIZE = [256, 256]
_C.MODEL.HEATMAP_SIZE = [64, 64]
_C.MODEL.SIGMA = 2

_C.MODEL.EXTRA = CN(new_allowed=True)

_C.DATASET = CN()
_C.DATASET.ROOT = 'data/mpii'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.TRAIN_SET = 'train'

_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 16

_C.TRAIN = CN()
_C.TRAIN.TYPE = 'KD'
_C.TRAIN.INTER = True

_C.TRAIN.BATCH_SIZE = 12

_C.TRAIN.LR = 0.00025
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [70, 100]

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 150

_C.TRAIN.HINT_EPOCH = 10
_C.TRAIN.KD_WEIGHT = 0.5
_C.TRAIN.OUTPUT = 'experiments/output'

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


def get_cfg(cfg_file):
    cfg = _C.clone()

    cfg.defrost()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    return cfg
