import enum


class ImageInfo(enum.Enum):

    IMAGE_TRAIN_URL = 'data/chest_xray/train'
    IMAGE_TEST_URL = 'data/chest_xray/test'
    IMAGE_VALIDATION_URL = 'data/chest_xray/val'
    BATCH_SIZE = 22
    IMAGE_SIZE = (224, 224)
    IMAGE_SHAPE = (224, 224, 3)
    EPOCHS = 10
