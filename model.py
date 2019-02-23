import segmentation_models as seg_models
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras_contrib.losses import jaccard_distance

from utils import tversky_index


MODELS = {
    'unet': seg_models.Unet,
    'fpn': seg_models.FPN,
    'pspnet': seg_models.PSPNet,
    'linknet': seg_models.Linknet
}


def tversky_loss(t, p):
    return 1 - tversky_index(t, p)


def combined_loss(t, p):
    return categorical_crossentropy(t, p) + tversky_loss(t, p)


def build(**kwargs):
    architecture = kwargs['architecture']
    backbone = kwargs['backbone']
    input_shape = kwargs['input_shape']
    num_classes = kwargs['num_classes']
    freeze_encoder = kwargs.get('freeze_encoder', True)
    learning_rate = kwargs.get('learning_rate', 1e-5)
    weights = kwargs.get('weights', None)

    model = MODELS[architecture](
        backbone_name=backbone,
        input_shape=input_shape,
        classes=num_classes,
        activation='softmax',
        encoder_freeze=freeze_encoder,
    )
    if weights:
        model.load_weights(weights)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss=combined_loss,
        metrics=['categorical_accuracy']
    )

    model.summary()

    return model
