import argparse
import json
import os
from ast import literal_eval
from collections import defaultdict
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import model as SegModel
from generator import CarlaBatchGenerator
from utils import load_dataset
from utils import BASE_DIR


def train(**kwargs):
    model_name = kwargs['model_name']
    model_dir = kwargs.get('model_dir', os.path.join(BASE_DIR, 'models'))
    input_images = kwargs['input_images']
    target_images = kwargs['target_images']
    
    architecture = kwargs.get('architecture', 'fpn')
    backbone = kwargs.get('backbone', 'vgg16')
    freeze_encoder = kwargs.get('freeze_encoder', False)
    input_shape = (
        kwargs['input_shape']['height'],
        kwargs['input_shape']['width'],
        kwargs['input_shape']['channels']
    )
    learning_rate = kwargs.get('learning_rate', 1e-5)
    weights = kwargs.get('weights', None)

    if weights:
        weights = os.path.join(model_dir, weights)

    train_val_ratio = kwargs.get('train_val_ratio', 0.1)
    batch_size = kwargs.get('batch_size', 16)
    epochs = kwargs.get('epochs', 10)
    num_classes = kwargs['num_classes']
    image_size = (kwargs['image_size']['width'], kwargs['image_size']['height'])
    encoding = kwargs['encoding']
    workers = kwargs.get('workers', 1)
    multiprocessing = kwargs.get('multiprocessing', False)

    dataset = load_dataset(input_images)
    train_set, val_set = train_test_split(dataset, test_size=train_val_ratio, random_state=99)

    train_gen = CarlaBatchGenerator(
        dataset=train_set,
        input_dir=input_images,
        target_dir=target_images,
        batch_size=batch_size,
        image_size=image_size,
        encoding=encoding
    )
    val_gen = CarlaBatchGenerator(
        dataset=val_set,
        input_dir=input_images,
        target_dir=target_images,
        batch_size=batch_size,
        image_size=image_size,
        encoding=encoding
    )

    model = SegModel.build(
        architecture=architecture,
        backbone=backbone,
        weights=weights,
        freeze_encoder=freeze_encoder,
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate
    )

    filepath = os.path.join(model_dir, f"{model_name}.h5")

    model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        epochs=epochs,
        workers=workers,
        use_multiprocessing=multiprocessing,
        shuffle=True,
        verbose=1,
        callbacks=[
            TensorBoard(
                batch_size=batch_size,
                update_freq=20 * batch_size
            ),
            ModelCheckpoint(
                filepath=filepath,
                save_best_only=True,
            )
        ]
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Road segmentation module training routine')

    arg_parser.add_argument(
        '-c',
        '--conf',
        help='Path to the configuration file',
        default='config.json',
    )

    args = arg_parser.parse_args()

    with open(args.conf, 'r') as f:
        config = json.load(f)

    CARLA_ENCODING = defaultdict(lambda: [1, 0, 0])
    for key, val in config['common']['carla_encoding'].items():
        CARLA_ENCODING[literal_eval(key)] = val
    config['training']['encoding'] = CARLA_ENCODING

    train(**config['training'])
