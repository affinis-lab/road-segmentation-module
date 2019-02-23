import argparse
import json
import os
from ast import literal_eval
from collections import defaultdict
from keras.models import load_model

import model as SegModel
from generator import CarlaBatchGenerator
from utils import BASE_DIR
from utils import load_dataset
from utils import tversky_index


def mean_iou(y_true, y_pred):
    return tversky_index(y_true, y_pred, alpha=1, beta=1)


def evaluate(**kwargs):
    model_name = kwargs['model_name']
    model_dir = kwargs.get('model_dir', os.path.join(BASE_DIR, 'models'))
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    
    input_images = kwargs['input_images']
    target_images = kwargs['target_images']
    
    batch_size = kwargs.get('batch_size', 16)
    image_size = (kwargs['input_shape']['width'], kwargs['input_shape']['height'])
    encoding = kwargs['encoding']
    
    model = load_model(model_path, compile=False)

    model.compile(
        optimizer='Adam',
        loss="categorical_crossentropy",
        metrics=[mean_iou, 'categorical_accuracy', ]
    )

    dataset = load_dataset(input_images)

    test_gen = CarlaBatchGenerator(
        dataset=dataset,
        input_dir=input_images,
        target_dir=target_images,
        batch_size=batch_size,
        image_size=image_size,
        encoding=encoding
    )

    metrics = model.evaluate_generator(
        generator=test_gen,
        workers=10,
        use_multiprocessing=False,
        verbose=1
    )

    print(f'Loss: {metrics[0]}')
    print(f'Mean IOU: {metrics[1]}')
    print(f'Acc: {metrics[2]}')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Road segmentation module evaluation')

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
    config['evaluation']['encoding'] = CARLA_ENCODING

    evaluate(**config['evaluation'])
