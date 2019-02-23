import numpy as np
import os
from keras.utils import Sequence

from utils import load_image
from utils import to_onehot


class CarlaBatchGenerator(Sequence):

    def __init__(self, dataset, input_dir, target_dir, batch_size, image_size, encoding):
        self.dataset = dataset
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.encoding = encoding

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]

        inputs, targets = [], []
        for filename in batch:
            input_path = os.path.join(self.input_dir, filename)
            input_img = load_image(input_path, self.image_size)
            inputs.append(input_img)

            target_path = os.path.join(self.target_dir, filename)
            target_img = load_image(target_path, self.image_size)
            target_onehot = to_onehot(target_img, self.encoding)
            targets.append(target_onehot)

        return np.array(inputs), np.array(targets)
