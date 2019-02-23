# Road Segmentation Module

- Module for extracting drivable paths and road lines in [CARLA simulator](http://carla.org/) (version: 0.8.4)

## Installation

For accelerated neural network training using dedicated GPU cards follow this [link](https://www.tensorflow.org/install/).

Install virtual environment
```bash
pip install virtualenv
virtualenv venv
```

Install required packages
```bash
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

## Data

You can download CARLA simulator from this [link]() and use it's client_example.py module to collect data.


## Configuration

This module supports configuration through json file format. In the root folder you can find config.json.example file. Rename it to config.json and set missing values.


## Training

To train the model run
```bash
python train.py
```

## Evaluation

To evaluate the model run
```bash
python evaluate.py
```

Metrics used for evaluation of this model are: mean IoU and categorical accuracy.