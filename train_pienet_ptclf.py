import datetime as dt
import os
import pathlib
import sys

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import constants as C
from models.pienet_ptclf_model import PIENET_PT_MODEL

tf.random.set_seed(42)

N_POINTS = 8096


def train():
    model = PIENET_PT_MODEL(config['batch_size'], config['num_classes'], config['bn'])

    print("Loading data..")
    train_ds, val_ds = load_dataset(config)

    callbacks = [
        keras.callbacks.TensorBoard(
            f'./logs/{config["log_dir"]}', update_freq=50),
        # keras.callbacks.ModelCheckpoint(
        #     f'./logs/{config["log_dir"]}/model/weights', 'val_sparse_categorical_accuracy', save_best_only=True)
    ]

    print("Building..")
    model.build((config['batch_size'], N_POINTS, 3))
    print(model.summary())

    print("Compiling..")
    model.compile(
        optimizer=keras.optimizers.Adam(config['lr']),
        loss={
            # pt clf
            'output_1': _modified_binary_focal_crossentropy,
            # offset reg
            'output_2': keras.metrics.MeanSquaredError(),
        },
        metrics={
            'output_1': [keras.metrics.SparseCategoricalAccuracy()],
            # 'output_2': [keras.metrics.MeanSquaredError()],
        },
        loss_weights={'output_1': 1.0, 'output_2': 1.0},
    )

    print("Fitting..")
    model.fit(
        train_ds,
        validation_data=val_ds,
        validation_steps=10,
        validation_freq=1,
        callbacks=callbacks,
        epochs=100,
        verbose=1
    )


def load_dataset(config):
    if config['regen_tfds'] or not config['tfds_train_path'].exists():
        print("Preparing TF Dataset from raw PCloud data")
        in_path = config['pcloud_path']
        batch_size = config['batch_size']
        label_type = config['label_type']

        assert os.path.exists(in_path), f'[error] dataset {in_path} not found'

        features, labels = _load_dataset_from_dir(in_path, label_type=label_type)

        n_tr = int(len(labels) * 0.7)
        train_ds = _convert_to_tf_dataset(features[:n_tr], labels[:n_tr], batch_size)
        val_ds = _convert_to_tf_dataset(features[n_tr:], labels[n_tr:], batch_size)

        train_ds.save(str(config['tfds_train_path']))
        val_ds.save(str(config['tfds_val_path']))

        return train_ds, val_ds

    print("Loading previously saved TF Datasets")
    train_ds = tf.data.Dataset.load(str(config['tfds_train_path']))
    val_ds = tf.data.Dataset.load(str(config['tfds_val_path']))
    return train_ds, val_ds


def _modified_binary_focal_crossentropy(y_true, y_pred):
    y_pred_pos = y_pred[:, :, 1:]
    return tf.keras.metrics.binary_focal_crossentropy(y_true, y_pred_pos)


def _load_dataset_from_dir(path, label_type="edge", n_files=None):
    paths = sorted(str(p) for p in pathlib.Path(path).glob("*.parq"))
    if n_files:
        paths = paths[:n_files]

    features, labels = [], []
    for path in paths:
        coords, per_pt_labels = _load_single_pcloud(path, label_type=label_type)
        features.append(coords)
        labels.append(per_pt_labels)

    return np.array(features), np.array(labels)


def _load_single_pcloud(path, label_type=""):
    label_col = f"is_{label_type}"
    points = (
        pd.read_parquet(path)
            .drop_duplicates(["x", "y", "z"])  # TODO shouuld keep the closest
            .assign(label=lambda df: df[label_col].astype(int),
                    x_diff=lambda df: df.x - df.x_orig,
                    y_diff=lambda df: df.y - df.y_orig,
                    z_diff=lambda df: df.z - df.z_orig)
            .fillna(0)
    )
    point_coords = points[["x", "y", "z"]].values
    point_labels = points[["label", "x_diff", "y_diff", "z_diff"]].values
    return point_coords, point_labels


def _convert_to_tf_dataset(features, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(features, tf.float32),
        {
            "output_1": tf.constant(labels[:, :, :1], tf.int64),  # pt clf
            "output_2": tf.constant(labels[:, :, 1:4], tf.float32)  # offset reg
        }
    ))
    ds = ds.shuffle(6000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


if __name__ == '__main__':

    config = {
        'pcloud_path': pathlib.Path('.').resolve().parent/'data'/'pcloud',
        'tfds_train_path': pathlib.Path('.')/'data'/'pienet_ptclf_tr',
        'tfds_val_path': pathlib.Path('.')/'data'/'pienet_ptclf_val',
        'regen_tfds': False,
        'label_type': 'corner',
        'log_freq' : 10,
        'test_freq' : 100,
        'batch_size' : 4,
        'num_classes' : 2,
        'lr' : 0.001,
        'bn' : True,
    }

    timestamp = dt.datetime.now().isoformat(timespec="minutes")
    config.update({
        'log_dir': f'pienet_{config["label_type"]}_ptcls_{timestamp}'
    })

    train()
