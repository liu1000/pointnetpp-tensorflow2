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

from models.sem_seg_model import SEM_SEG_Model

tf.random.set_seed(42)

N_POINTS = 8096


def train():
    model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])

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
        loss=_modified_binary_focal_crossentropy,
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
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
            .assign(label=lambda df: df[label_col].astype(int))
    )
    point_coords = points[["x", "y", "z"]].values
    point_labels = points[["label"]].values
    return point_coords, point_labels


def _convert_to_tf_dataset(features, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(features, tf.float32),
        tf.constant(labels, tf.int64)))
    ds = ds.shuffle(6000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


if __name__ == '__main__':

    config = {
        'pcloud_path': pathlib.Path('.').resolve().parent/'data'/'pcloud',
        'tfds_train_path': pathlib.Path('.')/'data'/'pienet_ptclf_tr',
        'tfds_val_path': pathlib.Path('.')/'data'/'pienet_ptclf_val',
        'regen_tfds': True,
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
