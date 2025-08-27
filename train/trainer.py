import numpy as np
import tensorflow as tf
from tensorflow import keras

from train.distiller import Distiller


def train_with_KD(model, train_data, valid_data, config):
    # Train the model using knowledge distillation
    # (following the procedure from Keras code examples)

    distiller = Distiller(student=model)

    distiller.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["training"]["lr"]),
        metrics=["accuracy"],
        student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=False),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=config["training"]["alpha"],
        temperature=config["training"]["temperature"],
    )

    callbacks = []
    best_weights = config["training"]["restore_best_weights"]
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=best_weights
        )
    )
    """callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config['paths']["saving_folder"] / "best_distiller.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            initial_value_threshold=0.1,
            verbose=1,
            )
    )"""
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            # cooldown=2,
            min_lr=5e-5,
            verbose=0,
        )
    )

    history = distiller.fit(
        train_data,
        validation_data=valid_data,
        epochs=config["training"]["num_epochs"],
        callbacks=callbacks,
    )

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=config["training"]["lr"]),
        metrics=["accuracy"],
    )

    return model


#
#
#


def train_from_scratch(model, train_data, valid_data, config):
    # Train the model from scratch

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(learning_rate=config["training"]["lr"]),
        metrics=["accuracy"],
    )

    callbacks = []
    best_weights = config["training"]["restore_best_weights"]
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=best_weights
        )
    )
    """callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config['paths']["saving_folder"] / "best_distiller.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            initial_value_threshold=0.1,
            verbose=1,
            )
    )"""
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            # cooldown=2,
            min_lr=5e-5,
            verbose=0,
        )
    )

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=config["training"]["num_epochs"],
        callbacks=callbacks,
    )

    return model
