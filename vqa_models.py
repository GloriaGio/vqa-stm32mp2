from tensorflow import keras
import numpy as np

# su STM32MP257F-EV1 impiega 151.3 ms da quantizzato.


def MFB_Baseline(
    k=5,
    maxlen=15,
    num_words=5000,
    emb_dim=50,
    glove_emb=None,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.1,
    last_softmax=True,
):
    # Inputs
    text_input = keras.layers.Input(shape=(maxlen,), name="text_input")
    image_input = keras.layers.Input(
        shape=(im_size, im_size, num_channels), name="image_input"
    )

    # Textual Feature Extraction
    x1 = keras.layers.Embedding(
        input_dim=num_words, output_dim=emb_dim, mask_zero=False
    )(text_input)
    if glove_emb is not None:
        x2 = keras.layers.Embedding(
            input_dim=num_words, output_dim=emb_dim, mask_zero=False, trainable=False
        )(text_input)
        x1 = keras.layers.Concatenate(axis=-1)([x1, x2])
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512,
        6,
        dilation_rate=2,
        padding="causal",
        kernel_initializer="he_normal",
    )(x1)
    x2 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(x2)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512,
        6,
        dilation_rate=2,
        padding="causal",
        kernel_initializer="he_normal",
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Concatenate(axis=-1)([x2, x1])
    x1 = keras.layers.GlobalAveragePooling1D()(x1)

    # Image Feature Extraction
    mobilenet_base = keras.applications.MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_preprocessing=False,
    )
    mobilenet_base.trainable = False
    x2 = mobilenet_base(image_input, training=False)
    x2 = keras.layers.GlobalAveragePooling2D()(x2)
    x2 = keras.layers.UnitNormalization(axis=-1)(x2)

    # MFB pooling
    x1 = keras.layers.Dense(1024 * k, kernel_initializer="he_normal")(x1)
    x2 = keras.layers.Dense(1024 * k, kernel_initializer="he_normal")(x2)
    x = keras.layers.Multiply()([x1, x2])
    x = keras.layers.Reshape((-1, 1))(x)
    x = keras.layers.AveragePooling1D(pool_size=k, strides=k, padding="same")(x)
    x = keras.layers.Reshape((-1,))(x)
    # no pow norm
    x = keras.layers.UnitNormalization(axis=-1)(x)

    # Classification
    x = keras.layers.Dense(2048, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    if last_softmax:
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    else:
        outputs = keras.layers.Dense(num_classes)(x)

    # Output
    model = keras.models.Model(inputs=[text_input, image_input], outputs=outputs)

    return model


def MFB_Attention(
    k=5,
    maxlen=15,
    num_words=5000,
    emb_dim=50,
    glove_emb=None,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.1,
    last_softmax=True,
):
    # prima di farlo rivedo bene architettura
    return 0


def MFB_CoAttention(
    k=5,
    maxlen=15,
    num_words=5000,
    emb_dim=50,
    glove_emb=None,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.1,
    last_softmax=True,
):
    # prima di farlo rivedo bene architettura
    return 0


if __name__ == "__main__":
    model = MFB_Baseline()
    print("Number of parameters:", model.count_params())

    num_words = 5000
    maxlen = 15
    im_size = 224
    num_ch = 3
    q = np.random.randint(0, num_words, (1, maxlen))
    im = np.random.randint(0, 256, (1, im_size, im_size, num_ch))
    im = im / 255 * 2 - 1
    pred = model.predict((q, im))
    print("Output shape:", pred.shape)

    if False:
        model.save("PROVA.keras")
