from tensorflow import keras
import numpy as np

#
#
#
#
#
#
#


# o net6
def tiny_net(
    maxlen=15,
    num_words=2000,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.1,
    last_softmax = True
):

    # Inputs
    text_input = keras.layers.Input(shape=(maxlen,), name="text_input")
    image_input = keras.layers.Input(
        shape=(im_size, im_size, num_channels), name="image_input"
    )

    # Textual Feature Extraction
    x1 = keras.layers.Embedding(input_dim=num_words, output_dim=300, mask_zero=False)(
        text_input
    )
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 6, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.GlobalAveragePooling1D()(x1)
    # output: (bs, 512)

    # Visual Feature Extraction
    x2 = keras.layers.Conv2D(
        64, (3, 3), padding="same", kernel_initializer="he_normal"
    )(image_input)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    x2 = keras.layers.SeparableConv2D(
        256,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    x2 = keras.layers.SeparableConv2D(
        512,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # output: (bs, 28, 28, 512)

    # Visual attention weight and fusion
    x = keras.layers.Reshape((1, 1, -1))(x1)
    x = keras.layers.Multiply()([x, x2])
    x = keras.layers.Dropout(dropout_rate)(x)
    # output: (bs, 28, 28, 512)
    x = keras.layers.Reshape((-1, 1))(x)
    x = keras.layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(
        x
    )  # Avg Pooling (1D window size = 2)
    x = keras.layers.Reshape((28, 28, -1))(x)
    # output: (bs, 28, 28, 256)
    x = keras.layers.UnitNormalization(axis=-1)(x)  # L2 Normalization
    # output: (bs, 28, 28, 256)
    x = keras.layers.Conv2D(1, (1, 1))(x)
    x = keras.layers.Reshape((28 * 28,))(x)
    x = keras.layers.Softmax()(x)
    x = keras.layers.Reshape((28, 28, 1))(x)  # Visual Attention Weight
    # output: (bs, 28, 28, 1)
    x = keras.layers.Multiply()([x, x2])  # Weighted feature matrix
    # output: (bs, 28, 28, 512)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # output: (bs, 512)

    x = keras.layers.Add()([x, x1])  # Fusion
    # output: (bs, 512)

    # Classification
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(1024, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(2048, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(4096, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    if last_softmax:
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    else:
        outputs = keras.layers.Dense(num_classes)(x)

    # Output
    model = keras.models.Model(inputs=[text_input, image_input], outputs=outputs)

    return model


#
#
#
#
#
#
#


# o net4
def small_net(
    maxlen=15,
    num_words=2000,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.1,
    last_softmax = True
):

    # Inputs
    text_input = keras.layers.Input(shape=(maxlen,), name="text_input")
    image_input = keras.layers.Input(
        shape=(im_size, im_size, num_channels), name="image_input"
    )

    # Textual Feature Extraction
    x1 = keras.layers.Embedding(input_dim=num_words, output_dim=300, mask_zero=False)(
        text_input
    )
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        1024, 6, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.GlobalAveragePooling1D()(x1)
    # output: (bs, 1024)

    # Visual Feature Extraction
    # 1
    x2 = keras.layers.Conv2D(
        64, (3, 3), padding="same", kernel_initializer="he_normal"
    )(image_input)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.SeparableConv2D(
        128,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # 2
    x2 = keras.layers.SeparableConv2D(
        256,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.SeparableConv2D(
        256,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # 3
    x2 = keras.layers.SeparableConv2D(
        512,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.SeparableConv2D(
        1024,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # output: (bs, 28, 28, 1024)

    # Visual attention weight and fusion
    x = keras.layers.Reshape((1, 1, -1))(x1)
    x = keras.layers.Multiply()([x, x2])
    x = keras.layers.Dropout(dropout_rate)(x)
    # output: (bs, 28, 28, 1024)
    x = keras.layers.Reshape((-1, 1))(x)
    x = keras.layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(
        x
    )  # Avg Pooling (1D window size = 2)
    x = keras.layers.Reshape((28, 28, -1))(x)
    # output: (bs, 28, 28, 512)
    x = keras.layers.UnitNormalization(axis=-1)(x)  # L2 Normalization
    # output: (bs, 28, 28, 512)
    x = keras.layers.Conv2D(1, (1, 1))(x)
    x = keras.layers.Reshape((28 * 28,))(x)
    x = keras.layers.Softmax()(x)
    x = keras.layers.Reshape((28, 28, 1))(x)  # Visual Attention Weight
    # output: (bs, 28, 28, 1)
    x = keras.layers.Multiply()([x, x2])  # Weighted feature matrix
    # output: (bs, 28, 28, 1024)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # output: (bs, 1024)

    x = keras.layers.Add()([x, x1])  # Fusion
    # output: (bs, 1024)

    # Classification
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(2048, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(4096, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    if last_softmax:
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    else:
        outputs = keras.layers.Dense(num_classes)(x)

    # Output
    model = keras.models.Model(inputs=[text_input, image_input], outputs=outputs)

    return model


#
#
#
#
#
#
#


# o net5
def big_net(
    maxlen=15,
    num_words=2000,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.1,
    last_softmax = True
):

    # Inputs
    text_input = keras.layers.Input(shape=(maxlen,), name="text_input")
    image_input = keras.layers.Input(
        shape=(im_size, im_size, num_channels), name="image_input"
    )

    # Textual Feature Extraction
    x1 = keras.layers.Embedding(input_dim=num_words, output_dim=300, mask_zero=False)(
        text_input
    )
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        1024, 6, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.GlobalAveragePooling1D()(x1)
    # output: (bs, 1024)

    # Visual Feature Extraction
    # 1
    x2 = keras.layers.Conv2D(
        64, (3, 3), padding="same", kernel_initializer="he_normal"
    )(image_input)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.SeparableConv2D(
        128,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # 2
    x2 = keras.layers.SeparableConv2D(
        256,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.SeparableConv2D(
        256,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # 3
    x2 = keras.layers.SeparableConv2D(
        512,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.SeparableConv2D(
        512,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # 4
    x2 = keras.layers.SeparableConv2D(
        1024,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.SeparableConv2D(
        1024,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # output: (bs, 14, 14, 1024)

    # Visual attention weight and fusion
    x = keras.layers.Reshape((1, 1, -1))(x1)
    x = keras.layers.Multiply()([x, x2])
    x = keras.layers.Dropout(dropout_rate)(x)
    # output: (bs, 14, 14, 1024)
    x = keras.layers.Reshape((-1, 1))(x)
    x = keras.layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(
        x
    )  # Avg Pooling (1D window size = 2)
    x = keras.layers.Reshape((14, 14, -1))(x)
    # output: (bs, 14, 14, 512)
    x = keras.layers.UnitNormalization(axis=-1)(x)  # L2 Normalization
    # output: (bs, 14, 14, 512)
    x = keras.layers.Conv2D(1, (1, 1))(x)
    x = keras.layers.Reshape((14 * 14,))(x)
    x = keras.layers.Softmax()(x)
    x = keras.layers.Reshape((14, 14, 1))(x)  # Visual Attention Weight
    # output: (bs, 14, 14, 1)
    x = keras.layers.Multiply()([x, x2])  # Weighted feature matrix
    # output: (bs, 14, 14, 1024)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # output: (bs, 1024)

    x = keras.layers.Add()([x, x1])  # Fusion
    # output: (bs, 1024)

    # Classification
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(2048, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(4096, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    if last_softmax:
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    else:
        outputs = keras.layers.Dense(num_classes)(x)

    # Output
    model = keras.models.Model(inputs=[text_input, image_input], outputs=outputs)

    return model


#
#
#
#
#
#
#

# come tiny ma con CSA
def tinyCSA_net(
    maxlen=15,
    num_words=2000,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.1,
    last_softmax = True
):

    # Inputs
    text_input = keras.layers.Input(shape=(maxlen,), name="text_input")
    image_input = keras.layers.Input(
        shape=(im_size, im_size, num_channels), name="image_input"
    )

    # Textual Feature Extraction
    x1 = keras.layers.Embedding(input_dim=num_words, output_dim=300, mask_zero=False)(
        text_input
    )
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 6, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.GlobalAveragePooling1D()(x1)
    # output: (bs, 512)

    # Visual Feature Extraction
    x2 = keras.layers.Conv2D(
        64, (3, 3), padding="same", kernel_initializer="he_normal"
    )(image_input)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    x2 = keras.layers.SeparableConv2D(
        256,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    x2 = keras.layers.SeparableConv2D(
        512,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # output: (bs, 28, 28, 512)

    #
    # Fusion and Convolutional Self-Attention
    x = keras.layers.Reshape((1, 1, -1))(x1)
    x = keras.layers.Multiply()([x, x2])
    # output: (bs, 28, 28, 512)    
    
    x_V = keras.layers.DepthwiseConv2D(
        (3,3), padding="same", depthwise_initializer="he_normal")(x)
    x_Q = keras.layers.Conv2D(
        28*28, (1,1), padding="same", kernel_initializer="he_normal")(x_V)
    x_K = keras.layers.Reshape((28*28, -1))(x_Q)
    x_K = keras.layers.Permute((2,1))(x_K)
    x_K = keras.layers.Reshape((28, 28, -1))(x_K)
    
    x_QK = keras.layers.Multiply()([x_Q, x_K])
    x_QK = keras.layers.Conv2D(
        512, (1,1), padding="same", kernel_initializer="he_normal")(x_QK)
    x_QK = keras.layers.Activation("sigmoid")(x_QK)

    x_QKV = keras.layers.Multiply()([x_QK, x_V])
    x_QKV = keras.layers.Conv2D(
        512*3, (1,1), padding="same", kernel_initializer="he_normal")(x_QKV)
    x_QKV = keras.layers.Conv2D(
        512, (1,1), padding="same", kernel_initializer="he_normal")(x_QKV)
    
    x = keras.layers.Add()([x_QKV, x])
    # output: (bs, 28, 28, 512)

    x = keras.layers.GlobalAveragePooling2D()(x)
    # output: (bs, 512)
    #

    # Classification
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(1024, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(2048, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(4096, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    if last_softmax:
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    else:
        outputs = keras.layers.Dense(num_classes)(x)

    # Output
    model = keras.models.Model(inputs=[text_input, image_input], outputs=outputs)

    return model


#
#
#
#
#
#
#

# mix tra tiny e small, 2 CSA
def tinyCSA2_net(
    maxlen=15,
    num_words=2000,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.1,
    last_softmax = True
):

    # Inputs
    text_input = keras.layers.Input(shape=(maxlen,), name="text_input")
    image_input = keras.layers.Input(
        shape=(im_size, im_size, num_channels), name="image_input"
    )

    # Textual Feature Extraction
    x1 = keras.layers.Embedding(input_dim=num_words, output_dim=300, mask_zero=False)(
        text_input
    )
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.Conv1D(
        512, 6, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(x1)
    x1 = keras.layers.ReLU()(x1)
    x1 = keras.layers.GlobalAveragePooling1D()(x1)
    # output: (bs, 512)

    # Visual Feature Extraction
    # 1
    x2 = keras.layers.Conv2D(
        64, (3, 3), padding="same", kernel_initializer="he_normal"
    )(image_input)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.SeparableConv2D(
        128,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # 2
    x2 = keras.layers.SeparableConv2D(
        256,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.SeparableConv2D(
        256,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # 3
    x2 = keras.layers.SeparableConv2D(
        512,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.SeparableConv2D(
        512,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ReLU()(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x2)
    # output: (bs, 28, 28, 512)

    #
    # Fusion and Convolutional Self-Attention
    #1
    x = keras.layers.Reshape((1, 1, -1))(x1)
    x = keras.layers.Multiply()([x, x2])
    # output: (bs, 28, 28, 512)    
    
    x_V = keras.layers.DepthwiseConv2D(
        (3,3), padding="same", depthwise_initializer="he_normal")(x)
    x_Q = keras.layers.Conv2D(
        28*28, (1,1), padding="same", kernel_initializer="he_normal")(x_V)
    x_K = keras.layers.Reshape((28*28, -1))(x_Q)
    x_K = keras.layers.Permute((2,1))(x_K)
    x_K = keras.layers.Reshape((28, 28, -1))(x_K)
    
    x_QK = keras.layers.Multiply()([x_Q, x_K])
    x_QK = keras.layers.Conv2D(
        512, (1,1), padding="same", kernel_initializer="he_normal")(x_QK)
    x_QK = keras.layers.Activation("sigmoid")(x_QK)

    x_QKV = keras.layers.Multiply()([x_QK, x_V])
    x_QKV = keras.layers.Conv2D(
        512*3, (1,1), padding="same", kernel_initializer="he_normal")(x_QKV)
    x_QKV = keras.layers.Conv2D(
        512, (1,1), padding="same", kernel_initializer="he_normal")(x_QKV)
    
    x = keras.layers.Add()([x_QKV, x])
    # output: (bs, 28, 28, 512)

    #2
    x = keras.layers.SeparableConv2D(
        1024, (3, 3), padding="same", depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=2, padding="same")(x)
    # output: (bs, 14, 14, 1024)

    x_V = keras.layers.DepthwiseConv2D(
        (3,3), padding="same", depthwise_initializer="he_normal")(x)
    x_Q = keras.layers.Conv2D(
        14*14, (1,1), padding="same", kernel_initializer="he_normal")(x_V)
    x_K = keras.layers.Reshape((14*14, -1))(x_Q)
    x_K = keras.layers.Permute((2,1))(x_K)
    x_K = keras.layers.Reshape((14, 14, -1))(x_K)
    
    x_QK = keras.layers.Multiply()([x_Q, x_K])
    x_QK = keras.layers.Conv2D(
        1024, (1,1), padding="same", kernel_initializer="he_normal")(x_QK)
    x_QK = keras.layers.Activation("sigmoid")(x_QK)

    x_QKV = keras.layers.Multiply()([x_QK, x_V])
    x_QKV = keras.layers.Conv2D(
        1024*3, (1,1), padding="same", kernel_initializer="he_normal")(x_QKV)
    x_QKV = keras.layers.Conv2D(
        1024, (1,1), padding="same", kernel_initializer="he_normal")(x_QKV)
    
    x = keras.layers.Add()([x_QKV, x])
    # output: (bs, 14, 14, 1024)

    x = keras.layers.GlobalAveragePooling2D()(x)
    # output: (bs, 1024)
    #

    # Classification
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(2048, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(4096, kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    if last_softmax:
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    else:
        outputs = keras.layers.Dense(num_classes)(x)

    # Output
    model = keras.models.Model(inputs=[text_input, image_input], outputs=outputs)

    return model

#
#
#
#
#
#
#
#
#
#

def get_net(
        which_net=None,
        maxlen=15,
        num_words=2000,
        im_size=224,
        num_channels=3,
        num_classes=1000,
        dropout_rate=0.1,
        last_softmax = True
        ):
    
    if which_net == "tiny":
        model = tiny_net(
            maxlen=maxlen,
            num_words=num_words,
            im_size=im_size,
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            last_softmax=last_softmax
        )
    elif which_net == "small":
        model = small_net(
            maxlen=maxlen,
            num_words=num_words,
            im_size=im_size,
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            last_softmax=last_softmax
        )
    elif which_net == "big":
        model = big_net(
            maxlen=maxlen,
            num_words=num_words,
            im_size=im_size,
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            last_softmax=last_softmax
        )
    elif which_net == "tinyCSA":
        model = tinyCSA_net(
            maxlen=maxlen,
            num_words=num_words,
            im_size=im_size,
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            last_softmax=last_softmax
        )
    elif which_net == "tinyCSA2":
        model = tinyCSA2_net(
            maxlen=maxlen,
            num_words=num_words,
            im_size=im_size,
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            last_softmax=last_softmax
        )
    else:
        model = None
        raise TypeError("Choose between the available nets")
    
    return model

#
#
#
#
#

if __name__ == "__main__":
    maxlen = 15
    num_words = 2000
    im_size = 224
    num_channels = 1

    which_net = "tinyCSA2"
    model = get_net(
        which_net=which_net,
        maxlen=maxlen,
        num_words=num_words,
        im_size=im_size,
        num_channels=num_channels
    )

    show = True
    if show:
        print(model.summary())

    save = False
    if save:
        dummy_text = np.random.randint(0, num_words, (1, maxlen))
        dummy_image = np.random.rand(1, im_size, im_size, num_channels)
        pred = model.predict((dummy_text, dummy_image), verbose=0)
        print(pred.shape)

        model.save(f"dummy_{which_net}_net.h5")