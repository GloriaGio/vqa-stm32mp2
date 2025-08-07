import json

from tensorflow import keras
import numpy as np

from models.get_glove import get_GloVe_emb

#
#
#


def MFB_Baseline(
    k=5,
    output_MFB=1024,
    maxlen=15,
    num_words=5000,
    emb_dim=50,
    glove_emb=None,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.3,
    last_softmax=True,
):
    # Inputs
    text_input = keras.layers.Input(shape=(maxlen,), name="text_input")
    image_input = keras.layers.Input(
        shape=(im_size, im_size, num_channels), name="image_input"
    )

    # Textual Feature Extraction
    q_emb = keras.layers.Embedding(
        input_dim=num_words, output_dim=emb_dim, mask_zero=False
    )(text_input)
    if glove_emb is not None:
        q_glove_emb = keras.layers.Embedding(
            input_dim=num_words, output_dim=emb_dim, mask_zero=False, trainable=False
        )(text_input)
        q_emb = keras.layers.Concatenate(axis=-1)([q_emb, q_glove_emb])

    q_feat1 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(q_emb)
    q_feat1 = keras.layers.ReLU()(q_feat1)
    q_feat1 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(q_feat1)
    q_feat1 = keras.layers.ReLU()(q_feat1)
    q_feat1 = keras.layers.Conv1D(
        512,
        6,
        dilation_rate=2,
        padding="causal",
        kernel_initializer="he_normal",
    )(q_feat1)
    q_feat1 = keras.layers.ReLU()(q_feat1)

    q_feat2 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(q_feat1)
    q_feat2 = keras.layers.ReLU()(q_feat2)
    q_feat2 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(q_feat2)
    q_feat2 = keras.layers.ReLU()(q_feat2)
    q_feat2 = keras.layers.Conv1D(
        512,
        6,
        dilation_rate=2,
        padding="causal",
        kernel_initializer="he_normal",
    )(q_feat2)
    q_feat2 = keras.layers.ReLU()(q_feat2)

    q_feat = keras.layers.Concatenate(axis=-1)([q_feat1, q_feat2])
    q_feat = keras.layers.GlobalAveragePooling1D()(q_feat)

    # Image Feature Extraction
    mobilenet_base = keras.applications.MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_preprocessing=False,
    )
    mobilenet_base.trainable = False
    im_feat = mobilenet_base(image_input, training=False)
    im_feat = keras.layers.GlobalAveragePooling2D()(im_feat)
    im_feat = keras.layers.UnitNormalization(axis=-1)(im_feat)

    # MFB pooling
    q_mfb = keras.layers.Dense(output_MFB * k, kernel_initializer="he_normal")(q_feat)
    im_mfb = keras.layers.Dense(output_MFB * k, kernel_initializer="he_normal")(im_feat)
    fus_mfb = keras.layers.Multiply()([q_mfb, im_mfb])
    fus_mfb = keras.layers.Dropout(dropout_rate)(fus_mfb)
    fus_mfb = keras.layers.Reshape((-1, 1))(fus_mfb)
    fus_mfb = keras.layers.AveragePooling1D(pool_size=k, strides=k, padding="same")(
        fus_mfb
    )
    fus_mfb = keras.layers.Reshape((-1,))(fus_mfb)
    # no pow norm
    fus_mfb = keras.layers.UnitNormalization(axis=-1)(fus_mfb)

    # Classification
    fus_feat = keras.layers.Dropout(dropout_rate)(fus_mfb)
    fus_feat = keras.layers.Dense(2048, kernel_initializer="he_normal")(fus_feat)
    fus_feat = keras.layers.BatchNormalization()(fus_feat)
    fus_feat = keras.layers.ReLU()(fus_feat)

    if last_softmax:
        outputs = keras.layers.Dense(num_classes, activation="softmax")(fus_feat)
    else:
        outputs = keras.layers.Dense(num_classes)(fus_feat)

    # Output
    model = keras.models.Model(inputs=[text_input, image_input], outputs=outputs)

    return model


#
#
#


def MFB_Attention(
    k=5,
    output_MFB=1024,
    num_glimps=2,
    maxlen=15,
    num_words=5000,
    emb_dim=50,
    glove_emb=None,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.3,
    last_softmax=True,
):
    # Inputs
    text_input = keras.layers.Input(shape=(maxlen,), name="text_input")
    image_input = keras.layers.Input(
        shape=(im_size, im_size, num_channels), name="image_input"
    )

    # Textual Feature Extraction
    q_emb = keras.layers.Embedding(
        input_dim=num_words, output_dim=emb_dim, mask_zero=False
    )(text_input)
    if glove_emb is not None:
        q_glove_emb = keras.layers.Embedding(
            input_dim=num_words, output_dim=emb_dim, mask_zero=False, trainable=False
        )(text_input)
        q_emb = keras.layers.Concatenate(axis=-1)([q_emb, q_glove_emb])

    q_feat1 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(q_emb)
    q_feat1 = keras.layers.ReLU()(q_feat1)
    q_feat1 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(q_feat1)
    q_feat1 = keras.layers.ReLU()(q_feat1)
    q_feat1 = keras.layers.Conv1D(
        512,
        6,
        dilation_rate=2,
        padding="causal",
        kernel_initializer="he_normal",
    )(q_feat1)
    q_feat1 = keras.layers.ReLU()(q_feat1)

    q_feat2 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(q_feat1)
    q_feat2 = keras.layers.ReLU()(q_feat2)
    q_feat2 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(q_feat2)
    q_feat2 = keras.layers.ReLU()(q_feat2)
    q_feat2 = keras.layers.Conv1D(
        512,
        6,
        dilation_rate=2,
        padding="causal",
        kernel_initializer="he_normal",
    )(q_feat2)
    q_feat2 = keras.layers.ReLU()(q_feat2)

    q_feat = keras.layers.Concatenate(axis=-1)([q_feat1, q_feat2])
    q_feat = keras.layers.GlobalAveragePooling1D()(q_feat)

    # Image Feature Extraction
    mobilenet_base = keras.applications.MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_preprocessing=False,
    )
    mobilenet_base.trainable = False
    im_feat = mobilenet_base(image_input, training=False)

    # MFB pooling 1
    q_mfb1 = keras.layers.Dense(output_MFB * k, kernel_initializer="he_normal")(q_feat)
    q_mfb1 = keras.layers.Reshape((1, 1, -1))(q_mfb1)
    im_mfb1 = keras.layers.Conv2D(
        output_MFB * k, 1, padding="same", kernel_initializer="he_normal"
    )(im_feat)
    fus_mfb1 = keras.layers.Multiply()([q_mfb1, im_mfb1])
    fus_mfb1 = keras.layers.Dropout(dropout_rate)(fus_mfb1)
    fus_mfb1 = keras.layers.Reshape((-1, 1))(fus_mfb1)
    fus_mfb1 = keras.layers.AveragePooling1D(pool_size=k, strides=k, padding="same")(
        fus_mfb1
    )
    fus_mfb1 = keras.layers.Reshape(
        (
            7,
            7,
            -1,
        )
    )(fus_mfb1)
    # no pow norm
    fus_mfb1 = keras.layers.UnitNormalization(axis=-1)(fus_mfb1)

    # Image Attention
    im_att1 = keras.layers.Conv2D(
        256, 1, padding="same", kernel_initializer="he_normal"
    )(fus_mfb1)
    im_att1 = keras.layers.ReLU()(im_att1)
    for i in range(num_glimps):
        im_att2 = keras.layers.Conv2D(
            1, 1, padding="same", kernel_initializer="he_normal"
        )(im_att1)
        im_att2 = keras.layers.Softmax(axis=1)(im_att2)
        im_att2 = keras.layers.Multiply()([im_att2, fus_mfb1])
        im_att2 = keras.layers.GlobalAveragePooling2D()(im_att2)

        if i == 0:
            im_att = im_att2
        else:
            im_att = keras.layers.Concatenate(axis=-1)([im_att, im_att2])

    # MFB pooling 2
    q_mfb2 = keras.layers.Dense(output_MFB * k, kernel_initializer="he_normal")(q_feat)
    im_mfb2 = keras.layers.Dense(output_MFB * k, kernel_initializer="he_normal")(im_att)
    fus_mfb2 = keras.layers.Multiply()([q_mfb2, im_mfb2])
    fus_mfb2 = keras.layers.Dropout(dropout_rate)(fus_mfb2)
    fus_mfb2 = keras.layers.Reshape((-1, 1))(fus_mfb2)
    fus_mfb2 = keras.layers.AveragePooling1D(pool_size=k, strides=k, padding="same")(
        fus_mfb2
    )
    fus_mfb2 = keras.layers.Reshape((-1,))(fus_mfb2)
    # no pow norm
    fus_mfb2 = keras.layers.UnitNormalization(axis=-1)(fus_mfb2)

    # Classification
    fus_feat = keras.layers.Dropout(dropout_rate)(fus_mfb2)
    fus_feat = keras.layers.Dense(2048, kernel_initializer="he_normal")(fus_feat)
    fus_feat = keras.layers.BatchNormalization()(fus_feat)
    fus_feat = keras.layers.ReLU()(fus_feat)

    if last_softmax:
        outputs = keras.layers.Dense(num_classes, activation="softmax")(fus_feat)
    else:
        outputs = keras.layers.Dense(num_classes)(fus_feat)

    # Output
    model = keras.models.Model(inputs=[text_input, image_input], outputs=outputs)

    return model


#
#
#


def MFB_CoAttention(
    k=5,
    output_MFB=1024,
    num_glimps=2,
    maxlen=15,
    num_words=5000,
    emb_dim=50,
    glove_emb=None,
    im_size=224,
    num_channels=3,
    num_classes=1000,
    dropout_rate=0.3,
    last_softmax=True,
):
    # Inputs
    text_input = keras.layers.Input(shape=(maxlen,), name="text_input")
    image_input = keras.layers.Input(
        shape=(im_size, im_size, num_channels), name="image_input"
    )

    # Textual Feature Extraction
    q_emb = keras.layers.Embedding(
        input_dim=num_words, output_dim=emb_dim, mask_zero=False
    )(text_input)
    if glove_emb is not None:
        q_glove_emb = keras.layers.Embedding(
            input_dim=num_words, output_dim=emb_dim, mask_zero=False, trainable=False
        )(text_input)
        q_emb = keras.layers.Concatenate(axis=-1)([q_emb, q_glove_emb])

    q_feat1 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(q_emb)
    q_feat1 = keras.layers.ReLU()(q_feat1)
    q_feat1 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(q_feat1)
    q_feat1 = keras.layers.ReLU()(q_feat1)
    q_feat1 = keras.layers.Conv1D(
        512,
        6,
        dilation_rate=2,
        padding="causal",
        kernel_initializer="he_normal",
    )(q_feat1)
    q_feat1 = keras.layers.ReLU()(q_feat1)

    q_feat2 = keras.layers.Conv1D(
        512, 3, dilation_rate=1, padding="causal", kernel_initializer="he_normal"
    )(q_feat1)
    q_feat2 = keras.layers.ReLU()(q_feat2)
    q_feat2 = keras.layers.Conv1D(
        512, 3, dilation_rate=2, padding="causal", kernel_initializer="he_normal"
    )(q_feat2)
    q_feat2 = keras.layers.ReLU()(q_feat2)
    q_feat2 = keras.layers.Conv1D(
        512,
        6,
        dilation_rate=2,
        padding="causal",
        kernel_initializer="he_normal",
    )(q_feat2)
    q_feat2 = keras.layers.ReLU()(q_feat2)

    q_feat = keras.layers.Concatenate(axis=-1)([q_feat1, q_feat2])

    # Question Attention
    q_att1 = keras.layers.Conv1D(
        256, 1, dilation_rate=1, padding="same", kernel_initializer="he_normal"
    )(q_feat)
    q_att1 = keras.layers.ReLU()(q_att1)
    for i in range(num_glimps):
        q_att2 = keras.layers.Conv1D(
            1, 1, dilation_rate=1, padding="same", kernel_initializer="he_normal"
        )(q_att1)
        q_att2 = keras.layers.Softmax(axis=1)(q_att2)
        q_att2 = keras.layers.Multiply()([q_att2, q_feat])
        q_att2 = keras.layers.GlobalAveragePooling1D()(q_att2)

        if i == 0:
            q_att = q_att2
        else:
            q_att = keras.layers.Concatenate(axis=-1)([q_att, q_att2])

    # Image Feature Extraction
    mobilenet_base = keras.applications.MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_preprocessing=False,
    )
    mobilenet_base.trainable = False
    im_feat = mobilenet_base(image_input, training=False)

    # MFB pooling 1
    q_mfb1 = keras.layers.Dense(output_MFB * k, kernel_initializer="he_normal")(q_att)
    q_mfb1 = keras.layers.Reshape((1, 1, -1))(q_mfb1)
    im_mfb1 = keras.layers.Conv2D(
        output_MFB * k, 1, padding="same", kernel_initializer="he_normal"
    )(im_feat)
    fus_mfb1 = keras.layers.Multiply()([q_mfb1, im_mfb1])
    fus_mfb1 = keras.layers.Dropout(dropout_rate)(fus_mfb1)
    fus_mfb1 = keras.layers.Reshape((-1, 1))(fus_mfb1)
    fus_mfb1 = keras.layers.AveragePooling1D(pool_size=k, strides=k, padding="same")(
        fus_mfb1
    )
    fus_mfb1 = keras.layers.Reshape(
        (
            7,
            7,
            -1,
        )
    )(fus_mfb1)
    # no pow norm
    fus_mfb1 = keras.layers.UnitNormalization(axis=-1)(fus_mfb1)

    # Image Attention
    im_att1 = keras.layers.Conv2D(
        256, 1, padding="same", kernel_initializer="he_normal"
    )(fus_mfb1)
    im_att1 = keras.layers.ReLU()(im_att1)
    for i in range(num_glimps):
        im_att2 = keras.layers.Conv2D(
            1, 1, padding="same", kernel_initializer="he_normal"
        )(im_att1)
        im_att2 = keras.layers.Softmax(axis=1)(im_att2)
        im_att2 = keras.layers.Multiply()([im_att2, fus_mfb1])
        im_att2 = keras.layers.GlobalAveragePooling2D()(im_att2)

        if i == 0:
            im_att = im_att2
        else:
            im_att = keras.layers.Concatenate(axis=-1)([im_att, im_att2])

    # MFB pooling 2
    q_mfb2 = keras.layers.Dense(output_MFB * k, kernel_initializer="he_normal")(q_att)
    im_mfb2 = keras.layers.Dense(output_MFB * k, kernel_initializer="he_normal")(im_att)
    fus_mfb2 = keras.layers.Multiply()([q_mfb2, im_mfb2])
    fus_mfb2 = keras.layers.Dropout(dropout_rate)(fus_mfb2)
    fus_mfb2 = keras.layers.Reshape((-1, 1))(fus_mfb2)
    fus_mfb2 = keras.layers.AveragePooling1D(pool_size=k, strides=k, padding="same")(
        fus_mfb2
    )
    fus_mfb2 = keras.layers.Reshape((-1,))(fus_mfb2)
    # no pow norm
    fus_mfb2 = keras.layers.UnitNormalization(axis=-1)(fus_mfb2)

    # Classification
    fus_feat = keras.layers.Dropout(dropout_rate)(fus_mfb2)
    fus_feat = keras.layers.Dense(2048, kernel_initializer="he_normal")(fus_feat)
    fus_feat = keras.layers.BatchNormalization()(fus_feat)
    fus_feat = keras.layers.ReLU()(fus_feat)

    if last_softmax:
        outputs = keras.layers.Dense(num_classes, activation="softmax")(fus_feat)
    else:
        outputs = keras.layers.Dense(num_classes)(fus_feat)

    # Output
    model = keras.models.Model(inputs=[text_input, image_input], outputs=outputs)

    return model


#
#
#


def get_model(config, tokenizer_path=None):
    # Initialize and return the model specified in the config.
    # Loads GloVe embeddings if enabled and available, and supports different MFB-based architectures.

    # Load GloVe embeddings if enabled and tokenizer is provided
    use_glove = config["model"]["use_glove"]
    if use_glove and tokenizer_path is not None:
        with open(tokenizer_path) as file:
            word_index = json.load(file)

        glove_emb, _ = get_GloVe_emb(
            config["paths"]["glove_path"],
            dim=config["model"]["embedding_dim"],
            word_index=word_index,
        )
    else:
        glove_emb = None

    # Determine whether to apply softmax at the end, based on knowledge distillation setting
    last_softmax = not config["training"]["knowledge_distillation"]

    # Instantiate the selected model architecture with parameters from config
    # Raise an error if an unsupported architecture is specified
    if config["model"]["model_architecture"] == "MFBBaseline":
        config["model"]["num_attention_glimps"] = -1
        model = MFB_Baseline(
            k=config["model"]["k_window"],
            output_MFB=config["model"]["output_MFB"],
            maxlen=config["model"]["max_length"],
            num_words=config["model"]["num_vocab_words"],
            emb_dim=config["model"]["embedding_dim"],
            glove_emb=glove_emb,
            im_size=config["model"]["image_size"],
            num_channels=config["model"]["num_channels"],
            num_classes=config["model"]["num_classes"],
            dropout_rate=config["model"]["dropout_rate"],
            last_softmax=last_softmax,
        )
    elif config["model"]["model_architecture"] == "MFBAttention":
        num_glimps = config["model"]["num_attention_glimps"]
        if num_glimps < 1:
            raise KeyError(f"invalid number of attention glimps ({num_glimps})")
        model = MFB_Attention(
            k=config["model"]["k_window"],
            output_MFB=config["model"]["output_MFB"],
            num_glimps=config["model"]["num_attention_glimps"],
            maxlen=config["model"]["max_length"],
            num_words=config["model"]["num_vocab_words"],
            emb_dim=config["model"]["embedding_dim"],
            glove_emb=glove_emb,
            im_size=config["model"]["image_size"],
            num_channels=config["model"]["num_channels"],
            num_classes=config["model"]["num_classes"],
            dropout_rate=config["model"]["dropout_rate"],
            last_softmax=last_softmax,
        )
    elif config["model"]["model_architecture"] == "MFBCoAttention":
        num_glimps = config["model"]["num_attention_glimps"]
        if num_glimps < 1:
            raise KeyError(f"invalid number of attention glimps ({num_glimps})")
        model = MFB_CoAttention(
            k=config["model"]["k_window"],
            output_MFB=config["model"]["output_MFB"],
            num_glimps=config["model"]["num_attention_glimps"],
            maxlen=config["model"]["max_length"],
            num_words=config["model"]["num_vocab_words"],
            emb_dim=config["model"]["embedding_dim"],
            glove_emb=glove_emb,
            im_size=config["model"]["image_size"],
            num_channels=config["model"]["num_channels"],
            num_classes=config["model"]["num_classes"],
            dropout_rate=config["model"]["dropout_rate"],
            last_softmax=last_softmax,
        )
    else:
        arch = config["model"]["model_architecture"]
        raise KeyError(f"{arch} architecture does not exits")

    print("Number of parameters:", model.count_params())
    return model
