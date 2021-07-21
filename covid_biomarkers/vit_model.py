import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, preprocessing

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

os.makedirs('./vit_loss/', exist_ok=True)
os.makedirs('./vit_loss/log_dir', exist_ok=True)
os.makedirs('./vit_loss/ckp', exist_ok=True)


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, 
                                      (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(mlp_dim, activation=gelu),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


class VisionTransformer(tf.keras.Model):
    def __init__(self,  image_size=256, patch_size=16, num_layers=4, d_model=64, num_heads=4,
        mlp_dim=1024, channels=1, dropout=0.1):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches + 1, d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        self.patch_proj = layers.Dense(d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(mlp_dim, activation=gelu),
                layers.Dropout(dropout),                
            ]
        )

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x
    
def create_vit_classifier(inputs):     
    vit = VisionTransformer(image_size=256, patch_size=8, num_layers=8, d_model=128, num_heads=16, mlp_dim=1024, channels=1, dropout=0.1)
    
    features = vit(inputs['dat'])
    logits = {}
    logits['ratio'] = layers.Dense(1, name='ratio')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model, gen_train, gen_valid):
    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 400 #100

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, 
    )

    model.compile(
        optimizer=optimizer,
        loss={'ratio':'mse'},
        metrics={'ratio':['mae']},
    )

    checkpoint_filepath = "./vit_loss/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_mse",
        save_best_only=True,
        save_weights_only=True,
    )
    tensorboard_callback = keras.callbacks.TensorBoard('./vit_loss/log_dir/', profile_batch=0)
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='mae', factor=1-weight_decay, patience=2, mode="min", verbose=1)


    history = model.fit(
        x=gen_train,              
        steps_per_epoch=4000,
        epochs=num_epochs,
        validation_data=gen_valid,
        validation_steps=500,
        validation_freq=1,
        callbacks=[checkpoint_callback, reduce_lr_callback, tensorboard_callback],
    )

    model.load_weights(checkpoint_filepath)    
    return history



from jarvis.train.client import Client
from tensorflow.keras import Input
try:
    from jarvis.utils.general import gpus
    gpus.autoselect(4)
except:
    pass

configs = {'batch': {'size': 32, 'fold': 0}}
client = Client('./vit-client-uci-256.yml', configs=configs)
gen_train, gen_valid = client.create_generators()
inputs = client.get_inputs(Input)


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    vit_classifier = create_vit_classifier(inputs)
    history = run_experiment(vit_classifier, gen_train, gen_valid)
vit_classifier.save('./vit_loss/model.h5', overwrite=True)
