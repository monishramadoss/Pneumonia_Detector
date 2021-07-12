import os
import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input, layers, Model

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


os.makedirs('./log_dir', exist_ok=True)
os.makedirs('./ckp', exist_ok=True)


if tf.__version__[:3] == '2.3':
    tf.compat.v1.disable_eager_execution()

def conv_bn_relu(input, filters, kernel_size=3, stride=1, name=None):
    x = layers.Conv3D(filters, (1, kernel_size, kernel_size), padding='same', strides=(1, stride, stride), name=name, use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    return x

label = 'pna'
def unet(inputs, label, filters = 32, size=4):
    x = inputs['dat']
    encoder_block = []
    filter_start = 4
    #encoder_block
    for en in range(size):
        x = conv_bn_relu(x, filters * filter_start, name='en_'+str(en))
        x = conv_bn_relu(x, filters * filter_start)        
        x = conv_bn_relu(x, filters * filter_start, stride=2)
        filter_start = 2
        encoder_block.append(x)

    x = conv_bn_relu(x, filters * filter_start)
    x = conv_bn_relu(x, filters * filter_start)
    r1 = layers.Flatten()(x)
    r1 = layers.Dense(512)(r1)
    r1 = layers.Dense(512)(r1)

    for de in range(size):
        skip = encoder_block.pop(-1)
        x = layers.concatenate([skip, x])
        x = conv_bn_relu(x, filters * filter_start, name='de_'+str(de))
        x = conv_bn_relu(x, filters * filter_start)
        x = layers.UpSampling3D(size=(1,2,2))(x)
        filter_start = filter_start * 2

    logits = {}
    logits['ratio'] = layers.Dense(1, activation='sigmoid', name='ratio', use_bias=False)(r1)
    logits[label] = layers.Conv3D(2, (1,3,3), padding='same', name=label, use_bias=False)(x)
    return Model(inputs, logits)

from jarvis.train.client import Client
from jarvis.train import models, params, custom
from jarvis.utils.general import overload

try:
    from jarvis.utils.general import gpus
    gpus.autoselect(1)
except:
    pass

@overload(Client)
def preprocess(self, arrays, row, **kwargs):
    if row['corhort-uci']:
        arrays['xs']['msk-pna'][:] = 0.0
        arrays['xs']['msk-ratio'][:] = 1.0
    else:
        arrays['xs']['msk-pna'][:] = 1.0
        arrays['xs']['msk-ratio'][:] = 0.0
    arrays['xs']['msk-pna'] = arrays['xs']['msk-pna'] > 0
    return arrays

# --- Create a test Client

#client = Client('D:\\data/raw/covid_biomarker/data/ymls/client-dual-256.yml')

client = Client('/data/raw/covid_biomarker/data/ymls/client-dual-256.yml')

gen_train, gen_valid = client.create_generators()
inputs = client.get_inputs(Input)



def dsc_soft(weights=None, scale=1.0, epsilon=0.01, cls=1):
    @tf.function
    def dsc(y_true, y_pred):
        true = tf.cast(y_true[..., 0] == cls, tf.float32)
        pred = tf.nn.softmax(y_pred, axis=-1)[..., cls]
        if weights is not None:
            true = true * (weights[...])
            pred = pred * (weights[...])
        A = tf.math.reduce_sum(true * pred) * 2
        B = tf.math.reduce_sum(true) + tf.math.reduce_sum(pred) + epsilon
        return (1.0 - A / B) * scale
    return dsc


def sce(weights=None, scale=1.0):
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    @tf.function
    def sce(y_true, y_pred):
        return loss(y_true=y_true, y_pred=y_pred, sample_weight=weights) * scale
    return sce


def happy_meal(weights=None, alpha=5, beta=1,  epsilon=0.01, cls=1):
    l2 = sce(None, alpha)
    l1 = dsc_soft(weights, beta, epsilon, cls)
    @tf.function
    def calc_loss(y_true, y_pred):
        return l2(y_true, y_pred) + l1(y_true, y_pred)
    return calc_loss

model = unet(inputs,label,  32)
#model = da_unet(inputs)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss={ label: happy_meal(weights=inputs['msk-pna'], alpha=1.0, beta=1.0),
            'ratio': custom.mse(weights=inputs['msk-ratio'])},
    metrics={ label: custom.dsc(weights=inputs['msk-pna']),
            'ratio': custom.mae(weights=inputs['msk-ratio']), },
    experimental_run_tf_function=False
)

client.load_data_in_memory()
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc_1', factor=0.8, patience=2, mode="max", verbose=1)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_dsc_1', patience=5, verbose=0, mode='max', restore_best_weights=False)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./ckp/', monitor='val_dsc_1', mode='max', save_best_only=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard('./log_dir', profile_batch=0)

model.fit(
    x=gen_train,
    epochs=40,
    steps_per_epoch=1000,
    validation_data=gen_valid,
    validation_steps=500,
    validation_freq=1,
    callbacks=[reduce_lr_callback, early_stop_callback, tensorboard_callback, model_checkpoint_callback]
)

model.save('./model.h5', overwrite=True, include_optimizer=False)
