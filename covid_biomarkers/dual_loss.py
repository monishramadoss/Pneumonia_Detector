import os
import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input, layers, Model

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

os.makedirs('./dual_loss', exist_ok=True)
os.makedirs('./dual_loss/log_dir', exist_ok=True)
os.makedirs('./dual_loss/ckp', exist_ok=True)


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


def denseunet(inputs, label, filters=32, scale_0 = 1, scale_1=1, stage_1=1, stage_2=3 , stage_3=3, stage_4=1):
    '''Model Creation'''
    # Define model#
    # Define kwargs dictionary#
    kwargs = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
        'use_bias': False}  # zeros, ones, golorit_uniform
    # Define lambda functions#
    conv = lambda x, filters, strides: layers.Conv3D(filters=int(filters), strides=(1, strides, strides), **kwargs)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    relu = lambda x: layers.LeakyReLU()(x)
    # Define stride-1, stride-2 blocks#
    conv1 = lambda filters, x: relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x: relu(norm(conv(x, filters, strides=2)))
    # Define single transpose#
    tran = lambda x, filters, strides: layers.Conv3DTranspose(filters=int(filters), strides=(1, strides, strides),
                                                              **kwargs)(x)
    # Define transpose block#
    tran2 = lambda filters, x: relu(norm(tran(x, filters, strides=2)))
    concat = lambda a, b: layers.Concatenate()([a, b])

    # Define Dense Block#
    def dense_block(filters, input, DB_depth):
        ext = 3 + DB_depth
        outside_layer = input
        for _ in range(0, int(ext)):
            inside_layer = conv1(filters, outside_layer)
            outside_layer = concat(outside_layer, inside_layer)
        return outside_layer

    def td_block(conv1_filters, conv2_filters, input, DB_depth):
        TD = conv1(conv1_filters, conv2(conv2_filters, input))
        DB = dense_block(conv1_filters, TD, DB_depth)
        return DB

    def tu_block(conv1_filters, tran2_filters, input, td_input, DB_depth, skip_DB_depth=1):
        t1 = tran2(tran2_filters, input)
        TU = dense_block(conv1_filters, t1, skip_DB_depth)
        C = concat(TU, td_input)
        DB = dense_block(conv1_filters, C, DB_depth)
        return DB

    TD0 = conv1(filters*scale_0, inputs['dat'])
    TD1 = td_block(filters * 1, filters * 1, TD0, stage_1)
    TD2 = td_block(filters * 1.5, filters * 1, TD1, stage_1)
    TD3 = td_block(filters * 2, filters * 1.5, TD2, stage_1)
    TD4 = td_block(filters * 2.5, filters * 2, TD3, stage_2)
    
    TD5 = td_block(filters * 3, filters * 2.5, TD4, stage_2)
    r1 = layers.Flatten()(TD5)
    r1 = layers.Dense(512)(r1)
    r1 = layers.Dense(512)(r1)
    
    TU1 = tu_block(filters * 2.5, filters * 3, TD5, TD4, stage_2, stage_3)
    TU2 = tu_block(filters * 2, filters * 2.5, TU1, TD3, stage_2, stage_3)
    TU3 = tu_block(filters * 1.5, filters * 2, TU2, TD2, stage_1, stage_4)
    TU4 = tu_block(filters * 1, filters * 1.5, TU3, TD1, stage_1, stage_4)
    TU5 = tran2(filters * scale_1, TU4)
    logits = {}
    logits['ratio'] = layers.Dense(1, activation='sigmoid', name='ratio', use_bias=False)(r1)
    logits[label] = layers.Conv3D(filters=2, name=label, **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits)
    return model

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
    if row['cohort-uci']:
        arrays['xs']['msk-pna'][:] = 0.0
        arrays['xs']['msk-ratio'][:] = 1.0
    else:
        arrays['xs']['msk-pna'][:] = 1.0
        arrays['xs']['msk-ratio'][:] = 0.0
    arrays['xs']['msk-pna'] = arrays['xs']['msk-pna'] > 0
    return arrays

# --- Create a test Client

#client = Client('D:\\data/raw/covid_biomarker/data/ymls/client-dual-256.yml')
configs = {'batch': {'size': 4, 'fold': 0}}
client = Client('/data/raw/covid_biomarker/data/ymls/client-dual-256.yml', configs=configs)

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

model = denseunet(inputs, label,  32)
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
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='ratio_mae', factor=0.8, patience=2, mode="min", verbose=1)
#early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_ratio_mae', patience=20, verbose=1, mode='min', restore_best_weights=False)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./dual_loss/ckp/', monitor='val_ratio_mae', mode='min', save_best_only=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard('./dual_loss/log_dir', profile_batch=0)

model.fit(
    x=gen_train,
    epochs=100,
    steps_per_epoch=1000,
    validation_data=gen_valid,
    validation_steps=500,
    validation_freq=1,
    callbacks=[reduce_lr_callback,  tensorboard_callback, model_checkpoint_callback]
)

model.save('./dual_loss/model.h5', overwrite=True, include_optimizer=False)
