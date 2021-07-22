import os, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import losses, optimizers
from tensorflow.keras import Input, Model, models, layers, metrics, backend
from jarvis.train import datasets, custom
from jarvis.train.client import Client
from jarvis.utils.general import overload, tools as jtools
import matplotlib.pyplot as plt


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

os.makedirs('./fusion_loss/', exist_ok=True)
os.makedirs('./fusion_loss/log_dir', exist_ok=True)
os.makedirs('./fusion_loss/ckp', exist_ok=True)
file_writer = tf.summary.create_file_writer('./fusion_loss/log_dir')


try:
    from jarvis.utils.general import gpus
    gpus.autoselect(1)
except:
    pass

conv3 = lambda x, filters : layers.Conv3D(kernel_size=(1, 3, 3), filters=filters, 
                                          strides=1, padding='same')(x)
conv1 = lambda x, filters : layers.Conv3D(kernel_size=1, filters=filters, 
                                          strides=1, padding='same')(x)
pool = lambda x : layers.AveragePooling3D(pool_size=(1, 2, 2),strides=(1, 2, 2),
                                          padding='same')(x)
norm = lambda x : layers.BatchNormalization()(x)
relu = lambda x : layers.LeakyReLU()(x)
concat = lambda a, b : layers.Concatenate()([a, b])
dense = lambda x, k : conv3(relu(norm(x)), filters=k)
bneck = lambda x, b : conv1(relu(norm(x)), filters=b)
trans = lambda x, b : pool(bneck(x, b))
conv3t = lambda x, filters : layers.Conv3DTranspose(filters, kernel_size=(1, 2, 2),
                                            strides=(1, 2, 2))(x)
convT = lambda x, filters : conv3t(relu(norm(x)), filters)

def densenet(inputs, input_label='dat',  label='ratio'):
    def dense_block(x, k=8, n=3, b=1, verbose=False):
        ds_layer = None
        for i in range(n):
            cc_layer = concat(cc_layer, ds_layer) if ds_layer is not None else x
            bn_layer = bneck(cc_layer, b * k) if i >= b else cc_layer
            ds_layer = dense(bn_layer, k)
            if verbose:
                print('Creating layer {:02d}: cc_layer = {}'.format(i, cc_layer.shape))
                print('Creating layer {:02d}: bn_layer = {}'.format(i, bn_layer.shape))
                print('Creating layer {:02d}: ds_layer = {}'.format(i, ds_layer.shape))    
        return concat(cc_layer, ds_layer)

    dense_block_ = lambda x : dense_block(x, k=16, n=3, b=1)
    b0 = conv3(inputs[input_label], filters=8)
    b1 = pool(dense_block_(b0))
    b2 = pool(dense_block_(b1))

    dense_block_ = lambda x : dense_block(x, k=24, n=4, b=2)
    b3 = trans(dense_block_(b2), 80)
    b4 = trans(dense_block_(b3), 96)
    b5 = trans(dense_block_(b4), 112)
    b6 = dense_block_(b5)
    p1 = layers.GlobalAveragePooling3D()(b6)
    f1 = layers.Dense(512)(p1)
    logits = {
        label: layers.Dense(1, activation='sigmoid', name=label)(f1),
        'act': b6,
        'pool': p1
    }
    return Model(inputs={input_label: inputs[input_label]}, outputs=logits)

def fusion_block(inputs, global_model, local_model, label):
    p1 = global_model.get_layer(index=-2).output
    p2 = local_model.get_layer(index=-2).output
    x = layers.concatenate([p1, p2])
    f1 = layers.Dense(512)(x)  
    f2 = layers.Dense(1, activation='sigmoid', name=label)(f1)
    
    logits = {
        label: f2,
    }
    return Model(inputs=inputs, outputs=logits)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='mae', factor=0.8, patience=2, mode="min", verbose=1)
# early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=20, verbose=0, mode='min', restore_best_weights=False)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./fusion_loss/ckp/', monitor='val_mae', mode='min', save_best_only=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard('./fusion_loss/log_dir', profile_batch=0)

configs = {'batch': {'size': 4, 'fold': 0}}
client = Client('./ymls/fusion-client-uci-256.yml', configs=configs)
gen_train, gen_valid = client.create_generators()
inputs = client.get_inputs(Input)

global_model = densenet(inputs, 'dat', 'ratio')
local_model = densenet(inputs, 'local-dat', 'ratio')
fusion_model = fusion_block(inputs, global_model, local_model,'ratio')

def heatmap_crop(img, heatmap, threshold=0.6):
    heatmap = np.squeeze(heatmap)
    heatmap_one = np.abs(heatmap)
    heatmap_two = np.amax(heatmap, axis=-1)
    max1 = np.amax(heatmap_two)
    min1 = np.amin(heatmap_two)
    heatmap_two = (heatmap_two - min1) / (max1 - min1)
    heatmap_msk = heatmap_two > threshold
    
    where = np.argwhere(heatmap_msk)
    xmin = np.amin(where, axis=0)[0]
    xmax = np.amax(where, axis=0)[0]
    ymin = np.amin(where, axis=0)[1]
    ymax = np.amax(where, axis=0)[1]
    
    if xmin == xmax:
        xmax += 1
    if ymin == ymax:
        ymax += 1
    box = tf.stack([ymin, xmin, ymax, xmax], axis=0)
    box = tf.math.divide(box, 8)
    box = box.numpy().reshape([ 1, 4])
    new_img = tf.image.crop_and_resize(img, box, [0], [256,256])
    
    plt.imsave('msk.png', np.squeeze(heatmap_msk))
    plt.imsave('new_img.png', np.squeeze(new_img))
    plt.imsave('img.png', np.squeeze(img))    
    new_img = tf.reshape(new_img, img.shape)
    return new_img.numpy()

@overload(Client)
def preprocess(self, arrays, **kwargs):
    dat = arrays['xs']['dat'] 
    arrays['xs']['local-dat'] = dat
    global_logit = global_model.predict({'dat': np.expand_dims(arrays['xs']['dat'], axis=0)})   
    if type(global_logit) is list:
        actmap = global_logit[0]
        p1 = global_logit[1]
        logit = global_logit[2]
    elif type(global_logit) is dict:
        logit = global_logit['ratio']
        actmap = global_logit['act']
        p1 = global_logit['pool']
            
    arrays['xs']['local-dat'] = heatmap_crop(arrays['xs']['dat'], actmap, 0.6)
    
#     local_logit = local_model.predict({'local-dat': np.expand_dims(arrays['xs']['local-dat'], axis=0)})
#     if type(local_logit) is list:
#         heatmap = local_logit[0]
#         p2 = local_logit[1]    
#         logit = local_logit[2]
#     elif type(local_logit) is dict:
#         logit = local_logit['ratio']
#         heatmap = local_logit['act']
#         p2 = local_logit['pool']
   
    return arrays

xs, ys = next(gen_train)

global_model.compile(
    optimizer=optimizers.Adam(learning_rate=2e-4),
    loss={'ratio': 'mse'},
    metrics={'ratio': 'mae'},
    experimental_run_tf_function=False
)

global_model.fit(
    x=gen_train,
    epochs=4,
    steps_per_epoch=1000,
    validation_data=gen_valid,
    validation_steps=500,
    validation_freq=1,
    callbacks=[reduce_lr_callback, tensorboard_callback, model_checkpoint_callback]
)

local_model.compile(
    optimizer=optimizers.Adam(learning_rate=2e-4),
    loss={'ratio': 'mse'},
    metrics={'ratio': 'mae'},
    experimental_run_tf_function=False
)

local_model.fit(
    x=gen_train,
    epochs=4,
    steps_per_epoch=1000,
    validation_data=gen_valid,
    validation_steps=500,
    validation_freq=1,
    callbacks=[reduce_lr_callback, tensorboard_callback]
)


fusion_model.compile(
    optimizer=optimizers.Adam(learning_rate=2e-4),
    loss={'ratio': 'mse'},
    metrics={'ratio': 'mae'},
    experimental_run_tf_function=False
)


fusion_model.fit(
    x=gen_train,
    epochs=80,
    steps_per_epoch=1000,
    validation_data=gen_valid,
    validation_steps=500,
    validation_freq=1,
    callbacks=[reduce_lr_callback, tensorboard_callback]
)

global_model.save('./fusion_loss/model.h5', overwrite=True, include_optimizer=False)
