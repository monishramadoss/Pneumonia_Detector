import os, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import losses, optimizers
from tensorflow.keras import Input, Model, models, layers, metrics, backend
from jarvis.train import datasets, custom
from jarvis.train.client import Client
from jarvis.utils.general import overload, tools as jtools
from jarvis.utils.display import imshow


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

os.makedirs('./fusion_loss/', exist_ok=True)
os.makedirs('./fusion_loss/log_dir', exist_ok=True)
os.makedirs('./fusion_loss/ckp', exist_ok=True)


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
conv3t = lambda x, filters : layers.Conv3DTranspose(filters, kernel_size=(1,2,2),
                                            strides=(1,2,2))(x)
convT = lambda x, filters : conv3t(relu(norm(x)), filters)

def densenet(inputs, label='ratio'):
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
    b0 = conv3(inputs['dat'], filters=8)
    b1 = pool(dense_block_(b0))
    b2 = pool(dense_block_(b1))

    dense_block_ = lambda x : dense_block(x, k=24, n=4, b=2)
    b3 = trans(dense_block_(b2), 80)
    b4 = trans(dense_block_(b3), 96)
    b5 = trans(dense_block_(b4), 112)
    b6 = dense_block_(b5)
    p1 = layers.GlobalAveragePooling3D()(b6)
    f1 = layers.Dense(512)(p1)
    return layers.Dense(1, activation='sigmoid', name=label)(f1), b6, p1

@tf.function
def heatmap_crop(image, heatmap, threshold = 0.6):
    # normalize function
    heatmap_norm =  tf.norm(heatmap)    
    where = tf.where(heatmap_norm > threshold)

    xmin = tf.math.reduce_min(where, axis=0)[0][0] * 146
    xmax = tf.math.reduce_max(where, axis=0)[0][0] * 146
    ymin = tf.math.reduce_min(where, axis=0)[0][1] * 146
    ymax = tf.math.reduce_max(where, axis=0)[0][1] * 146

    print(xmin, ymin, xmax, ymin)
    if xmin == xmax:
        xmax = xmax+1
    if ymin == ymax:
        ymax = ymax+1
            
    crop = image[:, xmin:xmax, ymin:ymax, :]
    img = tf.image.resize(crop, (256,256))
    return img
        

def fusion_block(inputs, label):
    f1 = layers.Dense(512)(inputs['dat'])
    f2 = layers.Dense(1, activaiton='sigmoid', name=label)(f1)
    return f2, f2, f2


def make_model(inputs):
    global_layer, global_acitvation_layer, global_pool_layer = densenet(inputs, 'global_ratio')
    heatmap = heatmap_crop(global_acitvation_layer, inputs['dat'])
    local_layer, local_acitvation_layer, local_pool_layer = densenet({'dat', heatmap}, 'local_ratio')
    p1 = tf.concat([global_pool_layer, local_pool_layer])
    fusion_layer, fusion_activation_layer, fusion_pool_layer = fusion_block({'dat': p1}, 'ratio')

    logits = {}
    logits['ratio'] = fusion_layer
    logits['global_ratio'] = global_layer
    logits['local_ratio'] = local_layer 

    model = Model(inputs=inputs, outputs=logits)
    
    # model.compile(
    #     optimizer=optimizers.Adam(learning_rate=2e-4),
    #     loss={'global_ratio': 'mse', 'local_ratio': 'mse', 'fusion_ratio': 'mse' },
    #     metrics='mae',
    #     experimental_run_tf_function=False
    # )

    return model


reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='mae', factor=0.8, patience=2, mode="min", verbose=1)
# early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=20, verbose=0, mode='min', restore_best_weights=False)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./fusion_loss/ckp/', monitor='val_mae', mode='min', save_best_only=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard('./fusion_loss/log_dir', profile_batch=0)


# configs = {'batch': {'size': 4, 'fold': 0}}
# client = Client('./fusion-client-uci-256.yml', configs=configs)
# gen_train, gen_valid = client.create_generators()
# inputs = client.get_inputs(Input)

make_model({'dat': Input(shape=(1,256, 256, 1))})

# model.fit(
#     x=gen_train,
#     epochs=80,
#     steps_per_epoch=1000,
#     validation_data=gen_valid,
#     validation_steps=500,
#     validation_freq=1,
#     callbacks=[reduce_lr_callback, tensorboard_callback, model_checkpoint_callback]
# )

# model.save('./fusion_loss/model.h5', overwrite=True, include_optimizer=False)
