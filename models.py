import keras
import keras.backend as K
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Model
from keras.layers import Input, Dense, Activation, MaxPooling2D, Conv2D, BatchNormalization
from keras.layers import Concatenate, Cropping2D, Lambda

from keras.losses import categorical_crossentropy

from unet import level_block
from utils import load_nlcd_stats

def jaccard_loss(y_true, y_pred, smooth=0.001, num_classes=7):                                                                              
    intersection = y_true * y_pred                                                                                                          
    sum_ = y_true + y_pred                                                                                                                  
    jac = K.sum(intersection + smooth, axis=(0,1,2)) / K.sum(sum_ - intersection + smooth, axis=(0,1,2))                                    
    return (1.0 - K.sum(jac) / num_classes)

def accuracy(y_true, y_pred):
    num = K.sum(K.cast(K.equal(K.argmax(y_true[:,:,:,1:], axis=-1), K.argmax(y_pred[:,:,:,1:], axis=-1)), dtype="float32") * y_true[:,:,:,0])
    denom = K.sum(y_true[:,:,:,0])
    return num / (denom + 1) # make sure we don't get divide by zero

def hr_loss(boundary=0):
    '''The first channel of y_true should be all 1's if we want to use hr_loss, or all 0's if we don't want to use hr_loss
    '''
    def loss(y_true, y_pred):
        return categorical_crossentropy(y_true[:,:,:,1:], y_pred[:,:,:,1:]) * y_true[:,:,:,0]
    return loss

def sr_loss(nlcd_class_weights, nlcd_means, nlcd_vars, boundary=0):
    '''Calculate superres loss according to ICLR paper'''

    def ddist(prediction, c_interval_center, c_interval_radius):
        return K.relu(K.abs(prediction - c_interval_center) - c_interval_radius)
    
    def loss(y_true, y_pred):
        
        super_res_crit = 0
        mask_size =  K.expand_dims(K.sum(y_true, axis=(1,2,3)) + 10, -1) # shape 16x1

        for nlcd_idx in range(nlcd_class_weights.shape[0]):

            c_mask = K.expand_dims(y_true[:,:,:,nlcd_idx], -1) # shape 16x240x240x1
            c_mask_size = K.sum(c_mask, axis=(1,2)) + 0.000001 # shape 16x1
            
            c_interval_center = nlcd_means[nlcd_idx] # shape 5,
            c_interval_radius = nlcd_vars[nlcd_idx] # shape 5,

            masked_probs = y_pred * c_mask # (16x240x240x5) * (16x240x240x1) --> shape (16x240x240x5)
            
            # Mean mean of predicted distribution
            mean = K.sum(masked_probs, axis=(1,2)) / c_mask_size # (16x5) / (16,1) --> shape 16x5
            
            # Mean var of predicted distribution
            var = K.sum(masked_probs * (1.-masked_probs), axis=(1,2)) / (c_mask_size * c_mask_size) # (16x5) / (16,1) --> shape 16x5
            
            c_super_res_crit = K.square(ddist(mean, c_interval_center, c_interval_radius)) # calculate numerator of equation 7 in ICLR paper
            c_super_res_crit = c_super_res_crit / (var + (c_interval_radius * c_interval_radius) + 0.000001) # calculate denominator
            c_super_res_crit = c_super_res_crit + K.log(var + 0.03) # calculate log term
            c_super_res_crit = c_super_res_crit * (c_mask_size / mask_size) * nlcd_class_weights[nlcd_idx] # weight by the fraction of NLCD pixels and the NLCD class weight
            
            super_res_crit = super_res_crit + c_super_res_crit # accumulate
        
        super_res_crit = K.sum(super_res_crit, axis=1) # sum superres loss across highres classes
        return super_res_crit
    
    return loss

def baseline_model_landcover(input_shape, num_classes, lr=0.003, loss=None):
    inputs = Input(input_shape)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(inputs)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(inputs)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(inputs)
    #x4 = Conv2D(64, kernel_size=(5,5), strides=(5,5), padding="same", activation="relu")(inputs)
    #x5 = Conv2D(64, kernel_size=(3,3), strides=(2,2), padding="same", activation="relu")(inputs)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(32, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding="same", activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(lr=lr)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    else:
        print("Loss function not specified, model not compiled")
    return model


def extended_model_landcover(input_shape, num_classes, lr=0.003, loss=None):
    inputs = Input(input_shape)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(inputs)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(inputs)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(inputs)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(96, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(x)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding="same", activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(lr=lr)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    else:
        print("Loss function not specified, model not compiled")
    return model


def extended_model_bn_landcover(input_shape, num_classes, lr=0.003, loss=None):
    inputs = Input(input_shape)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(inputs)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(inputs)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(inputs)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(96, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(x)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding="same", activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(lr=lr)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    else:
        print("Loss function not specified, model not compiled")
    return model


def extended2_model_bn_landcover(input_shape, num_classes, lr=0.003, loss=None):
    inputs = Input(input_shape)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(inputs)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(inputs)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(inputs)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(96, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(x)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(96, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(x)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding="same", activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(lr=lr)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    else:
        print("Loss function not specified, model not compiled")
    return model


def unet_landcover(img_shape, out_ch=7, start_ch=64, depth=4, inc_rate=2., activation='relu', 
    dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False, lr=0.003, loss="crossentropy"):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(64, 3, activation="relu", padding="same")(o)
    o = Conv2D(out_ch, 1, activation=None)(o)
    outputs_hr = Activation("softmax", name="outputs_hr")(o)
    outputs_sr = Lambda(lambda x: x, name="outputs_sr")(outputs_hr)

    if loss == "superres":
        model = Model(inputs=i, outputs=[outputs_hr, outputs_sr])
    else:
        model = Model(inputs=i, outputs=outputs_hr)

    optimizer = Adam(lr=lr)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "superres":

        nlcd_class_weights, nlcd_means, nlcd_vars = load_nlcd_stats()
        model.compile(
            optimizer=optimizer,
            loss={
                "outputs_hr": hr_loss(),
                "outputs_sr": sr_loss(nlcd_class_weights, nlcd_means, nlcd_vars)
            },
            loss_weights={
                "outputs_hr": 0.97560975609,
                "outputs_sr": 0.025
            },
            metrics={
                "outputs_hr": accuracy
            }
        )
    else:
        print("Loss function not specified, model not compiled")
    return model