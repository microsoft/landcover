import cntk
from cntk.layers import *
from cntk.initializer import *
from cntk.ops import *
import cntk.train.distributed as distributed


import numpy as np

def ddist(prediction, c_interval_center, c_interval_radius):
    ''' Distance of the predictions from the edges of the intervals '''
    return cntk.relu(cntk.abs(prediction - c_interval_center) - c_interval_radius)
    #return cntk.abs(prediction - c_interval_center)


def make_trainer(epoch_size, mb_size_in_samples, output, high_res_loss, loss, max_epochs,
    my_rank, number_of_workers, lr_adjustment_factor, log_dir):
    ''' Define the learning rate schedule, trainer, and evaluator '''
    lr_per_mb = [0.001] * 30 + [0.0001] * 30 + [0.00001] * 30 + [0.000001] * 1000
    lr_per_mb = [lr * lr_adjustment_factor for lr in lr_per_mb]

    lr_schedule = cntk.learning_parameter_schedule(lr_per_mb, epoch_size=epoch_size*mb_size_in_samples)

    learner = cntk.rmsprop(
        parameters=output.parameters,
        lr=lr_schedule,
        gamma=0.95,
        inc=1.1,
        dec=0.9,
        max=1.1,
        min=0.9
    )

    '''
    learner = cntk.learners.adam(
        parameters=output.parameters,
        lr=lr_schedule
    )
    '''

    progress_printer = cntk.logging.ProgressPrinter(
        tag='Training',
        num_epochs=max_epochs,
        freq=epoch_size,
        rank=my_rank
    )

    tensorboard = cntk.logging.TensorBoardProgressWriter(freq=1, log_dir=log_dir, rank=None, model=output)
    trainer = cntk.Trainer(output, (loss, high_res_loss), learner, [progress_printer, tensorboard])
    #evaluator = cntk.Evaluator(loss)

    return (trainer, tensorboard)


def get_model(f_dim, c_dim, l_dim, m_dim, num_stack_layers,
        super_res_class_weight, super_res_loss_weight, high_res_loss_weight):
    # Define the variables into which the minibatch data will be loaded.
    num_nlcd_classes, num_landcover_classes = c_dim
    _, block_size, _ = f_dim
    input_im = cntk.input_variable(f_dim, np.float32)
    lc = cntk.input_variable(l_dim, np.float32)
    lc_weight_map = cntk.input_variable((1, l_dim[1], l_dim[2]), np.float32)
    interval_center = cntk.input_variable(c_dim, np.float32)
    interval_radius = cntk.input_variable(c_dim, np.float32)
    mask = cntk.input_variable(m_dim, np.float32)

    # Create the model definition. c_map defines the number of filters trained
    # at layers of different depths in the model. num_stack_layers defines the
    # number of (modified) residual units per layer.
    # model = dense_fc_model(
    #     input_tensor=input_im,
    #     num_stack_layers=num_stack_layers,
    #     c_map=[32, 32, 16, 16, 16],
    #     num_classes=num_landcover_classes,
    #     bs=block_size
    # )

    model = cnn_model(
        input_tensor=input_im,
        num_stack_layers=num_stack_layers,
        c_map=[64, 32, 32, 32, 32],
        num_classes=num_landcover_classes,
        bs=block_size
    )


    # At this stage the model produces output for the whole region in the input
    # image, but we will focus only on the center of that region during
    # training. Here we drop the predictions at the edges.
    output = cntk.reshape(model, (num_landcover_classes, block_size, block_size))
    probs = cntk.reshape(cntk.softmax(output, axis=0),
                         (num_landcover_classes, block_size, block_size))

    # Now we calculate the supre-res loss. Note that this loss function has the
    # potential to become negative since the variance is fractional.
    # Additionally, we need to make sure that when the nlcd mask[0, ...]
    # is always 1, which means that there's no nlcd label everywhere,
    # the supre_res_loss comes out as a constant.
    super_res_crit = 0
    mask_size = cntk.reshape(
                    cntk.reduce_sum(cntk.slice(mask, 0, 1, num_nlcd_classes)), (1,)
                ) + 10.0
    # Not considering nlcd class 0
    for nlcd_id in range(1, num_nlcd_classes):
        c_mask = cntk.reshape(cntk.slice(mask, 0, nlcd_id, nlcd_id+1),
                              (1, block_size, block_size))
        c_mask_size = cntk.reshape(cntk.reduce_sum(c_mask), (1,)) + 0.000001
        c_interval_center = cntk.reshape(cntk.slice(interval_center, 0, nlcd_id, nlcd_id+1),
                                         (num_landcover_classes, ))
        c_interval_radius = cntk.reshape(cntk.slice(interval_radius, 0, nlcd_id, nlcd_id+1),
                                         (num_landcover_classes, ))

        # For each nlcd class, we have a landcover distribution:
        masked_probs = probs * c_mask
        # Mean mean of predicted distribution
        mean = cntk.reshape(cntk.reduce_sum(masked_probs, axis=(1, 2)),
                            (num_landcover_classes, )) / c_mask_size
        # Mean var of predicted distribution
        var  = cntk.reshape(cntk.reduce_sum(masked_probs * (1.-masked_probs), axis=(1, 2)),
                            (num_landcover_classes, )) / c_mask_size
        c_super_res_crit = cntk.square(ddist(mean, c_interval_center, c_interval_radius)) / (
                            var / c_mask_size + c_interval_radius * c_interval_radius + 0.000001) \
                        + cntk.log(var + 0.03)
        super_res_crit += c_super_res_crit * c_mask_size / mask_size * super_res_class_weight[nlcd_id]

    # Weight super_res loss according to the ratio of unlabeled LC pixels
    super_res_loss = cntk.reduce_sum(super_res_crit) * cntk.reduce_mean(cntk.slice(lc, 0, 0, 1))

    log_probs = cntk.log(probs)
    high_res_crit = cntk.times([0.0, 1.0, 1.0, 1.0, 1.0],
            cntk.element_times(-cntk.element_times(log_probs, lc), lc_weight_map),
                                output_rank=2)
    # Average across spatial dimensions
    # Sum over all landcover classes, only one of the landcover classes is non-zero


    #high_res_loss = cntk.reduce_mean(high_res_crit)

    print("probs", probs)
    print("lc", lc)
    print("lc_weight_map", lc_weight_map)
    print("cntk.element_times(probs, lc)", cntk.element_times(probs, lc))

    iou_loss_i = cntk.element_times(
        [0.0, 1.0, 1.0, 1.0, 1.0],
        cntk.reduce_sum(cntk.element_times(cntk.element_times(probs, lc), lc_weight_map), axis=(1,2))
    )
    print("iou_loss_i",iou_loss_i)
    iou_loss_u = cntk.element_times(
        [0.0, 1.0, 1.0, 1.0, 1.0],
        cntk.reduce_sum(cntk.minus(
            cntk.plus(probs, lc), cntk.element_times(probs, lc)
        ), axis=(1,2))
    )
    print("iou_loss_u",iou_loss_u)

    high_res_loss = 1.0 - ((1/4.0) * cntk.reduce_mean(cntk.element_divide(iou_loss_i, iou_loss_u)))

    print("high_res_loss", high_res_loss)

    loss = super_res_loss_weight * super_res_loss + high_res_loss_weight * high_res_loss

    return input_im, lc, lc_weight_map, mask, interval_center, interval_radius, \
           output, high_res_loss_weight * high_res_loss, loss





#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Model structure code
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=True)(input)
    b = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=True)(c)
    r = relu(b)
    return r

def pool_block(input, fsize=(2,2), strides=(2,2)):
    p = MaxPooling(fsize, strides=strides, pad=False)(input)
    return p

def conv_block(input, num_stack_layers, num_filters):
    assert (num_stack_layers >= 0)
    l = input
    for i in range(num_stack_layers):
        l = conv_bn_relu(l, (3,3), num_filters)
    return l

def unpl_block(input, output_shape, num_filters, strides, pname):
    #x_dup0 = cntk.reshape(input, (input.shape[0], input.shape[1], 1, input.shape[2], 1))
    #x_dup1 = cntk.splice(x_dup0, x_dup0, axis=-1)
    #x_dup2 = cntk.splice(x_dup1, x_dup1, axis=-3)
    #upsampled = cntk.reshape(x_dup2, (input.shape[0], input.shape[1]*2, input.shape[2]*2))
    #return upsampled
    c = Convolution(
            (1,1), num_filters, activation=cntk.relu,
            init=he_normal(), strides=(1,1), name=pname)(input)
    ct = ConvolutionTranspose(
            (3,3), num_filters, strides=strides,
            output_shape=output_shape, pad=True,
            bias=True, init=bilinear(3,3))(c)
    ctf = cntk.ops.combine([ct]).clone(
            cntk.ops.functions.CloneMethod.freeze,
            {c: cntk.ops.placeholder(name=pname)})(c)
    l = conv_bn_relu(ctf, (3,3), num_filters)
    return l

def merg_block(in1, in2, num_filters):
    #c = Convolution((1, 1), num_filters, activation=None, bias=True)(in1) + \
    #    Convolution((1, 1), num_filters, activation=None, bias=True)(in2)
    #b = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=True)(c)
    #r = relu(b)
    #return r
    s = splice(in1, in2, axis=0)
    return s

def shallow_cnn_model(input_tensor, num_stack_layers, c_map, num_classes, bs):
    r1_1 = input_tensor

    r1_2 = conv_block(r1_1, num_stack_layers=8, num_filters=64)
    o1_1 = Convolution((3, 3), num_classes, activation=None, pad=True, bias=True)(r1_2)
    return o1_1

def cnn_model(input_tensor, num_stack_layers, c_map, num_classes, bs):
    r1_1 = input_tensor

    r1_2 = conv_block(r1_1, num_stack_layers, c_map[0])

    r2_1 = pool_block(r1_2)
    r2_2 = conv_block(r2_1, num_stack_layers, c_map[1])

    r3_1 = pool_block(r2_2)
    r3_2 = conv_block(r3_1, num_stack_layers, c_map[2])

    r4_1 = pool_block(r3_2)
    r4_2 = conv_block(r4_1, num_stack_layers, c_map[3])

    r5_1 = pool_block(r4_2)
    r5o5 = conv_block(r5_1, num_stack_layers, c_map[4])
    o5_1 = unpl_block(r5o5, (bs/8, bs/8),     c_map[4], 2, 'o5_1')

    o4_3 = merg_block(o5_1, r4_2,             c_map[3])
    o4_2 = conv_block(o4_3, num_stack_layers, c_map[3])
    o4_1 = unpl_block(o4_2, (bs/4, bs/4),     c_map[3], 2, 'o4_1')

    o3_3 = merg_block(o4_1, r3_2,             c_map[2])
    o3_2 = conv_block(o3_3, num_stack_layers, c_map[2])
    o3_1 = unpl_block(o3_2, (bs/2, bs/2),     c_map[2], 2, 'o3_1')

    o2_3 = merg_block(o3_1, r2_2,             c_map[1])
    o2_2 = conv_block(o2_3, num_stack_layers, c_map[1])
    o2_1 = unpl_block(o2_2, (bs/1, bs/1),     c_map[1], 2, 'o2_1')

    o1_3 = merg_block(o2_1, r1_2,             c_map[0])
    o1_2 = conv_block(o1_3, num_stack_layers, c_map[0])
    o1_1 = Convolution((3, 3), num_classes, activation=None, pad=True, bias=True)(o1_2)

    return o1_1


def dense_merge(in1, in2):
    s = splice(in1, in2, axis=0)
    return s

def dense_up(input, output_shape, num_filters, strides, pname):
    c = Convolution((1,1), num_filters, activation=cntk.relu, init=he_normal(), strides=(1,1), name=pname)(input)
    ct = ConvolutionTranspose((3,3), num_filters, strides=strides,output_shape=output_shape, pad=True, bias=True, init=bilinear(3,3))(c)
    ctf = cntk.ops.combine([ct]).clone(cntk.ops.functions.CloneMethod.freeze, {c: cntk.ops.placeholder(name=pname)})(c)
    #l = conv_bn_relu(ctf, (3,3), num_filters)
    #return l
    return ctf

def dense_down(input, fsize=(2,2), strides=(2,2)):
    p = MaxPooling(fsize, strides=strides, pad=False)(input)
    return p

def dense_block(input, num_stack_layers, num_filters):
    l = input
    layer_outputs = []
    for i in range(num_stack_layers):
        conv = conv_bn_relu(l, (3,3), num_filters)
        l = splice(conv, l, axis=0)
        layer_outputs.append(conv)
    return splice(*layer_outputs, axis=0)

def dense_fc_model(input_tensor, num_stack_layers, c_map, num_classes, bs, verbose=True):
    
    r1_1 = input_tensor

    input_conv = Convolution((3, 3), c_map[0], activation=None, pad=True, bias=True)(r1_1)

    left_1 = dense_block(input_conv, num_stack_layers, c_map[1])
    left_1 = dense_merge(left_1, input_conv)
    left_1_down = dense_down(left_1)
    
    left_2 = dense_block(left_1_down, num_stack_layers, c_map[2])
    left_2 = dense_merge(left_2, left_1_down)
    left_2_down = dense_down(left_2)

    left_3 = dense_block(left_2_down, num_stack_layers, c_map[3])
    left_3 = dense_merge(left_3, left_2_down)
    left_3_down = dense_down(left_3)
    
    # ---------------------------------------------------
    center_1 = dense_block(left_3_down, num_stack_layers, c_map[4])
    # ---------------------------------------------------
    
    right_3_up = dense_up(center_1, (bs/4, bs/4), c_map[3], 2, 'right_3_up')
    right_3_up = dense_merge(right_3_up, left_3)
    right_3 = dense_block(right_3_up, num_stack_layers, c_map[2])

    right_2_up = dense_up(right_3, (bs/2, bs/2), c_map[2], 2, 'right_2_up')
    right_2_up = dense_merge(right_2_up, left_2)
    right_2 = dense_block(right_2_up, num_stack_layers, c_map[1])
    
    right_1_up = dense_up(right_2, (bs/1, bs/1), c_map[1], 2, 'right_1_up')
    right_1_up = dense_merge(right_1_up, left_1)
    right_1 = dense_block(right_1_up, num_stack_layers, c_map[0])
    
    output_conv = Convolution((1, 1), num_classes, activation=None, pad=True, bias=True)(right_1)

    if verbose:
        print("Output", output_conv)
    
    return output_conv

