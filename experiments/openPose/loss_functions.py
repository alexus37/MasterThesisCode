import tensorflow as tf

def multi_stage_loss(e_source, target_heat, target_paf, start_stage=2):
    losses_per_stage = []
    # compute loss per stage
    for stage in range(start_stage, 7):
        paf_op = e_source.graph.get_operation_by_name(f'TfPoseEstimator/Mconv7_stage{stage}_L1/BiasAdd').outputs[0]
        heat_op = e_source.graph.get_operation_by_name(f'TfPoseEstimator/Mconv7_stage{stage}_L2/BiasAdd').outputs[0]

        # compute the loss for every channel
        loss_paf = tf.nn.l2_loss(tf.concat(paf_op, axis=-1) - tf.concat(target_paf, axis=-1), name='AX_loss_l1_stage%d_' % (stage))
        loss_heat = tf.nn.l2_loss(tf.concat(heat_op, axis=-1) - tf.concat(target_heat, axis=-1), name='AX_loss_l2_stage%d_' % (stage))
        losses_per_stage.append(tf.reduce_sum([loss_paf, loss_heat]))

    # combined loss form every stage
    total_loss = tf.reduce_sum(losses_per_stage)
    summary = tf.summary.scalar('grad_norm', total_loss)


    grad = tf.gradients(total_loss, e_source.tensor_image)[0]
    return grad, summary

def multi_stage_loss_single_arg(e_source):
    target_heat = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, 19))
    target_paf = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, 38))

    eps = tf.constant(0.05, name='Epsilon')


    losses_per_stage = []
    # compute loss per stage
    for stage in range(2, 7):
        paf_op = e_source.graph.get_operation_by_name(f'TfPoseEstimator/Mconv7_stage{stage}_L1/BiasAdd').outputs[0]
        heat_op = e_source.graph.get_operation_by_name(f'TfPoseEstimator/Mconv7_stage{stage}_L2/BiasAdd').outputs[0]

        # check if concat works
        # compute the loss for every channel
        loss_paf = tf.nn.l2_loss(tf.concat(paf_op, axis=-1) - tf.concat(target_paf, axis=-1), name='AX_loss_l1_stage%d_' % (stage))
        loss_heat = tf.nn.l2_loss(tf.concat(heat_op, axis=-1) - tf.concat(target_heat, axis=-1), name='AX_loss_l2_stage%d_' % (stage))
        losses_per_stage.append(tf.reduce_sum([loss_paf, loss_heat]))

    # combined loss form every stage
    total_loss = tf.reduce_sum(losses_per_stage)


    grad = tf.gradients(total_loss, e_source.tensor_image)[0]
    return grad, total_loss

# only use the final stage and only care about the hgeatmaps
def final_stage_heat_loss(e_source, target_heat):
    heat_op = e_source.graph.get_operation_by_name(f'TfPoseEstimator/Mconv7_stage6_L2/BiasAdd').outputs[0]

    loss_heat = tf.nn.l2_loss(tf.concat(heat_op, axis=-1) - tf.concat(target_heat, axis=-1), name='AX_loss_l2_stage7_')

    # combined loss form every stage
    total_loss = tf.reduce_sum(loss_heat)


    grad = tf.gradients(total_loss, e_source.tensor_image)[0]
    return grad, total_loss

# only use a single heatmap for the loss
def final_stage_single_heat_loss(e_source, target_heat, index):
    heat_op = e_source.graph.get_operation_by_name(f'TfPoseEstimator/Mconv7_stage6_L2/BiasAdd').outputs[0][:, :, :, index]

    loss_heat = tf.nn.l2_loss(tf.concat(heat_op, axis=-1) - tf.concat(target_heat, axis=-1), name='AX_loss_l2_stage7_')

    # combined loss form every stage
    total_loss = tf.reduce_sum(loss_heat)


    grad = tf.gradients(total_loss, e_source.tensor_image)[0]
    return grad, total_loss
# x_adv = e_source.tensor_image - eps * tf.sign(grad)
# x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
# x_adv = tf.stop_gradient(x_adv)
def final_stage_single_heat_upsampled_loss(e_source, target_heat, index):
    # No gradient defined for operation 'upsample_heatmat' (op type: ResizeArea)
    heat_op = e_source.graph.get_operation_by_name(f'upsample_heatmat').outputs[0][:, :, :, index]

    loss_heat = tf.nn.l2_loss(tf.concat(heat_op, axis=-1) - tf.concat(target_heat, axis=-1), name='AX_upsample_heatmat')

    # combined loss form every stage
    total_loss = tf.reduce_sum(loss_heat)


    grad = tf.gradients(total_loss, e_source.tensor_image)[0]
    return grad, total_loss

def multi_stage_loss_batch(estimator, target_heat, target_paf, start_stage=2):
    losses_per_stage = []
    # compute loss per stage
    for stage in range(start_stage, 7):
        paf_op = estimator.graph.get_operation_by_name(f'TfPoseEstimator/Mconv7_stage{stage}_L1/BiasAdd').outputs[0]
        heat_op = estimator.graph.get_operation_by_name(f'TfPoseEstimator/Mconv7_stage{stage}_L2/BiasAdd').outputs[0]

        # compute the loss for every channel
        loss_paf = tf.nn.l2_loss(tf.concat(paf_op, axis=-1) - tf.concat(target_paf, axis=-1), name='AX_loss_l1_stage%d_' % (stage))
        loss_heat = tf.nn.l2_loss(tf.concat(heat_op, axis=-1) - tf.concat(target_heat, axis=-1), name='AX_loss_l2_stage%d_' % (stage))
        losses_per_stage.append(tf.reduce_sum([loss_paf, loss_heat]))

    # combined loss form every stage
    total_loss = tf.reduce_sum(losses_per_stage)
    summary = tf.summary.scalar('batch_loss', total_loss)

    grad = tf.gradients(total_loss, estimator.tensor_image)[0]
    # compute the average
    grad = tf.reduce_mean(grad, 0)
    return grad, summary
