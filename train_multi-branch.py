from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob
import config as cfg
import time
from utils.utils import get_variables_in_checkpoint_file, get_variables_to_restore, get_variables_to_restore_rpcnet
from networks.RPCNet import inference_prior_occ
from utils.loss_function import cross_entropy_loss, weighted_cross_entropy_loss_artdeco
from image_reader5 import ImageReader5
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU


def main(argv=None):
    input_size = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    print('Start Train: ' + cfg.TRAIN_DATA_LIST)

    is_finetune = False

    with tf.name_scope("create_inputs"):
        reader = ImageReader5(
            cfg.TRAIN_DATA_DIR,
            cfg.TRAIN_DATA_LIST,
            input_size,
            cfg.RANDOM_SCALE,
            cfg.RANDOM_MIRROR,
            cfg.RANDOM_CROP_PAD,
            cfg.IGNORE_LABEL,
            cfg.IMG_MEAN,
            coord)
        image_batch, label_batch, label_occ, label_win, label_bal = reader.dequeue(cfg.BATCH_SIZE)
        label_occ = tf.cast(tf.scalar_mul(1.0 / 255, label_occ), tf.float32)

    # window balcony prior
    label_win = tf.cast(tf.scalar_mul(1.0 / 255, label_win), tf.float32)
    label_bal = tf.cast(tf.scalar_mul(1.0 / 255, label_bal), tf.float32)
    label_win_bal = label_win + label_bal

    label_bg_one = tf.ones(label_win.get_shape().as_list(), tf.float32)
    label_bg_zero = tf.zeros(label_win.get_shape().as_list(), tf.float32)
    label_bg = tf.where(tf.equal(label_win_bal, 0), label_bg_one, label_bg_zero)  # 0： ori, 1： modified

    image_input = tf.concat([label_bg, label_win, label_bal], 3)

    label_win_bal = tf.expand_dims(tf.argmax(image_input, 3), 3)

    pred_annotation1, logits, logits_occ, window_balcony, logits_win_bal = inference_prior_occ(image_batch, is_training=True)

    # Loss
    if 'artdeco' in cfg.TRAIN_DATA_DIR:
        logits_loss = weighted_cross_entropy_loss_artdeco(logits, label_batch)
    else:
        logits_loss = cross_entropy_loss(logits, label_batch)
    logits_loss_occ = cross_entropy_loss(logits_occ, tf.cast(label_occ, tf.int32))

    if is_finetune:
        binary_window_loss = cross_entropy_loss(logits_win_bal, label_win_bal)
    else:
        binary_window_loss = tf.constant(0, tf.float32)

    logits_occ = tf.expand_dims(tf.argmax(tf.nn.softmax(logits_occ), 3), 3)
    window_show = tf.expand_dims(tf.argmax(logits_win_bal, 3), 3)

    # total loss
    if is_finetune:
        ce_loss = logits_loss + logits_loss_occ + binary_window_loss
    else:
        ce_loss = logits_loss + logits_loss_occ

    l2_loss = [cfg.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()
                 if 'weights' or 'w' in v.name or 'W' in v.name]
    l2_losses = tf.add_n(l2_loss)

    # Total loss
    loss = ce_loss + l2_losses

    # Summary
    tf.summary.scalar("ce_loss", ce_loss)
    tf.summary.scalar("l2_losses", l2_losses)
    tf.summary.scalar("total_loss", loss)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())

    # Using Poly learning rate policy
    base_lr = tf.constant(cfg.LEARNING_RATE)
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / cfg.NUM_STEPS), cfg.POWER))

    # Optimizer
    opt = tf.train.AdamOptimizer(learning_rate)

    ## Retrieve all trainable variables you defined in your graph
    if cfg.FREEZE_BN:
        tvs = [v for v in tf.trainable_variables()
               if 'beta' not in v.name and 'gamma' not in v.name]
    else:
        tvs = [v for v in tf.trainable_variables()]

    ## Creation of a list of variables with the same shape as the trainable ones
    # initialized with 0s

    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

    ## Calls the compute_gradients function of the optimizer to obtain... the list of gradients
    gvs = opt.compute_gradients(loss, tvs)

    ## Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

    ## Define the training step (part with variable value update)
    train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    # Set gpu usage
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # Build session
    sess = tf.Session(config=config)
    print("Setting up Saver...")
    # Max number of model
    saver = tf.train.Saver(max_to_keep=cfg.MAX_SNAPSHOT_NUM)
    train_writer = tf.summary.FileWriter(cfg.LOG_DIR + 'train', sess.graph)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Training examples
    _img = image_batch[0]
    _lab = label_batch[0]
    _mask = pred_annotation1[0]
    _occ = logits_occ[0]
    _bg = tf.expand_dims(tf.argmax(window_balcony, axis=3), dim=3)[0] # tf.expand_dims(tf.argmax(logits_bg, axis=3, name="prediction_bg"), dim=3)[0]
    _att = window_show[0]

    # Create save path
    if not os.path.exists(cfg.LOG_DIR):
        os.makedirs(cfg.LOG_DIR)
    if not os.path.exists(cfg.SAVE_DIR):
        os.makedirs(cfg.SAVE_DIR)
    if not os.path.exists(cfg.SAVE_DIR + 'temp_img'):
        os.mkdir(cfg.SAVE_DIR + 'temp_img')


    count = 0
    files = os.path.join(cfg.SAVE_DIR + 'model.ckpt-*.index')
    sfile = glob.glob(files)
    if len(sfile) > 0:
        sess.run(tf.global_variables_initializer())
        sfile = glob.glob(files)
        steps = []
        for s in sfile:
            part = s.split('.')
            step = int(part[1].split('-')[1])
            steps.append(step)
        count = max(steps)
        model = cfg.SAVE_DIR + 'model.ckpt-' + str(count)
        print('\nRestoring weights from: ' + model)
        saver.restore(sess, model)
        print('End Restore')
    else:
        # Restore from pre-train on imagenet
        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))

        if os.path.exists(cfg.PRE_TRAINED_MODEL) or os.path.exists(cfg.PRE_TRAINED_MODEL + '.index'):
            var_keep_dic = get_variables_in_checkpoint_file(cfg.PRE_TRAINED_MODEL)
            # Get the variables to restore, ignoring the variables to fix
            variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
            if len(variables_to_restore) > 0:
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, cfg.PRE_TRAINED_MODEL)
                print('-- Model pre-train loaded from ' + cfg.PRE_TRAINED_MODEL)
            else:
                print('-- Model inited random.')
        else:
            print('-- Model inited random.')

        set_name = cfg.TRAIN_DATA_DIR.split('/')[-2]
        if 'artdeco' in cfg.TRAIN_DATA_DIR:
            RPCNet_PRE_TRAINED_MODEL = 'saves/artdeco3_repetitive_pattern/' + set_name + '/model.ckpt-10000'
        else:
            RPCNet_PRE_TRAINED_MODEL = 'saves/ecp-occluded-25/' + set_name + '/repetitive_pattern/model.ckpt-10000'
            # RPCNet_PRE_TRAINED_MODEL = 'saves/repetitive_pattern_ecp-occluded/' + set_name + '/model.ckpt-10000'
        if os.path.exists(RPCNet_PRE_TRAINED_MODEL + '.index'):
            var_keep_dic = get_variables_in_checkpoint_file(RPCNet_PRE_TRAINED_MODEL)
            # Get the variables to restore, ignoring the variables to fix
            variables_to_restore = get_variables_to_restore_rpcnet(variables, var_keep_dic)
            if len(variables_to_restore) > 0:
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, RPCNet_PRE_TRAINED_MODEL)
                print('-- RPCNet Model pre-train loaded from ' + RPCNet_PRE_TRAINED_MODEL)
            else:
                print('-- RPCNet Model inited random.')
        else:
            print('-- RPCNet Model inited random.')

        # # Convert RGB -> BGR of resnet_v1_50
        # conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        # restorer_fc = tf.train.Saver({'resnet_v1_50/conv1/weights': conv1_rgb})
        # restorer_fc.restore(sess, cfg.PRE_TRAINED_MODEL)
        # sess.run(tf.assign(variables[0], tf.reverse(conv1_rgb, [2])))
        # print('-- ResNet Conv 1 RGB->BGR')

    print('Start Train: ' + cfg.TRAIN_DATA_LIST)
    print('---------------Hyper Paras---------------')
    print('-- batch_size: ', cfg.BATCH_SIZE)
    print('-- gradient accumulation: ', cfg.GRADIENT_ACCUMULATION)
    print('-- image height: ', cfg.IMAGE_HEIGHT)
    print('-- image width: ', cfg.IMAGE_WIDTH)
    print('-- learning rate: ', cfg.LEARNING_RATE)
    print('-- GPU: ', cfg.GPU)
    print('-- class num: ', cfg.DATASET_NUM_CLASSESS)
    print('-- total iter: ', cfg.NUM_STEPS)
    print('-- start save step: ', cfg.START_SAVE_STEP)
    print('-- save step every: ', cfg.SAVE_STEP_EVERY)
    print('-- model save num: ', cfg.MAX_SNAPSHOT_NUM)
    print('-- summary interval: ', cfg.SUMMARY_INTERVAL)
    print('-- weight decay: ', cfg.WEIGHT_DECAY)
    print('-- freeze BN: ', cfg.FREEZE_BN)
    print('-- decay rate: ', cfg.POWER)
    print('-- minScale: ', cfg.MIN_SCALE)
    print('-- maxScale: ', cfg.MAX_SCALE)
    print('-- random scale: ', cfg.RANDOM_SCALE)
    print('-- random mirror: ', cfg.RANDOM_MIRROR)
    print('-- random crop: ', cfg.RANDOM_CROP_PAD)
    print('-- pre-trained: ' + cfg.PRE_TRAINED_MODEL)
    print('----------------End---------------------')
    fcfg = open(cfg.SAVE_DIR + 'cfg.txt', 'w')
    fcfg.write('-- batch_size: ' + str(cfg.BATCH_SIZE) + '\n')
    fcfg.write('-- gradient accumulation: ' + str(cfg.GRADIENT_ACCUMULATION) + '\n')
    fcfg.write('-- image height: ' + str(cfg.IMAGE_HEIGHT) + '\n')
    fcfg.write('-- image width: ' + str(cfg.IMAGE_WIDTH) + '\n')
    fcfg.write('-- learning rate: ' + str(cfg.LEARNING_RATE) + '\n')
    fcfg.write('-- GPU: ' + str(cfg.GPU) + '\n')
    fcfg.write('-- class num: ' + str(cfg.DATASET_NUM_CLASSESS) + '\n')
    fcfg.write('-- total iter: ' + str(cfg.NUM_STEPS) + '\n')
    fcfg.write('-- start save step: ' + str(cfg.START_SAVE_STEP) + '\n')
    fcfg.write('-- save step every: ' + str(cfg.SAVE_STEP_EVERY) + '\n')
    fcfg.write('-- model save num: ' + str(cfg.MAX_SNAPSHOT_NUM) + '\n')
    fcfg.write('-- summary interval: ' + str(cfg.SUMMARY_INTERVAL) + '\n')
    fcfg.write('-- weight decay: ' + str(cfg.WEIGHT_DECAY) + '\n')
    fcfg.write('-- freeze BN: ' + str(cfg.FREEZE_BN) + '\n')
    fcfg.write('-- decay rate: ' + str(cfg.POWER) + '\n')
    fcfg.write('-- minScale: ' + str(cfg.MIN_SCALE) + '\n')
    fcfg.write('-- maxScale: ' + str(cfg.MAX_SCALE) + '\n')
    fcfg.write('-- random scale: ' + str(cfg.RANDOM_SCALE) + '\n')
    fcfg.write('-- random mirror: ' + str(cfg.RANDOM_MIRROR) + '\n')
    fcfg.write('-- random crop: ' + str(cfg.RANDOM_CROP_PAD) + '\n')
    fcfg.write('-- pre-trained: ' + str(cfg.PRE_TRAINED_MODEL) + '\n')
    fcfg.close()

    last_summary_time = time.time()

    # iteration number of each epoch
    record = cfg.DATASET_SIZE / cfg.BATCH_SIZE

    running_count = count
    epo = int(count / record)

    train_start_time = time.time()

    # Change the graph for read only
    sess.graph.finalize()

    # Start training
    while running_count < cfg.NUM_STEPS:
        time_start = time.time()
        itr = 0
        while itr < int(record):
            itr += 1
            running_count += 1

            # More than total iter, stopping training
            if running_count > cfg.NUM_STEPS:
                break

            feed_dict = {step_ph: (running_count)}

            # Save summary and example images
            now = time.time()
            if now - last_summary_time > cfg.SUMMARY_INTERVAL:
                summary_str = sess.run(summary_op, feed_dict={step_ph: running_count})
                train_writer.add_summary(summary_str, running_count)
                last_summary_time = now

                # Save tmp results
                s_img, s_lab, s_mask, s_occ, s_bg, s_att = sess.run([_img, _lab, _mask, _occ, _bg, _att],
                                                                    feed_dict=feed_dict)
                s_img = np.array(s_img + cfg.IMG_MEAN, np.uint8)
                s_lab = s_lab * 20
                s_lab = np.concatenate([s_lab, s_lab, s_lab], axis=2)
                s_mask = s_mask * 20
                s_mask = np.concatenate([s_mask, s_mask, s_mask], axis=2)
                s_occ = s_occ * 255
                s_occ = np.concatenate([s_occ, s_occ, s_occ], axis=2)
                s_bg = s_bg * 100
                s_bg = np.concatenate([s_bg, s_bg, s_bg], axis=2)
                print('s_att max: ' + str(np.max(s_att)) + ', min: ' + str(np.min(s_att)))
                s_att = s_att * 100
                s_att = np.array(s_att, np.int32)
                s_att = np.concatenate([s_att, s_att, s_att], axis=2)

                save_temp = np.zeros((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 6, 3), np.uint8)
                save_temp[0:cfg.IMAGE_HEIGHT, 0:cfg.IMAGE_WIDTH, :] = s_img
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH:cfg.IMAGE_WIDTH * 2, :] = s_lab
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 2:cfg.IMAGE_WIDTH * 3, :] = s_mask
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 3:cfg.IMAGE_WIDTH * 4, :] = s_occ
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 4:cfg.IMAGE_WIDTH * 5, :] = s_bg
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 5:cfg.IMAGE_WIDTH * 6, :] = s_att
                cv2.imwrite(cfg.SAVE_DIR + 'temp_img/' + str(now) + '_mask.jpg', save_temp)

            time_s = time.time()

            # Run the zero_ops to initialize it
            sess.run(zero_ops)

            # Accumulate the gradients 'n_minibatches' times in accum_vars using accum_ops
            for i in range(cfg.GRADIENT_ACCUMULATION):
                sess.run(accum_ops, feed_dict=feed_dict)

            train_loss, ls_ce, ls_occ, ls_win, ls_l2, lr = sess.run(
                [loss, logits_loss, logits_loss_occ, binary_window_loss,
                 l2_losses, learning_rate],
                feed_dict=feed_dict)


            # Run the train_step ops to update the weights based on your accumulated gradients
            sess.run(train_step, feed_dict=feed_dict)

            time_e = time.time()

            print("Epo: %d, Step: %d, Train_loss:%g, fg: %g, occ: %g, win: %g,"
                  " l2:%g,  lr:%g, time:%g" %
                  (epo, running_count, train_loss, ls_ce, ls_occ, ls_win,
                   ls_l2, lr, time_e - time_s))

            # Save step model
            if (running_count % cfg.SAVE_STEP_EVERY) == 0 \
                    and running_count >= cfg.START_SAVE_STEP:
                saver.save(sess, cfg.SAVE_DIR + 'model.ckpt', int(running_count))
                print('Model has been saved:' + str(running_count))
                files = os.path.join(cfg.SAVE_DIR + 'model.ckpt-*.data-00000-of-00001')
                sfile = glob.glob(files)
                if len(sfile) > cfg.MAX_SNAPSHOT_NUM:
                    steps = []
                    for s in sfile:
                        part = s.split('.')
                        re = int(part[1].split('-')[1])
                        steps.append(re)
                    re = min(steps)
                    model = cfg.SAVE_DIR + 'model.ckpt-' + str(re)
                    os.remove(model + '.data-00000-of-00001')
                    os.remove(model + '.index')
                    os.remove(model + '.meta')
                    print('Remove Model:' + model)

        epo += 1
        time_end = time.time()
        print('Epo ' + str(epo) + ' use time: ' + str(time_end - time_start))

    # Finish training
    train_end_time = time.time()
    print('Train total use: ' + str((train_end_time-train_start_time) / 3600) + ' h')
    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
