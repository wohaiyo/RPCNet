from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob
import config as cfg
import time
from utils.utils import get_variables_in_checkpoint_file, get_variables_to_restore
from networks.RPCNet import inference_rpcnet
from utils.loss_function import cross_entropy_loss
from image_reader5 import ImageReader5
from image_reader3 import ImageReader3
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU


def main(argv=None):

    input_size = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    print('Start Train: ' + cfg.TRAIN_DATA_LIST)

    # Load reader.

    # use the real facade data or synthetic data
    use_real_data = True

    if use_real_data:
        # train for real data
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
        label_win = tf.cast(tf.scalar_mul(1.0 / 255, label_win), tf.float32)
        label_bal = tf.cast(tf.scalar_mul(1.0 / 255, label_bal), tf.float32)
        label_win_bal = label_win + label_bal

        # To prevent overlap of window and balcony
        label_bg_one = tf.ones(label_win.get_shape().as_list(), tf.float32)
        label_bg_zero = tf.zeros(label_win.get_shape().as_list(), tf.float32)
        label_bg = tf.where(tf.equal(label_win_bal, 0), label_bg_one, label_bg_zero)  # 0： ori, 1： modified

        image_input = tf.concat([label_bg, label_win, label_bal], 3)
        label_input = tf.expand_dims(tf.argmax(image_input, 3), 3)
        image_input = image_input * (1 - label_occ)

    else:
        # Train for synthetic data
        with tf.name_scope("create_inputs"):
            reader = ImageReader3(
                cfg.TRAIN_DATA_DIR,
                cfg.TRAIN_DATA_LIST,
                input_size,
                cfg.RANDOM_SCALE,
                cfg.RANDOM_MIRROR,
                cfg.RANDOM_CROP_PAD,
                cfg.IGNORE_LABEL,
                cfg.IMG_MEAN,
                coord)
            image_batch, label_batch, label_occ = reader.dequeue(cfg.BATCH_SIZE)
            label_occ = tf.cast(tf.scalar_mul(1.0/255, tf.cast(label_occ, tf.float32)), tf.float32)

        # To one-hot
        ano_squ = tf.squeeze(image_batch, squeeze_dims=[3])
        image_input = tf.cast(tf.one_hot(indices=ano_squ, depth=3, on_value=1, off_value=0),
                             dtype=tf.float32)
        label_input = label_batch

    # Define network
    pred_annotation, logits_items = inference_rpcnet(image_input, label_occ, is_training=True)

    logits_loss0 = cross_entropy_loss(logits_items[0], label_input)
    logits_loss1 = cross_entropy_loss(logits_items[1], label_input)
    logits_loss2 = cross_entropy_loss(logits_items[2], label_input)
    logits_loss3 = cross_entropy_loss(logits_items[3], label_input)
    logits_loss4 = cross_entropy_loss(logits_items[4], label_input)

    pred_1 = tf.expand_dims(tf.argmax(tf.nn.softmax(logits_items[0]), axis=3), dim=3)
    pred_2 = tf.expand_dims(tf.argmax(tf.nn.softmax(logits_items[2]), axis=3), dim=3)
    pred_3 = tf.expand_dims(tf.argmax(tf.nn.softmax(logits_items[4]), axis=3), dim=3)


    ce_loss = logits_loss0 + logits_loss1 + logits_loss2 + logits_loss3 + logits_loss4


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
    _mask1 = pred_1[0]
    _mask2 = pred_2[0]
    _mask3 = pred_3[0]
    _gt = label_input[0]
    _img = tf.expand_dims(tf.argmax(tf.nn.softmax(image_input), axis=3), dim=3)[0]

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
                print('Model pre-train loaded from ' + cfg.PRE_TRAINED_MODEL)
            else:
                print('Model inited random.')
        else:
            print('Model inited random.')

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
                score_map, score_map2, score_map3, gt_map, img_map = sess.run([_mask1, _mask2, _mask3, _gt, _img],
                                                                              feed_dict=feed_dict)

                is_gray = False  # save temp image for gray image, or RGB
                if is_gray:
                    ignore_value = 100.
                    save_mask = np.array(score_map * ignore_value, np.uint8)
                    save_mask2 = np.array(score_map2 * ignore_value, np.uint8)
                    save_mask3 = np.array(score_map3 * ignore_value, np.uint8)
                    save_gt = np.array(gt_map * ignore_value, np.uint8)
                    save_img = np.array(img_map * ignore_value, np.uint8)
                    # save images
                    save_h = save_mask.shape[0]
                    save_w = int(save_mask.shape[1] * 5)
                    save_image = np.zeros((save_h, save_w, 1), np.uint8)
                    save_image[:, 0: save_img.shape[1], :] = save_img
                    save_image[:, save_img.shape[1]: (int(save_img.shape[1] * 2)), :] = save_gt
                    save_image[:, (int(save_img.shape[1] * 2)): (int(save_img.shape[1] * 3)), :] = save_mask
                    save_image[:, (int(save_img.shape[1] * 3)): (int(save_img.shape[1] * 4)), :] = save_mask2
                    save_image[:, (int(save_img.shape[1] * 4)): (int(save_img.shape[1] * 5)), :] = save_mask3
                    cv2.imwrite(cfg.SAVE_DIR + 'temp_img/' + str(now) + '_mask.jpg', save_image)

                else:
                    # range[0, 1, 2]. 1: red, 2: blue
                    def trans_color(mask):  # 1: red(bgr: 0, 0, 255), 2: blue(bgr: 255, 0, 128)
                        # mask :[h, w]
                        mask1 = np.zeros(mask.shape)
                        mask2 = np.zeros(mask.shape)
                        mask3 = np.zeros(mask.shape)

                        mask1[mask == 1] = 0
                        mask2[mask == 1] = 0
                        mask3[mask == 1] = 255

                        mask1[mask == 2] = 255
                        mask2[mask == 2] = 0
                        mask3[mask == 2] = 128

                        mask_new = np.concatenate([mask1, mask2, mask3], 2)

                        return np.array(mask_new, np.uint8)

                    save_mask = trans_color(score_map)
                    save_mask2 = trans_color(score_map2)
                    save_mask3 = trans_color(score_map3)
                    save_gt = trans_color(gt_map)
                    save_img = trans_color(img_map)

                    # cv2.imwrite(cfg.save_dir + 'temp_img/' + str(now) + '_save_gt.jpg', save_gt)

                    # save color images
                    save_h = save_mask.shape[0]
                    save_w = int(save_mask.shape[1] * 5)
                    save_image = np.zeros((save_h, save_w, 3), np.int32)
                    save_image[:, 0: save_img.shape[1], :] = save_img
                    save_image[:, save_img.shape[1]: (int(save_img.shape[1] * 2)), :] = save_gt
                    save_image[:, (int(save_img.shape[1] * 2)): (int(save_img.shape[1] * 3)), :] = save_mask
                    save_image[:, (int(save_img.shape[1] * 3)): (int(save_img.shape[1] * 4)), :] = save_mask2
                    save_image[:, (int(save_img.shape[1] * 4)): (int(save_img.shape[1] * 5)), :] = save_mask3
                    cv2.imwrite(cfg.SAVE_DIR + 'temp_img/' + str(now) + '_mask.jpg', save_image)

            time_s = time.time()

            # Run the zero_ops to initialize it
            sess.run(zero_ops)

            # Accumulate the gradients 'n_minibatches' times in accum_vars using accum_ops
            for i in range(cfg.GRADIENT_ACCUMULATION):
                sess.run(accum_ops, feed_dict=feed_dict)
            train_loss, ls_ce, ls_l2, lr = sess.run([loss, ce_loss, l2_losses, learning_rate], feed_dict=feed_dict)

            # Run the train_step ops to update the weights based on your accumulated gradients
            sess.run(train_step, feed_dict=feed_dict)

            time_e = time.time()

            print("Epo: %d, Step: %d, Train_loss:%g, ce: %g, l2:%g,  lr:%g, time:%g" %
                  (epo, running_count, train_loss, ls_ce, ls_l2, lr, time_e - time_s))

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
