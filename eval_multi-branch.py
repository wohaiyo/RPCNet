from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob
import imageio
import csv
import config as cfg
from networks.RPCNet import inference_prior_occ
from utils.utils import data_crop_eval_output, fast_hist, pred_vision, eval_pred, pred_vision_art, pred_vision_binary

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "multi_eval", "single_eval / multi_eval / inference / test_time to evaluate.")

CROP_OCC = False

if 'artdeco' in cfg.DATA_DIR:
    DATASET_NUM_CLASSESS = 8
else:
    DATASET_NUM_CLASSESS = 9
test_file = cfg.DATA_DIR + 'val.txt'
test_dir = cfg.EVAL_DIR

if DATASET_NUM_CLASSESS == 8:
    CLASSES_NAMES = ['Background', 'Door', 'Shop', 'Balcony', 'Window', 'Wall', 'Sky', 'Roof']    # Art-deco
else:
    CLASSES_NAMES = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']  # ECP


def main(argv=None):
    image = tf.placeholder(tf.float32, shape=[1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3], name="input_image")

    # Define model
    _, logits, logits_occ, _, logits_win_bal = inference_prior_occ(image, is_training=False)

    logits = tf.nn.softmax(logits)


    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()

    # Load parameters
    files = os.path.join(cfg.SAVE_DIR + 'model.ckpt-*.index')
    sfile = glob.glob(files)
    if len(sfile) >= 0:
        sess.run(tf.global_variables_initializer())

        sfile = glob.glob(files)
        steps = []
        for s in sfile:
            part = s.split('.')
            step = int(part[1].split('-')[1])
            steps.append(step)
        epo = max(steps)

        model = cfg.SAVE_DIR + 'model.ckpt-' + str(epo)

        print('\nRestoring weights from: ' + model)
        saver.restore(sess, model)
        print('End Restore')

    else:
        # restore from pre-train on imagenet or pre-trained
        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))
        print('Model initialized random.')

    eval_save_dir = cfg.SAVE_DIR + 'output/'
    if not os.path.exists(eval_save_dir):
        os.mkdir(eval_save_dir)


    f_test = open(test_file, 'r')
    img_list = []
    label_list = []
    occ_list = []

    for line in f_test:
        try:
            img, label, occ, _, _ = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            img = label = occ = _ = _ = line.strip("\n")
        img_list.append(test_dir + img)
        label_list.append(test_dir + label)
        occ_list.append(test_dir + occ)

    f_test.close()

    if FLAGS.mode == "multi_eval":
        print('---------Start multi-scale test img-------------')
        import cv2
        crop_size_h = cfg.IMAGE_HEIGHT
        crop_size_w = cfg.IMAGE_WIDTH
        print('crop size: ' + str(crop_size_h))
        stride = int(crop_size_w / 3)

        # File test result
        f = open(cfg.SAVE_DIR + 'output/result.txt', 'w')

        total_acc_cls = []
        total_tp_num = []
        total_all_num = []

        # Define confusion matrix
        hist = np.zeros((DATASET_NUM_CLASSESS, DATASET_NUM_CLASSESS))

        # Start evaluating
        for item in range(len(img_list)):
            # Read label
            label = imageio.imread(label_list[item])
            if CROP_OCC:
                mask = imageio.imread(occ_list[item]) / 255    # occlusion mask
                label = np.array(label * mask ,np.uint8)        # class GT

            # Read images
            ori_img = cv2.imread(img_list[item])
            im_name = img_list[item].split('/')[-1]
            ori_img_h, ori_img_w, _ = ori_img.shape

            # Scales to eval
            scs = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

            maps = []
            maps_occ = []
            maps_win_bal = []
            for sc in scs:
                img = cv2.resize(ori_img, (int(float(ori_img_w) * sc), int(float(ori_img_h) * sc)),
                                 interpolation=cv2.INTER_LINEAR)
                score_map = data_crop_eval_output(sess, image, logits, img, cfg.IMG_MEAN, crop_size_h,
                                                  crop_size_w, stride, DATASET_NUM_CLASSESS)
                score_map_occ = data_crop_eval_output(sess, image, logits_occ, img, cfg.IMG_MEAN, crop_size_h,
                                                  crop_size_w, stride, 2)
                score_map_win_bal = data_crop_eval_output(sess, image, logits_win_bal, img, cfg.IMG_MEAN, crop_size_h,
                                                  crop_size_w, stride, 3)

                score_map = cv2.resize(score_map, (ori_img_w, ori_img_h), interpolation=cv2.INTER_LINEAR)
                score_map_occ = cv2.resize(score_map_occ, (ori_img_w, ori_img_h), interpolation=cv2.INTER_LINEAR)
                score_map_win_bal = cv2.resize(score_map_win_bal, (ori_img_w, ori_img_h), interpolation=cv2.INTER_LINEAR)

                maps.append(score_map)
                maps_occ.append(score_map_occ)
                maps_win_bal.append(score_map_win_bal)
            score_map = np.mean(np.stack(maps), axis=0)
            score_map_occ = np.mean(np.stack(maps_occ), axis=0)
            score_map_win_bal = np.mean(np.stack(maps_win_bal), axis=0)

            maps2 = []
            maps2_occ = []
            maps2_win_bal = []
            for sc in scs:
                img2 = cv2.resize(ori_img, (int(float(ori_img_w) * sc), int(float(ori_img_h) * sc)),
                                  interpolation=cv2.INTER_LINEAR)
                img2 = cv2.flip(img2, 1)

                score_map2 = data_crop_eval_output(sess, image, logits, img2, cfg.IMG_MEAN, crop_size_h,
                                                   crop_size_w, stride, DATASET_NUM_CLASSESS)
                score_map2_occ = data_crop_eval_output(sess, image, logits_occ, img2, cfg.IMG_MEAN, crop_size_h,
                                                   crop_size_w, stride, 2)
                score_map2_win_bal = data_crop_eval_output(sess, image, logits_win_bal, img2, cfg.IMG_MEAN, crop_size_h,
                                                   crop_size_w, stride, 3)

                score_map2 = cv2.resize(score_map2, (ori_img_w, ori_img_h), interpolation=cv2.INTER_LINEAR)
                score_map2_occ = cv2.resize(score_map2_occ, (ori_img_w, ori_img_h), interpolation=cv2.INTER_LINEAR)
                score_map2_win_bal = cv2.resize(score_map2_win_bal, (ori_img_w, ori_img_h), interpolation=cv2.INTER_LINEAR)

                maps2.append(score_map2)
                maps2_occ.append(score_map2_occ)
                maps2_win_bal.append(score_map2_win_bal)
            score_map2 = np.mean(np.stack(maps2), axis=0)
            score_map2_occ = np.mean(np.stack(maps2_occ), axis=0)
            score_map2_win_bal = np.mean(np.stack(maps2_win_bal), axis=0)

            score_map2 = cv2.flip(score_map2, 1)
            score_map2_occ = cv2.flip(score_map2_occ, 1)
            score_map2_win_bal = cv2.flip(score_map2_win_bal, 1)

            # Mean
            score_map = (score_map + score_map2) / 2
            score_map_occ = (score_map_occ + score_map2_occ) / 2
            score_map_win_bal = (score_map_win_bal + score_map2_win_bal) / 2

            pred_label = np.argmax(score_map, 2)[:, :, np.newaxis]
            pred_label_ori = pred_label.copy()
            pred_label_occ = np.argmax(score_map_occ, 2)[:, :, np.newaxis]
            pred_label_win_bal = np.argmax(score_map_win_bal, 2)[:, :, np.newaxis]

            # Fusion module
            # # 1. Fill the occluison region with wall

            pred_label_copy = pred_label

            # 2. Fill the occlusion with window and blacony prior
            prior_bg = np.ones(pred_label_win_bal.shape, np.int32)
            prior_bg[pred_label_win_bal == 0] = 0
            pred_label_win_bal_copy = pred_label_win_bal.copy()
            pred_label_win_bal[pred_label_win_bal_copy == 1] = cfg.WINDOW_CHANNEL  # change window(prior: 1) to window(ecp:1, art: 4)
            pred_label_win_bal[pred_label_win_bal_copy == 2] = cfg.BALCONY_CHANNEL  # change balcony(prior: 2) to balcony(ecp: 3, art: 3)

            # 3. Merge two part
            pred_label = pred_label * (1 - prior_bg) + prior_bg * pred_label_win_bal

            # Save to path
            save_name = eval_save_dir + im_name.replace('.jpg', '.png')
            save_name_gt = save_name.split('.')[0] + '_gt.png'
            save_name_ori = save_name.split('.')[0] + '_ori.png'
            save_name_wall = save_name.split('.')[0] + '_wall.png'
            save_name_win_bal = save_name.split('.')[0] + '_win_bal.png'
            if DATASET_NUM_CLASSESS == 8:
                pred_vision_art(pred_label, save_name)
                pred_vision_art(np.expand_dims(label, 2), save_name_gt)
                pred_vision_art(pred_label_win_bal, save_name_win_bal)
                pred_vision_art(pred_label_ori, save_name_ori)
                pred_vision_art(pred_label_copy, save_name_wall)
            else:
                pred_vision(pred_label, save_name)
                pred_vision(np.expand_dims(label, 2), save_name_gt)
                pred_vision(pred_label_win_bal, save_name_win_bal)
                pred_vision(pred_label_ori, save_name_ori)
                pred_vision(pred_label_copy, save_name_wall)
            save_name_occ = save_name.split('.')[0] + '_occ.png'
            pred_vision_binary(pred_label_occ,save_name_occ)

            # Squeeze the axis 2
            pred_label = np.squeeze(pred_label, axis=2)

            # cumulate the hist
            hist += fast_hist(label.flatten(), pred_label.flatten(), DATASET_NUM_CLASSESS)

            # Eval singleimage
            cls_acc, img_acc, tp_num, all_num = eval_pred(label, pred_label, DATASET_NUM_CLASSESS)

            for cls in range(len(cls_acc)):
                print(CLASSES_NAMES[cls] + ': ' + str(cls_acc[cls]))
                f.write(CLASSES_NAMES[cls] + ': ' + str(cls_acc[cls]) + '\n')
            print('img-' + im_name + ': ' + str(img_acc))
            f.write('img-' + im_name + ' : ' + str(img_acc))
            print('-----------------------------')
            f.write('-------------------------------' + '\n')
            print('\n')
            f.write('\n')

            total_acc_cls.append(cls_acc)
            total_tp_num.append(tp_num)
            total_all_num.append(all_num)

        print('############### Summary #################')
        f.write('############### Summary #################\n')
        print('=>')
        f.write('=>\n')


        # Overall acc
        alpha =1e-8
        hist[0, :] = 0  # Ignore outlier
        over_acc = np.diag(hist).sum() / (hist.sum() + alpha)
        acc = np.diag(hist) / (hist.sum(1) + alpha)

        print('Overall accuracy', over_acc)
        f.write('Overall accuracy' + str(over_acc) + '\n')
        mean_acc = sum(acc) / (len(acc) - 1)
        print('Mean accuracy', mean_acc)
        f.write('Mean accuracy' + str(mean_acc) + '\n')
        print('Class accuracy', acc)
        f.write('Class accuracy' + str(acc) + '\n')

        # per-class IU
        numerator = np.diag(hist)
        denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
        numerator_noBg = np.delete(numerator, 0, axis=0)
        denominator_noBg = np.delete(denominator, 0, axis=0)
        iu = numerator_noBg / (denominator_noBg + alpha)
        print('IoU ' + str(iu))
        f.write('IoU ' + str(iu) + '\n')
        print('mean IoU ', np.nanmean(iu))
        f.write('mean IoU ' + str(np.nanmean(iu)) + '\n')

        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # f1-score
        f1_scores = []
        for c in range(1, DATASET_NUM_CLASSESS):
            TP = hist[c][c]
            FP = np.sum(hist[:, c]) - hist[c][c]
            FN = np.sum(hist[c, :]) - hist[c][c]
            precision = TP / (TP + FP + alpha)
            recall = TP / (TP + FN + alpha)

            f1 = (2 * precision * recall) / (precision + recall + alpha)
            f1_scores.append(f1)

        mean_f1_score = sum(f1_scores) / len(f1_scores)
        print('f1_score: ' + str(mean_f1_score))
        f.write('f1 score: ' + str(mean_f1_score) + '\n')

        total_acc_cls = np.array(total_acc_cls)
        total_tp_num = np.array(total_tp_num)
        total_all_num = np.array(total_all_num)
        print('Total Accuracy: ')
        f.write('Total Accuracy: \n')


        filename = cfg.SAVE_DIR + 'output/acc.csv'
        f_csv = open(filename, 'w')
        writer = csv.writer(f_csv)

        for column in range(total_acc_cls.shape[1]):

            cls_tp_num = []
            cls_all_num = []

            for row in range(total_acc_cls.shape[0]):
                cls_tp_num.append(total_tp_num[row][column])
                cls_all_num.append(total_all_num[row][column])

            print(CLASSES_NAMES[column+1] + '-acc:' + str(sum(cls_tp_num) / sum(cls_all_num)))
            f.write(CLASSES_NAMES[column+1] + '-acc:' + str(sum(cls_tp_num) / sum(cls_all_num)) + '\n')
            writer.writerow([CLASSES_NAMES[column+1], str(sum(cls_tp_num) / sum(cls_all_num))])

        print('\nAcc:' + str(np.sum(total_tp_num) / np.sum(total_all_num)))
        f.write('\nAcc:' + str(np.sum(total_tp_num) / np.sum(total_all_num)) + '\n')
        writer.writerow(['Total acc', str(np.sum(total_tp_num) / np.sum(total_all_num))])

        f_csv.close()
        f.close()


if __name__ == "__main__":
    tf.app.run()