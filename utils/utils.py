import cv2
import numpy as np
from tensorflow.python import pywrap_tensorflow

def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")

def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
        if v.name.split(':')[0] in var_keep_dic:
            # print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)
    return variables_to_restore

def get_variables_to_restore_rpcnet(variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
        if v.name.split(':')[0] in var_keep_dic and 'rpcnet' in v.name.split(':')[0]:
            # print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)
    return variables_to_restore

def data_crop_eval_output(session, gr_data, logits, image, mean_bgr, crop_size_h, crop_size_w, stride, num_classes):
    '''
    Multi-scale evaluation for segmentation

    :return: A predicted labels.
    '''
    image_h = image.shape[0]
    image_w = image.shape[1]
    pad_h = 0
    pad_w = 0
    if image_h >= crop_size_h and image_w >= crop_size_w:
        image_pad = image
    else:
        if image_h < crop_size_h:
            pad_h = crop_size_h - image_h
        if image_w < crop_size_w:
            pad_w = crop_size_w - image_w
        image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0)
    image_pad = np.asarray(image_pad, dtype='float32')

    # sub bgr mean
    image_pad = image_pad - mean_bgr

    image_crop_batch = []
    x_start = [x for x in range(0, image_pad.shape[0] - crop_size_h + 1, stride)]
    y_start = [y for y in range(0, image_pad.shape[1] - crop_size_w + 1, stride)]
    if (image_pad.shape[0] - crop_size_h) % stride != 0:
        x_start.append(image_pad.shape[0] - crop_size_h)
    if (image_pad.shape[1] - crop_size_w) % stride != 0:
        y_start.append(image_pad.shape[1] - crop_size_w)
    for x in x_start:
        for y in y_start:
            image_crop_batch.append(image_pad[x:x + crop_size_h, y:y + crop_size_w])

    logit = []
    for crop_batch in image_crop_batch:
        lo = session.run(
        logits,
        feed_dict={
            gr_data: [crop_batch]
        })
        logit.append(lo[0])


    num_class = num_classes
    score_map = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype='float32')
    count = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype='float32')
    crop_index = 0
    for x in x_start:
        for y in y_start:
            crop_logits = logit[crop_index]
            score_map[x:x + crop_logits.shape[0], y:y + crop_logits.shape[1], :] += crop_logits
            count[x:x + crop_logits.shape[0], y:y + crop_logits.shape[1], :] += 1
            crop_index += 1

    score_map = score_map[:image_h, :image_w] / count[:image_h, :image_w]
    return score_map

def pred_vision(pred, save_name):  # pred [h, w, 1]
    '''
    Visulize the predicted image
    :param pred: label [h, w, 1]
    :param save_name: save path
    :return:
    '''
    pred = np.array(pred)
    height = pred.shape[0]
    width = pred.shape[1]

    pred_B = np.zeros([height, width, 1], dtype=np.uint8)
    pred_G = np.zeros([height, width, 1], dtype=np.uint8)
    pred_R = np.zeros([height, width, 1], dtype=np.uint8)


    label_color = [[0, 0, 0, 0],
                   [1, 0, 0, 255],
                   [2, 0, 255, 255],
                   [3, 255, 0, 128],
                   [4, 0, 128, 255],
                   [5, 255, 0, 0],
                   [6, 128, 128, 128],
                   [7, 255, 255, 128],
                   [8, 0, 255, 0],
                   ]
    label_color = np.array(label_color, np.int)
    for i in range(label_color.shape[0]):
        pred_B[pred == label_color[i][0]] = label_color[i][1]
        pred_G[pred == label_color[i][0]] = label_color[i][2]
        pred_R[pred == label_color[i][0]] = label_color[i][3]

    pred_new = np.concatenate([pred_B, pred_G, pred_R], 2)
    cv2.imwrite(save_name, pred_new)
    print(save_name + ' is saved.')

def pred_vision_art(pred, save_name):  # pred [h, w, 1]
    '''
    Visulize the predicted image
    :param pred: label [h, w, 1]
    :param save_name: save path
    :return:
    '''
    pred = np.array(pred)
    height = pred.shape[0]
    width = pred.shape[1]

    pred_B = np.zeros([height, width, 1], dtype=np.uint8)
    pred_G = np.zeros([height, width, 1], dtype=np.uint8)
    pred_R = np.zeros([height, width, 1], dtype=np.uint8)

    label_color = [[0, 0, 0, 0],
                   [1, 0, 128, 255],
                   [2, 0, 255, 0],
                   [3, 255, 0, 128],
                   [4, 0, 0, 255],
                   [5, 0, 255, 255],
                   [6, 255, 255, 128],
                   [7, 255, 0, 0],
                   ]
    label_color = np.array(label_color, np.int)
    for i in range(label_color.shape[0]):
        pred_B[pred == label_color[i][0]] = label_color[i][1]
        pred_G[pred == label_color[i][0]] = label_color[i][2]
        pred_R[pred == label_color[i][0]] = label_color[i][3]

    pred_new = np.concatenate([pred_B, pred_G, pred_R], 2)
    cv2.imwrite(save_name, pred_new)
    print(save_name + ' is saved.')

def pred_vision_binary(pred, save_name):  # pred [h, w, 1]
    '''
    Visulize the predicted image
    :param pred: label [h, w, 1]
    :param save_name: save path
    :return:
    '''
    pred = np.array(pred)
    height = pred.shape[0]
    width = pred.shape[1]

    pred_B = np.zeros([height, width, 1], dtype=np.uint8)
    pred_G = np.zeros([height, width, 1], dtype=np.uint8)
    pred_R = np.zeros([height, width, 1], dtype=np.uint8)


    label_color = [[0, 0, 0, 0],
                   [1, 255, 255, 255],
                   ]
    label_color = np.array(label_color, np.int)
    for i in range(label_color.shape[0]):
        pred_B[pred == label_color[i][0]] = label_color[i][1]
        pred_G[pred == label_color[i][0]] = label_color[i][2]
        pred_R[pred == label_color[i][0]] = label_color[i][3]

    pred_new = np.concatenate([pred_B, pred_G, pred_R], 2)
    cv2.imwrite(save_name, pred_new)
    print(save_name + ' is saved.')

def fast_hist(a, b, n):
    # Calculate confusion matrix
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def eval_pred(gt, pred, num_classes):
    # Eval every single image
    cls_acc = []
    TP_num = []
    pos_all_total = []


    for i in range(1, num_classes):   # ignore 0
        gt_i = np.zeros(gt.shape, np.int)
        pred_i = np.zeros(gt.shape, np.int)
        zero = np.zeros(gt.shape, np.int)

        gt_i[gt == i] = 1
        pred_i[pred == i] = 1

        cls_i = gt_i.copy()
        cls_i[gt != i] = -1
        zero[cls_i == pred_i] = 1

        TP = np.sum(zero)

        pos_all = np.sum(gt_i)

        if pos_all == 0:
            cls_acc.append(-1)
        else:
            cls_acc.append(TP / pos_all)
        TP_num.append(TP)
        pos_all_total.append(pos_all)

    return cls_acc, sum(TP_num) / sum(pos_all_total),  TP_num, pos_all_total
