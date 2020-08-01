import cv2
import numpy as np
from PIL import Image

import cfg

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as ktf
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def location(model, img_path, pixel_threshold):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.image_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    preprocess_input(img,mode='tf')
    x = np.expand_dims(img, axis=0)
    y = model.predict(x)
    y = np.squeeze(y, axis=0)
    y[:, :, :3] = tf.nn.sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    results = []
    with Image.open(img_path) as im:
        d_wight, d_height = resize_image(im, cfg.image_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):
            if np.amin(score) > 0:
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                #im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
                im = crop_rectangle(im, rescaled_geo)
                results.append(im)
    return results

def recognition(model, img):
    img_size = img.shape
    if (img_size[1] / img_size[0] * 1.0) < 6:
        img_reshape = cv2.resize(img, (int(31.0 / img_size[0] * img_size[1]), cfg.height))

        mat_ori = np.zeros((cfg.height, cfg.width - int(31.0 / img_size[0] * img_size[1]), 3), dtype=np.uint8)
        out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
    else:
        out_img = cv2.resize(img, (cfg.width, cfg.height), interpolation=cv2.INTER_CUBIC)
        out_img = np.asarray(out_img)
        out_img = out_img.transpose([1, 0, 2])

    y_pred = model.predict(np.expand_dims(out_img, axis=0))
    shape = y_pred[:, 2:, :].shape
    ctc_decode = ktf.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = ktf.get_value(ctc_decode)[:, :cfg.label_len]
    result = ''.join([cfg.characters[k] for k in out[0]])
    return result

def loc_and_rec(location_model,recognition_model,img):
    
    d_wight, d_height = resize_image(img, cfg.image_size)
    img = cv2.resize(img,(d_wight, d_height))
    cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    st=time.time()
    x = preprocess_input(img,mode='tf')
    x = np.expand_dims(x, axis=0)
    y = location_model.predict(x)
    y = np.squeeze(y, axis=0)
    y[:, :, :3] = tf.nn.sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], cfg.pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    results=[]
    for score, geo, s in zip(quad_scores, quad_after_nms,range(len(quad_scores))):
        if np.amin(score) > 0:
            #rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo = geo

            ##字符序列识别（训练时使用BGR格式图片）
            
            imcrop = crop_rectangle(img, rescaled_geo)
            results.append(imcrop)
            re_text = recognition(recognition_model, imcrop)
            cv2.putText(img,'value:'+re_text,org=(100,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,
                        color=(255,0,0),thickness=2,lineType=cv2.LINE_AA) 
            
            ##四边形描框
            darwgeo=rescaled_geo.astype(np.int32)
            img=cv2.polylines(img,[darwgeo],True,(255,0,0),2)

    et=time.time()
    cv2.putText(img,'FPS:'+'%.1f' % (1/(et-st)),org=(300,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,
                color=(0,255,0),thickness=2,lineType=cv2.LINE_AA) 

    return img


def resize_image(im, max_img_size=cfg.image_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor)


def region_neighbor(region_set):
    region_pixels = np.array(list(region_set))
    j_min = np.amin(region_pixels, axis=0)[1] - 1
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor


def region_group(region_list):
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    return D


def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))
    return rows


def nms(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
    region_list = []
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
        if not merge:
            region_list.append({(i, j)})
    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        for row in group:
            for ij in region_list[row]:
                score = predict[ij[0], ij[1], 1]
                if score >= threshold:
                    ith_score = predict[ij[0], ij[1], 2:3]
                    if not (cfg.trunc_threshold <= ith_score < 1 -
                            cfg.trunc_threshold):
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        px = (ij[1] + 0.5) * cfg.pixel_size
                        py = (ij[0] + 0.5) * cfg.pixel_size
                        p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7],
                                              (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + cfg.epsilon)
    return score_list, quad_list


def crop_rectangle(img, geo):
    rect = cv2.minAreaRect(geo.astype(int))
    center, size, angle = rect[0], rect[1], rect[2]
    if(angle > -45):
        center = tuple(map(int, center))
        size = tuple([int(rect[1][0] + 10), int(rect[1][1] + 10)])
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    else:
        center = tuple(map(int, center))
        size = tuple([int(rect[1][1] + 10), int(rect[1][0]) + 10])
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop

def quad_loss(y_true, y_pred):
    # loss for inside_score
    logits = y_pred[:, :, :, :1]
    labels = y_true[:, :, :, :1]
    # balance positive and negative samples in an image
    beta = 1 - tf.reduce_mean(labels)
    # first apply sigmoid activation
    predicts = tf.nn.sigmoid(logits)
    # log +epsilon for stable cal
    inside_score_loss = tf.reduce_mean(
        -1 * (beta * labels * tf.math.log(predicts + cfg.epsilon) +
              (1 - beta) * (1 - labels) * tf.math.log(1 - predicts + cfg.epsilon)))
    inside_score_loss *= cfg.lambda_inside_score_loss

    # loss for side_vertex_code
    vertex_logits = y_pred[:, :, :, 1:3]
    vertex_labels = y_true[:, :, :, 1:3]
    vertex_beta = 1 - (tf.reduce_mean(y_true[:, :, :, 1:2])
                       / (tf.reduce_mean(labels) + cfg.epsilon))
    vertex_predicts = tf.nn.sigmoid(vertex_logits)
    pos = -1 * vertex_beta * vertex_labels * tf.math.log(vertex_predicts +
                                                    cfg.epsilon)
    neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * tf.math.log(
        1 - vertex_predicts + cfg.epsilon)
    positive_weights = tf.cast(tf.equal(y_true[:, :, :, 0], 1), tf.float32)
    side_vertex_code_loss = \
        tf.reduce_sum(tf.reduce_sum(pos + neg, axis=-1) * positive_weights) / (
                tf.reduce_sum(positive_weights) + cfg.epsilon)
    side_vertex_code_loss *= cfg.lambda_side_vertex_code_loss

    # loss for side_vertex_coord delta
    g_hat = y_pred[:, :, :, 3:]
    g_true = y_true[:, :, :, 3:]
    vertex_weights = tf.cast(tf.equal(y_true[:, :, :, 1], 1), tf.float32)
    pixel_wise_smooth_l1norm = smooth_l1_loss(g_hat, g_true, vertex_weights)
    side_vertex_coord_loss = tf.reduce_sum(pixel_wise_smooth_l1norm) / (
            tf.reduce_sum(vertex_weights) + cfg.epsilon)
    side_vertex_coord_loss *= cfg.lambda_side_vertex_coord_loss
    return inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss


def smooth_l1_loss(prediction_tensor, target_tensor, weights):
    n_q = tf.reshape(quad_norm(target_tensor), tf.shape(weights))
    diff = prediction_tensor - target_tensor
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    pixel_wise_smooth_l1norm = (tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
        axis=-1) / n_q) * weights
    return pixel_wise_smooth_l1norm


def quad_norm(g_true):
    shape = tf.shape(g_true)
    delta_xy_matrix = tf.reshape(g_true, [-1, 2, 2])
    diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
    square = tf.square(diff)
    distance = tf.sqrt(tf.reduce_sum(square, axis=-1))
    distance *= 4.0
    distance += cfg.epsilon
    return tf.reshape(distance, shape[:-1])
