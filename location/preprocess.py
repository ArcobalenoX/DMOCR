import os
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import cfg

#image process

def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


def reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
                    / (xy_list[index, 0] - xy_list[first_v, 0] + cfg.epsilon)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + cfg.epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


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


def preprocess():
    data_dir = cfg.data_dir
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    draw_gt_quad = cfg.draw_gt_quad
    show_gt_image_dir = os.path.join(data_dir, cfg.show_gt_image_dir_name)
    if not os.path.exists(show_gt_image_dir):
        os.mkdir(show_gt_image_dir)
    show_act_image_dir = os.path.join(cfg.data_dir, cfg.show_act_image_dir_name)
    if not os.path.exists(show_act_image_dir):
        os.mkdir(show_act_image_dir)

    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))
    train_val_set = []
    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))):
        with Image.open(os.path.join(origin_image_dir, o_img_fname)) as im:
            # d_wight, d_height = resize_image(im)
            d_wight, d_height = cfg.image_size, cfg.image_size
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            try:
                im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
                show_gt_im = im.copy()
                # draw on the img
                draw = ImageDraw.Draw(show_gt_im)
                with open(os.path.join(origin_txt_dir,
                                   o_img_fname[:-4] + '.txt'), 'r') as f:
                    anno_list = f.readlines()
                    #print(anno_list)
                xy_list_array = np.zeros((len(anno_list), 4, 2))
                for anno, i in zip(anno_list, range(len(anno_list))):
                    anno_colums = anno.strip().split(',')
                    anno_array = np.array(anno_colums)
                    xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                    xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                    xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                    xy_list = reorder_vertexes(xy_list)
                    xy_list_array[i] = xy_list
                    _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                    shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)
                    if draw_gt_quad:
                        draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
                               tuple(xy_list[2]), tuple(xy_list[3]),
                               tuple(xy_list[0])
                               ],
                               width=2, fill='green')
                        draw.line([tuple(shrink_xy_list[0]),
                               tuple(shrink_xy_list[1]),
                               tuple(shrink_xy_list[2]),
                               tuple(shrink_xy_list[3]),
                               tuple(shrink_xy_list[0])
                               ],
                               width=2, fill='blue')
                        vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                             [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                        for q_th in range(2):
                            draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                                   tuple(shrink_1[vs[long_edge][q_th][1]]),
                                   tuple(shrink_1[vs[long_edge][q_th][2]]),
                                   tuple(xy_list[vs[long_edge][q_th][3]]),
                                   tuple(xy_list[vs[long_edge][q_th][4]])],
                                   width=3, fill='yellow')
                if cfg.gen_origin_img:
                    im.save(os.path.join(train_image_dir, o_img_fname))
                np.save(os.path.join(train_label_dir,o_img_fname[:-4]+'.npy'),xy_list_array)
                if draw_gt_quad:
                    show_gt_im.save(os.path.join(show_gt_image_dir, o_img_fname))
                train_val_set.append(f'{o_img_fname},{d_wight},{d_height}\n')
            except Exception:
                pass

    train_img_list = os.listdir(train_image_dir)
    print('found %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))

    random.shuffle(train_val_set)
    val_count = int(cfg.validation_split_ratio * len(train_val_set))
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:
        f_val.writelines(train_val_set[:val_count])
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train:
        f_train.writelines(train_val_set[val_count:])


#label process

def point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False


def point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge):
    nth = -1
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
    for ith in range(2):
        quad_xy_list = np.concatenate((
            np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
            np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
        p_min = np.amin(quad_xy_list, axis=0)
        p_max = np.amax(quad_xy_list, axis=0)
        if point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
            if nth == -1:
                nth = ith
            else:
                nth = -1
                break
    return nth


def shrink(xy_list, ratio=cfg.shrink_ratio):
    if ratio == 0.0:
        return xy_list, xy_list
    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
    diff = np.concatenate((diff_1to3, diff_4), axis=0)
    dis = np.sqrt(np.sum(np.square(diff), axis=-1))
    # determine which are long or short edges
    long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))
    short_edge = 1 - long_edge
    # cal r length array
    r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
    # cal theta array
    diff_abs = np.abs(diff)
    diff_abs[:, 0] += cfg.epsilon
    theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])
    # shrink two long edges
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, ratio)
    shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, ratio)
    # shrink two short edges
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, ratio)
    return temp_new_xy_list, new_xy_list, long_edge


def shrink_edge(xy_list, new_xy_list, edge, r, theta, ratio=cfg.shrink_ratio):
    if ratio == 0.0:
        return
    start_point = edge
    end_point = (edge + 1) % 4
    long_start_sign_x = np.sign(
        xy_list[end_point, 0] - xy_list[start_point, 0])
    new_xy_list[start_point, 0] = \
        xy_list[start_point, 0] + \
        long_start_sign_x * ratio * r[start_point] * np.cos(theta[start_point])
    long_start_sign_y = np.sign(
        xy_list[end_point, 1] - xy_list[start_point, 1])
    new_xy_list[start_point, 1] = \
        xy_list[start_point, 1] + \
        long_start_sign_y * ratio * r[start_point] * np.sin(theta[start_point])
    # long edge one, end point
    long_end_sign_x = -1 * long_start_sign_x
    new_xy_list[end_point, 0] = \
        xy_list[end_point, 0] + \
        long_end_sign_x * ratio * r[end_point] * np.cos(theta[start_point])
    long_end_sign_y = -1 * long_start_sign_y
    new_xy_list[end_point, 1] = \
        xy_list[end_point, 1] + \
        long_end_sign_y * ratio * r[end_point] * np.sin(theta[start_point])


def process_label(data_dir=cfg.data_dir):
    with open(os.path.join(data_dir, cfg.val_fname), 'r') as f_val:
        f_list = f_val.readlines()
    with open(os.path.join(data_dir, cfg.train_fname), 'r') as f_train:
        f_list.extend(f_train.readlines())
    for line, _ in zip(f_list, tqdm(range(len(f_list)))):
        line_cols = str(line).strip().split(',')
        img_name, width, height = \
            line_cols[0].strip(), int(line_cols[1].strip()), \
            int(line_cols[2].strip())
        gt = np.zeros((height // cfg.pixel_size, width // cfg.pixel_size, 7))
        train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
        xy_list_array = np.load(os.path.join(train_label_dir,
                                             img_name[:-4] + '.npy'))
        train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
        with Image.open(os.path.join(train_image_dir, img_name)) as im:
            draw = ImageDraw.Draw(im)
            for xy_list in xy_list_array:
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)
                p_min = np.amin(shrink_xy_list, axis=0)
                p_max = np.amax(shrink_xy_list, axis=0)
                # floor of the float
                ji_min = (p_min / cfg.pixel_size - 0.5).astype(int) - 1
                # +1 for ceil of the float and +1 for include the end
                ji_max = (p_max / cfg.pixel_size - 0.5).astype(int) + 3
                imin = np.maximum(0, ji_min[1])
                imax = np.minimum(height // cfg.pixel_size, ji_max[1])
                jmin = np.maximum(0, ji_min[0])
                jmax = np.minimum(width // cfg.pixel_size, ji_max[0])
                for i in range(imin, imax):
                    for j in range(jmin, jmax):
                        px = (j + 0.5) * cfg.pixel_size
                        py = (i + 0.5) * cfg.pixel_size
                        if point_inside_of_quad(px, py,
                                                shrink_xy_list, p_min, p_max):
                            gt[i, j, 0] = 1
                            line_width, line_color = 1, 'red'
                            ith = point_inside_of_nth_quad(px, py,
                                                           xy_list,
                                                           shrink_1,
                                                           long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                            if ith in range(2):
                                gt[i, j, 1] = 1
                                if ith == 0:
                                    line_width, line_color = 2, 'yellow'
                                else:
                                    line_width, line_color = 2, 'green'
                                gt[i, j, 2:3] = ith
                                gt[i, j, 3:5] = \
                                    xy_list[vs[long_edge][ith][0]] - [px, py]
                                gt[i, j, 5:] = \
                                    xy_list[vs[long_edge][ith][1]] - [px, py]
                            draw.line([(px - 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size),
                                       (px + 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size),
                                       (px + 0.5 * cfg.pixel_size,
                                        py + 0.5 * cfg.pixel_size),
                                       (px - 0.5 * cfg.pixel_size,
                                        py + 0.5 * cfg.pixel_size),
                                       (px - 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size)],
                                      width=line_width, fill=line_color)
            act_image_dir = os.path.join(cfg.data_dir,
                                         cfg.show_act_image_dir_name)
            if cfg.draw_act_quad:
                im.save(os.path.join(act_image_dir, img_name))
        train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
        np.save(os.path.join(train_label_dir,img_name[:-4] + '_gt.npy'), gt)

if __name__ == '__main__':
    preprocess()
    process_label()