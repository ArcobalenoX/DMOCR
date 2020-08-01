#loss参数
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 1000
validation_split_ratio = 0.2
max_train_img_size = 512
image_size = 512         # (height == width, in [256, 384, 512, 640, 736])
batch_size = 1
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

#数据集预处理
data_dir = r'Z:\Code\Python\datas\meter512'
origin_image_dir_name = 'images/'
origin_txt_dir_name = 'east_txt/'
train_image_dir_name = 'imgs_east_%s/' % image_size
train_label_dir_name = 'labels_east_%s/' % image_size
show_gt_image_dir_name = 'show_gt_images_%s/' % image_size
show_act_image_dir_name = 'show_act_images_%s/' % image_size
train_fname = 'east_train.txt'
val_fname = 'east_val.txt'

gen_origin_img = True
draw_gt_quad = True
draw_act_quad = True
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.1
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.5
epsilon = 1e-4
# pixel_size
pixel_size = 4


#训练权重
checkpoint_path = "saved_model/weights_E{epoch:d}_L{loss:.5f}.h5"
saved_model_path = "saved_model/location_model.h5"
saved_model_weights_path = "saved_model/location_weights.h5"

#预测参数
pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_write2txt = False
detection_box_crop = True
