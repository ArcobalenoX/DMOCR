#location
image_size = 512
pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
pixel_size = 4
epsilon = 1e-4

location_model = "location\saved_model\location_model.h5"
location_weights = "location\saved_model\location_weights.h5"

#recognition
height = 64
width = 128
label_len = 8
characters = '0123456789.-' + '|'
label_classes = len(characters)

recognition_model = "recognition\saved_model\CRNN_model.h5"
recognition_weights = "recognition\saved_model\CRNN_weights.h5"


