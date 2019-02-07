num_parallel_calls = 4
num_classes = 3 # widerface: 1, multiaffect: 3

## Training dataset
train_batch_size = 10 # batch size
train_num =  50000 # widerface:12880 ,multiaffect: 95852
classes_path = './model_data/multiaffectnet_classes.txt' # widerface: './model_data/widerface_classes.txt'  , multiaffect: './model_data/multiaffectnet_classes.txt'
train_data_file = '/media/data/hojjat_data/affectnet/' #  '/home/hojjat/widerface/WIDER_train' '/media/data/hojjat_data/multiaffectnet/'
train_annotations_file = '/media/data/hojjat_data/affectnet/original25k.csv' # '/home/hojjat/widerface/wider_face_split/wider_face_train_bbx_gt.txt'  '/media/data/hojjat_data/multiaffectnet/file_list.csv'
gpu_index = "2"
input_shape = 416
max_boxes = 10
jitter = 0.3
hue = 0.1
sat = 1.0
cont = 0.8
bri = 0.1
norm_decay = 0.99
norm_epsilon = 1e-3
pre_train = True # If pretrain is on, then it only trains the yolo part.
num_anchors = 9
training = True
ignore_thresh = .5

## Learning Rate
learning_rate = 0.001
decay_step = 1000 # after decay_step the learning rate is multiplied with decay_rate
decay_rate = 0.96

## Validation
val_batch_size = 10
val_num = 5000

## Total number of epochs
Epoch = 100
obj_threshold = 0.3
nms_threshold = 0.5
log_dir = './logs/affectnet_backbone_1'
data_dir = '/media/data/hojjat_data/affectnet'
model_dir = './test_model/affectnet_backbone_1'
pre_train_yolo3 = False
yolo3_weights_path = './model_data/yolov3.weights'
darknet53_weights_path = './model_data/darknet53.weights'
anchors_path = './model_data/yolo_anchors.txt'
val_data_file = '/data0/dataset/coco/val2017'
val_annotations_file = '/data0/gaochen3/tensorflow-yolo3/annotations/instances_val2017.json'
