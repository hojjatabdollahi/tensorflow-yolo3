import os
import config
import json
import tensorflow as tf
import numpy as np
from collections import defaultdict

class Reader:
    def __init__(self, mode, data_dir, anchors_path, num_classes, tfrecord_num = 12, input_shape = 416, max_boxes = 20):
        """
        Introduction
        ------------
            initialize the paramters
        Parameters
        ----------
            data_dir: where to save the output tfrecords
            mode: train or val
            anchors: get the anchors
            num_classes: number of class in the dataset (widerface=1)
            input_shape: shape of the image for the network 416x416
            max_boxes: maximum number of faces per image (is 20 enough?)
        """
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.max_boxes = max_boxes
        self.mode = mode
        self.annotations_file = {'train' : config.train_annotations_file, 'val' : config.val_annotations_file}
        self.data_file = {'train': config.train_data_file, 'val': config.val_data_file}
        self.anchors_path = anchors_path
        self.anchors = self._get_anchors()
        self.num_classes = num_classes
        file_pattern = self.data_dir + "/*" + self.mode + '.tfrecords'
        self.TfrecordFile = tf.gfile.Glob(file_pattern)
        self.class_names = self._get_class(config.classes_path)
        print("Checking the tfrecords")
        if len(self.TfrecordFile) == 0:
            # self.convert_to_tfrecord(self.data_dir, tfrecord_num)
            pass

    def _get_anchors(self):
        """
        Introduction
        ------------
            get the anchors
        Returns
        -------
            anchors: array of anchors (9x2)
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_class(self, classes_path):
        """
        Introduction
        ------------
            get the name of the clases
        Returns
        -------
            class_names: widerface has only one class
        """
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def read_annotations(self):
        """
        Introduction
        ------------
            reading the annotation for widerface
        Parameters
        ----------
            data_file: wider_face_train_bbx_gt.txt
        """
        image_data = []
        boxes_data = []
        with open(self.annotations_file[self.mode], encoding='utf-8') as file:
            counter = 0
            for line in file:
                if counter >= 50000:
                    break
                counter += 1
                line_data = line.replace('"', '').strip().split(',')
                id = line_data[0]
                name = os.path.join('Y:/ExpressionNet/Manually_Annotated_Images/', id)
                image_data.append(name)
                box = [int(j) for j in line_data[1:6]]
                boxes = []
                boxes.append(np.array([box[1],box[2],box[1]+box[3],box[2]+box[4],box[0]]))
                boxes_data.append(np.array(boxes))
        return image_data, boxes_data


    def convert_to_tfrecord(self, tfrecord_path, num_tfrecords):
        """
        Introduction
        ------------
            将图片和boxes数据存储为tfRecord
        Parameters
        ----------
            tfrecord_path: tfrecord文件存储路径
            num_tfrecords: 分成多少个tfrecord
        """
        image_data, boxes_data = self.read_annotations()
        print("Finished reading the annotation, {}".format(str(len(image_data))))
        images_num = int(len(image_data) / num_tfrecords)
        for index_records in range(num_tfrecords):
            output_file = os.path.join(tfrecord_path, str(index_records) + '_' + self.mode + '.tfrecords')
            with tf.python_io.TFRecordWriter(output_file) as record_writer:
                for index in range(index_records * images_num, (index_records + 1) * images_num):
                    with tf.gfile.FastGFile(image_data[index], 'rb') as file:
                        image = file.read()
                        xmin, xmax, ymin, ymax, label = [], [], [], [], []
                        for box in boxes_data[index]:
                            xmin.append(box[0])
                            ymin.append(box[1])
                            xmax.append(box[2])
                            ymax.append(box[3])
                            label.append(box[4])
                        example = tf.train.Example(features = tf.train.Features(
                            feature = {
                                'image/encoded' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                                'image/object/bbox/xmin' : tf.train.Feature(float_list = tf.train.FloatList(value = xmin)),
                                'image/object/bbox/xmax': tf.train.Feature(float_list = tf.train.FloatList(value = xmax)),
                                'image/object/bbox/ymin': tf.train.Feature(float_list = tf.train.FloatList(value = ymin)),
                                'image/object/bbox/ymax': tf.train.Feature(float_list = tf.train.FloatList(value = ymax)),
                                'image/object/bbox/label': tf.train.Feature(float_list = tf.train.FloatList(value = label)),
                            }
                        ))
                        record_writer.write(example.SerializeToString())
                        if index % 1000 == 0:
                            print('Processed {} of {} images'.format(index + 1, len(image_data)))


    def parser(self, serialized_example):
        """
        Introduction
        ------------
            解析tfRecord数据
        Parameters
        ----------
            serialized_example: 序列化的每条数据
        """
        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image/encoded' : tf.FixedLenFeature([], dtype = tf.string),
                'image/object/bbox/xmin' : tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/xmax': tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/ymin': tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/ymax': tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/label': tf.VarLenFeature(dtype = tf.float32)
            }
        )
        image = tf.image.decode_jpeg(features['image/encoded'], channels = 3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['image/object/bbox/label'].values, tf.int32)
        one = tf.ones_like(label)
        label = tf.subtract(label, one) # removing 1 from the label to convert [1,2,3] to [0,1,2]\
        label_categorical = tf.one_hot(
            label,
            depth= 3,
            on_value=1,
            off_value=0,
            dtype=tf.int32,
        )
        label_categorical = tf.reshape(label_categorical, [3])
        label_categorical.set_shape([3])


        # bbox = tf.concat(axis = 0, values = [xmin, ymin, xmax, ymax, label-1]) 
        # bbox = tf.transpose(bbox, [1, 0])
        image = self.Preprocess(image)
        return image, label_categorical

    def Preprocess(self, image):
        """
        Introduction
        ------------
            对图片进行预处理，增强数据集
        Parameters
        ----------
            image: tensorflow解析的图片
            bbox: 图片中对应的box坐标
        """
        image_width, image_high = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[0], tf.float32)
        input_width = tf.cast(self.input_shape, tf.float32)
        input_high = tf.cast(self.input_shape, tf.float32)
        new_high = image_high * tf.minimum(input_width / image_width, input_high / image_high)
        new_width = image_width * tf.minimum(input_width / image_width, input_high / image_high)
        # Add padding to the image
        dx = (input_width - new_width) / 2
        dy = (input_high - new_high) / 2
        image = tf.image.resize_images(image, [tf.cast(new_high, tf.int32), tf.cast(new_width, tf.int32)], method = tf.image.ResizeMethod.BICUBIC)
        new_image = tf.image.pad_to_bounding_box(image, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32), tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
        image_ones = tf.ones_like(image)
        image_ones_padded = tf.image.pad_to_bounding_box(image_ones, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32), tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
        image_color_padded = (1 - image_ones_padded) * 128
        image = image_color_padded + new_image
        if self.mode == 'train':
            # randomly flip the image
            flip_left_right = tf.greater(tf.random_uniform([], dtype = tf.float32, minval = 0, maxval = 1), 0.5)
            image = tf.cond(flip_left_right, lambda : tf.image.flip_left_right(image), lambda : image)
        # 将图片归一化到0和1之间
        image = image / 255.
        image = tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)
        return image


    def build_dataset(self, batch_size):
        """
        Introduction
        ------------
            建立数据集dataset
        Parameters
        ----------
            batch_size: batch大小
        Return
        ------
            dataset: 返回tensorflow的dataset
        """
        dataset = tf.data.TFRecordDataset(filenames = self.TfrecordFile)
        dataset = dataset.map(self.parser, num_parallel_calls = config.num_parallel_calls)
        if self.mode == 'train':
            dataset = dataset.repeat(1000).shuffle(90).batch(batch_size).prefetch(batch_size)
        else:
            dataset = dataset.repeat(1000).batch(batch_size).prefetch(batch_size)
        return dataset
if __name__ == "__main__":
    reader = Reader('train', config.data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes)

