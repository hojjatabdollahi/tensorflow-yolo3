import os
import config
import argparse
import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from utils import letterbox_image, load_weights
import cv2
import time
# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

def detect( model_path, yolo_weights = None, image_path = None):
    """
    Introduction
    ------------
        加载模型，进行预测
    Parameters
    ----------
        model_path: 模型路径
        image_path: 图片路径
    """
    cap = None
    if image_path == None:
        cap = cv2.VideoCapture(0)
    input_image_shape = tf.placeholder(dtype = tf.int32, shape = (2,))
    input_image = tf.placeholder(shape = [None, 416, 416, 3], dtype = tf.float32)
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    classes = predictor.predict(input_image, input_image_shape)
    with tf.Session() as sess:
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                classes = predictor.predict(input_image, input_image_shape)
            load_op = load_weights(tf.global_variables(scope = 'predict'), weights_file = yolo_weights)
            sess.run(load_op)
        else:
            saver = tf.train.Saver()
            saver.restore(sess, "./test_model/model.ckpt-192192/model.ckpt-44865") # emotion
            # saver.restore(sess, "./test_model/model.ckpt-192192/model.ckpt-19940") # detection
        while True:
            start_time = time.time()    
            if  image_path ==None:    
                ret, image = cap.read()
                if ret == 0:
                    break
                [h, w] = image.shape[:2]
                print (h, w)
                image = cv2.flip(image, 1)
                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_np)
            else:
                image = Image.open(image_path)
            resize_image = letterbox_image(image, (416, 416))
            image_data = np.array(resize_image, dtype = np.float32)
            image_data /= 255.
            image_data = np.expand_dims(image_data, axis = 0)
        
            out_classes = sess.run(
                [classes],
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.size[1], image.size[0]]
                })
            print('Class:  {} '.format(str(classes)))
            font = ImageFont.truetype(font = 'font/FiraMono-Medium.otf', size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            # for i, c in reversed(list(enumerate(out_classes))):
            #     c = int(c[0])
            #     print ("i:{}, c:{}, type:{}".format(i,c, type(c)))
            #     if c > 2:
            #         continue
                
            #     predicted_class = predictor.class_names[c]
            #     box = out_boxes[i]
            #     score = out_scores[i]

            #     label = '{} {:.2f}'.format(predicted_class, score)
            #     draw = ImageDraw.Draw(image)
            #     label_size = draw.textsize(label, font)

            #     top, left, bottom, right = box
            #     top = max(0, np.floor(top + 0.5).astype('int32'))
            #     left = max(0, np.floor(left + 0.5).astype('int32'))
            #     bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            #     right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #     print(label, (left, top), (right, bottom))

            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     duration = time.time() - start_time
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle(
            #             [left + i, top + i, right - i, bottom - i],
            #             outline = predictor.colors[c])
            #     draw.rectangle(
            #         [tuple(text_origin), tuple(text_origin + label_size)],
            #         fill = predictor.colors[c])
            #     frame_rate = '{:.2f}'.format(1.0/duration)
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            #     draw.text(np.array([0,0]), frame_rate, fill = (0,0,0), font=font)
            #     del draw
            # image.show()
            # image.save('./result1.jpg')
            # cv_img = cv2.CreateImageHeader(image.size, cv2.IPL_DEPTH_8U, 3)  # RGB image
            # cv2.SetData(cv_img, image.tostring(), image.size[0]*3)

            # if  image_path != None:
            #     print('just one image')
            #     image.show()
            #     image.save('./result1.jpg')
            #     break
            # else:
            #     open_cv_image = np.array( image )[:, :, ::-1].copy() 
            #     cv2.imshow('cimage', open_cv_image)
            #     k = cv2.waitKey(1) & 0xff
            #     if k == ord('q') or k == 27:
            #         break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default = argparse.SUPPRESS)
    parser.add_argument(
        '--image_file', type = str, help = 'image file path'
    )
    FLAGS = parser.parse_args()
    try:
        im = FLAGS.image_file
    except:
        im = None
    if config.pre_train_yolo3 == True:
        detect(config.model_dir, config.yolo3_weights_path,  image_path=im)
    else:
        detect(config.model_dir, image_path=im)
