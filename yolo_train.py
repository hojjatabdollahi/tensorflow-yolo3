import os
import time
import config
import numpy as np
from PIL import Image
import tensorflow as tf
from multiaffectnet import Reader
from model.yolo3_model import yolo
from collections import defaultdict
from yolo_predict import yolo_predictor
from utils import draw_box, load_weights, letterbox_image, voc_ap

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
# tf.enable_eager_execution()

def train():
    """
    Introduction
    ------------
        This function will train the yolo3 (and you can just load the darknet weights)
    """
    train_reader = Reader('train', config.data_dir, config.anchors_path,
                          config.num_classes, input_shape=config.input_shape, max_boxes=config.max_boxes)
    train_data = train_reader.build_dataset(config.train_batch_size)
    is_training = tf.placeholder(tf.bool, shape=[])
    iterator = train_data.make_one_shot_iterator()
    images, label = iterator.get_next()
    images.set_shape([None, config.input_shape, config.input_shape, 3])
    grid_shapes = [config.input_shape // 32,
                   config.input_shape // 16, config.input_shape // 8]

    model = yolo(config.norm_epsilon, config.norm_decay,
                 config.anchors_path, config.classes_path, config.pre_train)

    output = model.yolo_inference(
        images, config.num_anchors / 3, config.num_classes, is_training)
    loss = model.yolo_loss(output, label)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss # TODO: why?
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(
        config.learning_rate, global_step, decay_steps=config.decay_step, decay_rate=0.96)
    tf.summary.scalar('learning rate', lr)
    merged_summary = tf.summary.merge_all()
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # 如果读取预训练权重，则冻结darknet53网络的变量
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if config.pre_train:
            train_var = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo')
            train_op = optimizer.minimize(
                loss=loss, global_step=global_step, var_list=train_var)
        else:
            train_op = optimizer.minimize(loss=loss, global_step=global_step)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        ckpt = tf.train.get_checkpoint_state(config.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('restore model', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)
        if config.pre_train is True:
            load_ops = load_weights(tf.global_variables(
                scope='darknet53'), config.darknet53_weights_path)
            sess.run(load_ops)
        summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)
        loss_value = 0
        for epoch in range(config.Epoch):
            for step in range(int(config.train_num / config.train_batch_size)):
                try:
                    start_time = time.time()

                    summary, train_loss, global_step_value, _ =\
                        sess.run([merged_summary, loss, global_step, train_op], feed_dict={is_training: True})
                    loss_value += train_loss
                    duration = time.time() - start_time
                    examples_per_sec = float(duration) / config.train_batch_size
                    format_str = (
                        'Epoch {} step {}, avg los: {:.3f}, train loss = {:.3f}, gs: {}, ( {:.3f} examples/sec; {:.3f} ''sec/batch)')
                    # format_str = (
                        # 'Epoch {} step {}, avg los: {:.3f}, train loss = {:.3f}, gs: {}  ( {:.3f} examples/sec; {:.3f} ''sec/batch)')
                    # print('.')
                    # print(format_str.format(epoch, step, train_loss, global_step_value, examples_per_sec, duration))
                    print(format_str.format(epoch, step, loss_value / global_step_value,  train_loss, global_step_value,
                                                examples_per_sec, duration))
                    # print(format_str.format(epoch, step, loss_value / global_step_value,  train_loss, global_step_value,
                                            # examples_per_sec, duration))
                    summary_writer.add_summary(summary=tf.Summary(value=[tf.Summary.Value(
                        tag="train loss", simple_value=train_loss)]), global_step=step)
                    summary_writer.add_summary(summary, global_step_value)
                    if step % 100:
                        summary_writer.flush()
                except Exception as ex:
                    print(ex)
            # 每3个epoch保存一次模型
            # if epoch % 1 == 0:
            checkpoint_path = os.path.join(config.model_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step_value)
            print('saved')


if __name__ == "__main__":
    train()
    # 计算模型的Map
    # eval(config.model_dir, yolo_weights = config.yolo3_weights_path)
