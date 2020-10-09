#!/usr/bin/env python

import os
import sys
import time
import select
import saliency
import numpy as np
import resnet_model
import tensorflow as tf
tf.enable_eager_execution()
from IPython import embed
from skimage.io import imsave
from datetime import datetime
import imagenet_input as data_input
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Dataset Configuration
tf.app.flags.DEFINE_string('test_dataset', 'scripts/val.txt', """Path to the test dataset list file""")
tf.app.flags.DEFINE_string('test_image_root', '/data1/common_datasets/imagenet_resized/ILSVRC2012_val/', """Path to the root of ILSVRC2012 test images""")
tf.app.flags.DEFINE_string('mean_path', './ResNet_mean_rgb.pkl', """Path to the imagenet mean""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_test_instance', 50000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "80.0,120.0,160.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")

# Training Configuration
tf.app.flags.DEFINE_string('checkpoint', './alexnet_baseline_2/model.ckpt-399999', """Path to the model checkpoint file""")
tf.app.flags.DEFINE_string('output_file', './alexnet_baseline_2/eval.pkl', """Path to the result pkl file""")
tf.app.flags.DEFINE_string('save_path', './', """Path to the save directory""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of test batches during the evaluation""")
tf.app.flags.DEFINE_integer('class_ind', 895, """Class index to be analyzed""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

FLAGS = tf.app.flags.FLAGS


def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr


def train():
    print('[Dataset Configuration]')
    print('\tImageNet test root: %s' % FLAGS.test_image_root)
    print('\tImageNet test list: %s' % FLAGS.test_dataset)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tCheckpoint file: %s' % FLAGS.checkpoint)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %s' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Evaluation Configuration]')
    print('\tOutput file path: %s' % FLAGS.output_file)
    print('\tTest iterations: %d' % FLAGS.test_iter)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)

    graph = tf.Graph()

    with graph.as_default() as g:
        global_step = tf.Variable(0, trainable=False, name='global_step', 
                dtype=tf.int64)

        # Get images and labels of ImageNet
        print('Load ImageNet dataset')
        with tf.device('/cpu:0'):
            print('\tLoading test data from %s' % FLAGS.test_dataset)
            with tf.variable_scope('test_image'):
                test_images, test_labels = data_input.inputs(FLAGS.test_image_root, FLAGS.test_dataset, FLAGS.batch_size, False, num_threads=1, center_crop=True)

        # Build a Graph that computes the predictions from the inference model.
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        images = tf.placeholder(tf.float32, [FLAGS.batch_size, data_input.IMAGE_HEIGHT, data_input.IMAGE_WIDTH, 3])
        labels = tf.placeholder(tf.int64, [FLAGS.batch_size])

        def build_network():
            network = resnet_model.resnet_v1(
                resnet_depth=50,
                num_classes=1000,
                dropblock_size=None,
                dropblock_keep_probs=[None]*4,
                data_format='channels_last')
            return network(inputs=images, is_training=False)

        logits = build_network()
        sess = tf.Session(
                graph=g,
                config=tf.ConfigProto(
                    allow_soft_placement=True, log_device_placement=True))

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        if FLAGS.checkpoint is not None:
           saver.restore(sess, FLAGS.checkpoint)
           print('Load checkpoint %s' % FLAGS.checkpoint)
        else:
            print('No checkpoint file of basemodel found. Start from the scratch.')

        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)

        # Test!
        test_loss = 0.0
        test_acc = 0.0
        test_time = 0.0
        confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int32)

        path = int(FLAGS.checkpoint.split('-')[1])        
        neuron_selector = tf.placeholder(tf.int32)
        y = logits[0][neuron_selector]
        gradient_saliency = saliency.GradientSaliency(g, sess, y, images)
        os.system('mkdir -p ./{}/weight_{:05d}'.format(FLAGS.save_path, path))
        classified_flag = []
        ground_truth = []
        pred_label = []
        count=0
        for i in range(FLAGS.test_iter):
            test_images_val, test_labels_val = sess.run([test_images[0], test_labels[0]])
            start_time = time.time()
            
            # Evaluate metrics
            # Replace True with "test_labels_val[0] == FLAGS.class_ind" for analyzing the specified
            # class_ind
            if True:
                predictions = np.argmax(logits.eval(session=sess, feed_dict={images: test_images_val}), axis=1)
                ones = np.ones([FLAGS.batch_size])
                zeros = np.zeros([FLAGS.batch_size])
                correct = np.where(np.equal(predictions, test_labels_val), ones, zeros)
                acc = np.mean(correct)
                duration = time.time() - start_time
                test_acc += acc
                test_time += duration
                classified_flag.append([i, acc])
                ground_truth.append(test_labels_val[0])
                pred_label.append(predictions[0])

                # Get gradients
                grad = gradient_saliency.GetMask(test_images_val[0, :], feed_dict = {neuron_selector: test_labels_val[0]})
                
                imsave('./{}/weight_{:05d}/img_{:05d}.jpg'.format(FLAGS.save_path, path, i), (test_images_val[0, :]*imagenet_std + imagenet_mean))
                np.save('./{}/weight_{:05d}/grad_{:05d}.npy'.format(FLAGS.save_path, path, i), np.mean(grad, axis=-1))
                count+=1
                
        test_acc /= FLAGS.test_iter
        np.save('./{}/weight_{:05d}/classified_flag.npy'.format(FLAGS.save_path, int(path)), np.array(classified_flag))
        np.save('./{}/weight_{:05d}/ground_truth.npy'.format(FLAGS.save_path, int(path)), np.array(ground_truth))
        np.save('./{}/weight_{:05d}/pred_label.npy'.format(FLAGS.save_path, int(path)), np.array(pred_label))

        # Print and save results
        sec_per_image = test_time/FLAGS.test_iter/FLAGS.batch_size
        print ('Done! Acc: %.6f, Test time: %.3f sec, %.7f sec/example' % (test_acc, test_time, sec_per_image))
        print ('done!')


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
