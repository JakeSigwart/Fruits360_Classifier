import os
import numpy as np
import tensorflow as tf
from Fruit_dataset import *
from Tensormodel import *

first_batch_size = 1000

num_batches = 4
batch_size = 100
num_classes = 75

data_path = os.path.dirname(__file__)+'\\processed_data'
model_path = os.path.dirname(__file__)+'\\model\\model.ckpt'
model_checkpoint = os.path.dirname(__file__)+'\\model\\checkpoint'

graph = tf.Graph()
with graph.as_default():
	#Initialize placeholders
	Training_status = tf.placeholder(tf.bool)
	Images = tf.placeholder(tf.float32, shape=[None, 100, 100, 3], name='Images')
	Labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='Labels')
	
	with tf.device("/device:GPU:0"):
		processed_images = pre_process_images(Images, Training_status)
		
		W_conv1 = tf.Variable(tf.truncated_normal([5,5,3,64],mean=0.0,stddev=0.163), name='W_conv1')
		b_conv1 = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv1')
		h_conv1 = tf.nn.conv2d(processed_images, W_conv1, strides=[1,2,2,1], padding='VALID', name='conv1') + b_conv1
		h_actv1 = tf.nn.relu(h_conv1)
		h_pool1 = tf.nn.max_pool(h_actv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		#Shape = [24, 24, 64]
		
		W_conv2 = tf.Variable(tf.truncated_normal([4,4,64,64],mean=0.0,stddev=0.163), name='W_conv2')
		b_conv2 = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv2')
		h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='VALID', name='conv2') + b_conv2
		h_actv2 = tf.nn.relu(h_conv2)
		h_pool2 = tf.nn.max_pool(h_actv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		#shape = [12, 12, 64]
		
		#4-Parrallel Layers: 3A, 3B, 3C, 3D
		W_conv3A = tf.Variable(tf.truncated_normal([3,3,64,64],mean=0.0,stddev=0.025), name='W_conv3A')
		b_conv3A = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv3A')
		h_conv3A = tf.nn.conv2d(h_pool2, W_conv3A, strides=[1,1,1,1], padding='SAME', name='conv3A') + b_conv3A
		h_actv3A = tf.nn.relu(h_conv3A)
		h_pool3A = tf.reshape(tf.nn.max_pool(h_actv3A, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), [-1,6*6,64])
		#Output shape: [6, 6, 64]

		W_conv3B = tf.Variable(tf.truncated_normal([2,2,64,64],mean=0.0,stddev=0.03125), name='W_conv3B')
		b_conv3B = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv3B')
		h_conv3B = tf.nn.conv2d(h_pool2, W_conv3B, strides=[1,1,1,1], padding='SAME', name='conv3B') + b_conv3B
		h_actv3B = tf.nn.relu(h_conv3B)
		h_pool3B = tf.reshape(tf.nn.max_pool(h_actv3B, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), [-1,6*6,64])
		#Output shape: [6, 6, 64]
		
		W_conv3C = tf.Variable(tf.truncated_normal([3,3,64,64],mean=0.0,stddev=0.03125), name='W_conv3C')
		b_conv3C = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv3C')
		h_conv3C = tf.nn.conv2d(h_pool2, W_conv3C, strides=[1,1,1,1], padding='SAME', name='conv3C') + b_conv3C
		h_actv3C = tf.nn.relu(h_conv3C)
		h_pool3C = tf.reshape(tf.nn.avg_pool(h_actv3C, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), [-1,6*6,64])
		#Output shape: [6, 6, 64]
		
		W_conv3D = tf.Variable(tf.truncated_normal([2,2,64,64],mean=0.0,stddev=0.03125), name='W_conv3D')
		b_conv3D = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv3D')
		h_conv3D = tf.nn.conv2d(h_pool2, W_conv3D, strides=[1,1,1,1], padding='SAME', name='conv3D') + b_conv3D
		h_actv3D = tf.nn.sigmoid(h_conv3D)
		h_pool3D = tf.reshape(tf.nn.avg_pool(h_actv3D, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), [-1,6*6,64])
		#Output shape: [6, 6, 64]
		
		h_AB_linked = tf.stack([h_pool3A, h_pool3B], axis=1)
		h_CD_linked = tf.stack([h_pool3C, h_pool3D], axis=1)
		h_linked = tf.reshape(tf.stack([h_AB_linked, h_CD_linked], axis=1), [-1, 4*6*6*64])
		
		W_fc = tf.Variable(tf.truncated_normal([4*6*6*64, 1024],mean=0.0,stddev=0.022))
		b_fc = tf.Variable(tf.constant(0.005, shape=[1024]))
		h_fc = tf.matmul(h_linked, W_fc) + b_fc
		h_fc_actv = tf.nn.relu(h_fc)
		h_fc_norm = tf.layers.batch_normalization(h_fc_actv, axis=1, training=Training_status)

		#Apply Drop-out
		h_fc_dropout = tf.layers.dropout(h_fc_norm, rate=0.5, training=Training_status)
		
		#Read-out layer
		W_read = tf.Variable(tf.truncated_normal([1024,num_classes],mean=0.0,stddev=0.0442))
		b_read = tf.Variable(tf.constant(0.005, shape=[num_classes]))
		y_conv = tf.matmul(h_fc_dropout, W_read) + b_read
		
	#Optimize, calculate accuracy and get the class prediction percentages as a set of normalized vectors
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Labels, logits=y_conv))
	optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	
	predictions = tf.argmax(y_conv, 1)
	y_normalized = tf.nn.softmax(y_conv)
	
	correct_predictions = tf.equal(predictions, tf.argmax(Labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

	#Initialize, save
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	
#Training session
with tf.Session(graph=graph) as sess:
	sess.run(init)
	if os.path.isfile(model_checkpoint):
		saver.restore(sess, model_path)
		print("Model restored.")
	else:
		print('Building new model...')
	#Open dataset
	fruit_data = Fruit_dataset(data_path)
	fruit_data.load_dataset(training_set=False)
	fruit_classes = fruit_data.get_classes()
	
	#One test batch to compute accuracy
	_, labels = fruit_data.get_random_batch(first_batch_size)
	images, hot_labels = fruit_data.format_last_batch()
	acc_output, predictions_out  = sess.run([accuracy, predictions], feed_dict={Images:images, Labels:hot_labels, Training_status:False})		
	print('Accuracy: '+str(acc_output))
	
	#TRAINING LOOP
	for n in range(0, num_batches):
		_, labels = fruit_data.get_random_batch(batch_size)
		images, hot_labels = fruit_data.format_last_batch()
		text_labels = fruit_data.lookup_labels(labels)
		
		#Run session
		acc_output, predictions_out  = sess.run([accuracy, predictions], feed_dict={Images:images, Labels:hot_labels, Training_status:False})		
		print('Batch: '+str(n)+ '  Accuracy: '+str(acc_output))
		print('Prediction: '+fruit_classes[predictions_out[0]]+'  Actual: '+text_labels[0])
		plot_image(images[0])
		



