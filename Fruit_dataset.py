import os
import numpy as np
import random
from PIL import Image
import pickle
import matplotlib.pyplot as plt


#Open fruit dataset files. Get random batches
class Fruit_dataset:
	#Input: path to processed data folder, dataset to get batches from
	def __init__(self, processed_path):
		self.processed_path = processed_path
	
	#Input: Path to original Training and Test data
	#Processing: 
	#Output: numpy files for each fruit image set in Training and Test forlders
	def process_data(self, original_path):
		training_path = os.path.join(original_path, "Training")
		test_path = os.path.join(original_path, "Test")
		
		#Get classes from file names
		self.fruit_classes = os.listdir(training_path)
		self.num_classes = len(self.fruit_classes)
		with open(self.processed_path+"\classes.pickle", 'wb') as a:
			pickle.dump(self.fruit_classes, a)
		training_sizes = []
		test_sizes = []
		
		#Iterate through training set. Stack images for each fruit and save to new processed path
		for n in range(0, self.num_classes):
			fruit_name = self.fruit_classes[n]
			fruit_path = os.path.join(training_path, fruit_name)
			image_files = os.listdir(fruit_path)
			fruit_data_array = []
			for image_file in image_files:
				image_path = os.path.join(fruit_path, image_file)
				im_image = Image.open(image_path)
				np_image = np.array(im_image.getdata()).reshape(im_image.size[0], im_image.size[1], 3)
				fruit_data_array.append(np_image)
			np_fruit_data_array = np.array(fruit_data_array, dtype='uint8')
			training_sizes.append(np_fruit_data_array.shape[0])
			np.save(self.processed_path+"\Training\class_"+str(n)+".npy", np_fruit_data_array)
			print("Saved Training: "+self.fruit_classes[n])
			if n==0:
				test = np.load(self.processed_path+"\Training\class_"+str(n)+".npy")
				test_image = test[0]
				plot_image(test_image)
			
		#Iterate through test set. Stack images for each fruit and save to new processed path
		for n in range(0, self.num_classes):
			fruit_name = self.fruit_classes[n]
			fruit_path = os.path.join(test_path, fruit_name)
			image_files = os.listdir(fruit_path)
			fruit_data_array = []
			for image_file in image_files:
				image_path = os.path.join(fruit_path, image_file)
				im_image = Image.open(image_path)
				np_image = np.array(im_image.getdata()).reshape(im_image.size[0], im_image.size[1], 3)
				fruit_data_array.append(np_image)
			np_fruit_data_array = np.array(fruit_data_array, dtype='uint8')
			test_sizes.append(np_fruit_data_array.shape[0])
			np.save(self.processed_path+"\Test\class_"+str(n)+".npy", np_fruit_data_array)
			print("Saved Test: "+self.fruit_classes[n])
			if n==0:
				test = np.load(self.processed_path+"\Test\class_"+str(n)+".npy")
				test_image = test[0]
				plot_image(test_image)
				
		#Save the sizes of the datasets
		with open(self.processed_path+"\Training\size.pickle", 'wb') as b:
			pickle.dump(training_sizes, b)
		with open(self.processed_path+"\Test\size.pickle", 'wb') as c:
			pickle.dump(test_sizes, c)
		
	
	#Input: training set (True -> training, False -> test)
	#Processing: Load classes. load sizes
	#Output: batch of images, class_labels
	def load_dataset(self, training_set=True):
		self.training_set = training_set
		if self.training_set:
			self.data_path = os.path.join(self.processed_path, "Training")
		else:
			self.data_path = os.path.join(self.processed_path, "Test")
		
		
		if os.path.isfile(self.processed_path+"\classes.pickle"):
			with open(self.processed_path+"\classes.pickle", 'rb') as f:
				fruit_classes = pickle.load(f)
		else:
			print("ERROR: Processed data not found.")
			fruit_classes = []
		
		if os.path.isfile(self.data_path+"\size.pickle"):
			with open(self.data_path+"\size.pickle", 'rb') as f:
				size = pickle.load(f)
		else:
			print("ERROR: Processed data not found.")
			size = []
		
		self.fruit_classes = fruit_classes
		self.size = size
		
	
	#Input: number of random images
	#Processing: 
	#Output: 
	def get_random_batch(self, batch_size):
		self.num_classes = len(self.fruit_classes)
		
		class_labels = []
		img_labels = []
		for n in range(0, batch_size):
			rand_num = random.randint(0, self.num_classes-1)
			class_labels.append(rand_num)
			max_imgs = self.size[rand_num]
			rand_img = random.randint(0, max_imgs-1)
			img_labels.append(rand_img)
		
		image_batch = []
		for c in range(0, batch_size):
			class_images = np.load(self.data_path+"\class_"+str(class_labels[c])+".npy")
			image = class_images[img_labels[c]]
			image_batch.append(image)
		
		batch = np.array(image_batch, dtype='uint8')
		labels = np.array(class_labels, dtype=np.int32)
		self.batch = batch
		self.labels = labels
		return batch, labels
	
	
	#Normalize images and convert labels to one-hot
	def format_last_batch(self):
		images = normalize_images(self.batch)
		hot_labels = one_hot_encode(self.labels, self.num_classes)
		return images, hot_labels
		
		
	#Convert integer labels to string labels
	def lookup_labels(self, labels):
		output = []
		for n in range(0, len(labels)):
			output.append(self.fruit_classes[labels[n]])
		return output
		
	#
	def display_image(self, fruit_num, img_num=0):
		class_imgs = np.load(self.data_path+"\class_"+str(fruit_num)+".npy")
		image = class_imgs[img_num]
		plot_image(image)
		
	def get_classes(self):
		return self.fruit_classes
		
		
	
#Display array as a color image
def plot_image(img_data):
    plt.axis('off')
    plt.imshow(img_data)
    plt.show()
	
	
#
def one_hot_encode(labels, num_classes):
	output = np.zeros([num_classes], dtype=np.float32)
	if isinstance(labels, int):
		output = np.zeros([1,num_classes], dtype=np.float32)
		output[0,labels] = 1.0
	elif isinstance(labels, list):
		output = np.zeros([len(list),num_classes], dtype=np.float32)
		for n in range(0, len(labels)):
			output[n,labels[n]] = 1.0
		
	elif 'numpy' in str(type(labels)):
		if labels.ndim==1:
			output = np.zeros([labels.shape[0],num_classes], dtype=np.float32)
		else:
			labels = labels.flatten()
			output = np.zeros([labels.shape[0],num_classes], dtype=np.float32)
		for n in range(0, labels.shape[0]):
			output[n,labels[n]] = 1.0
			
	else:
		print("ERROR: Wrong type passed to: one_hot_encode.")
		output = []
	return output

	
#
def one_hot_decode(hot_labels):
	if hot_labels.ndim==1:
		output = np.argmax(hot_labels, axis=0)
	elif hot_labels.ndim==2:
		output = np.argmax(hot_labels, axis=1)
	else:
		output = []
		print("ERROR: Input to: one_hot_decode has wrong number of dimentions.")
	return output

	
#
def normalize_images(images, span=1.0, min_val=0.0):
    images_out = span*(np.array(images, dtype=np.float32) / 255.0)  + min_val
    if images.ndim==3:
        images_out = np.reshape(images_out, (1, images.shape[0], images.shape[1], images.shape[2]) )
    return images_out

	
	
