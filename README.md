# Fruits360_Classifier
Tensorflow classifier for Fruits360 dataset.

## Getting Started
Required Libraries: Numpy, Pillow, Matplotlib, Tensorflow and Tensorflow-gpu. Python 3.6.5 was used.
#### Downloading Dataset
The dataset can be found at: https://www.kaggle.com/moltean/fruits
#### Processing Data
The repository contains the Fruit_dataset class. This contains functions which can be used to process the dataset one time before training. Run the following python code to process data:
```
from Fruit_dataset import *
new_path = os.path.dirname(__file__)+"\\processed_data"
old_path = os.path.dirname(__file__)+"\\downloaded_data"
f = Fruit_dataset(new_path)
f.process_data(old_path)
```
Where 'old_path' is the file location where the dataset was installed. New_path is the location of the combined files. The data will be read from this location when training or testing. Note: this processing step is different from the pre-processing function called to distort the training data.

## Performance
The training was done on an NVIDIA GEFORCE GTX 1060. It consisted of 400 batches of 64 images each. The amounts to 25,600 images total. During training, images were randomly distorted to increase the effective size of the dataset. The optimization function is AdamOptimizer with learning_rate=1e-4.
The accuracy results for the training session are shown below.

![alt text](https://github.com/JakeSigwart/Fruits360_Classifier/blob/master/train_acc_plot.png)

Accuracy was quite low for the testing dataset.
