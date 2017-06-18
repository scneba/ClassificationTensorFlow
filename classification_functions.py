import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import math
import random
from sklearn.utils import shuffle


def normalise_images(batch_images, min=0,max=1):
    """Normalise a batch of images between min and max value
    Args:
        batch_images:batch of images size(None,None,None,3)
        min: minimum value
        max: maximum value
    Returns: ndarray of normalised images
    
    """
    std = (batch_images- batch_images.min(axis=0)) / (batch_images.max(axis=0) - batch_images.min(axis=0))
    return std * (max - min) + min

def normalise(data):
    """Normalise an image to scale of between 0 and 1
    Args: 
        image: image to normalise
    Returns: ndarray of normalised image
    """
    return data/255.0

def gray_scale(image):
    """Convert and image to grayscale
    Arg:
        image: image to be converted to grayscale
    Returns: ndarray of grayscaled image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_blur(image_rgb,kernel=(5,5)):
    """Apply gaussian blur kernel to image
    Args:
        img_rgb: image converted to rgb
        kernel: tuple of kernel to be applied
    
    """
    return cv2.GaussianBlur(image_rgb,kernel,0).astype(np.uint8)

def rotate(img_rgb,angle=8,scale=1):
   
    """Rotate an image.
    Rotate an image with angle and scale
    
    Args:
        img_rgb: image converted to rgb
        angle: rotation angle in degrees
        scale: scale of image
    Return: Rotated image
    """
    rows,cols = img_rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle,scale)
    return cv2.warpAffine(img_rgb,matrix,(cols,rows)).astype(np.uint8)

def translate(img_rgb,x=2,y=2):
    """Translate an Image.
    Translate an rgb image by x and y pixels
    Args:
        img_rgb: image converted to rgb 
        x: translation pixels in x direction of image
        y: tranlation pixes in y direction of image
    Returns: translated image
        
    """
    rows,cols = img_rgb.shape[:2]
    M = np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(img_rgb,M,(cols,rows)).astype(np.uint8)

def affine_transform(image_rgb, src= np.float32([[10,10],[10,150],[100,100]]), dst= np.float32([[12,12],[0,150],[102,102]])):
    """Perform an affine transform on image with source and destination points
    Args:
        img_rgb: image converted to rgb
        src: ndarray of source points (3,2)
        dst: ndarray of destination points (3,2)
    Returns:
        image with perspective tarnsform applied
    """
    rows,cols = image_rgb.shape[:2]
    M = cv2.getAffineTransform(src,dst)

    return cv2.warpAffine(image_rgb,M,(cols,rows)).astype(np.uint8)




def unpickle(file):
    """Extract dictionary data from file.
    Args:
        file: The file to unpickle
    
    Returns:
        The extracted data.
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cf10_data(base_path,type="train"):
    """Load cf10 data from the base path given
    Args:
        base_path: the path to the directory where the batches are stored
        type: string of type of data: train or test. 
    Returns: Numpy array of all  the data size(None,32,32,3)
             Numpy array of the labels size (None,1)
            
    """
    data= np.empty((0,32,32,3)).astype(np.uint8)
    labels = np.empty((0))
    if type=="train":
        for i in range(5):   
            batch_dict = unpickle(base_path+"data_batch_"+str(i+1))
            batch_bin = batch_dict[b'data']
            batch_reshape = np.reshape(batch_bin,(-1,3,32,32))
            batch_rgb = batch_reshape.transpose([0,2,3,1])
            data = np.vstack((data,batch_rgb))

            labels_dict = batch_dict[b'labels']
            labels_np = np.array(labels_dict)
            labels = np.hstack((labels,labels_np))             
    elif type=="test":  
        batch_dict = unpickle(base_path+"test_batch")
        batch_bin = batch_dict[b'data']
        batch_reshape = np.reshape(batch_bin,(-1,3,32,32))
        batch_rgb = batch_reshape.transpose([0,2,3,1])
        data = np.vstack((data,batch_rgb))

        labels_dict = batch_dict[b'labels']
        labels = np.array(labels_dict)
    else: 
        raise ValueError("The value of type should be \"train\" or \"test\"")
    
    return data, labels



def visualize_cf10(cf10_data, cf10_labels,cmap=None):
    """Visualize first ten images for all categories
    Args: 
        cf10_data: ndarray of cf10_data
        cf10_labels: ndarray of cf10_labels
    
    """
    images = np.empty((0,32,32,3))
    for i in range(10):
        first_ten_indices = np.where(cf10_labels==i)[0]
        to_np_array = np.array(first_ten_indices)
        random_10 = np.random.choice(to_np_array, size=10)
        first_ten_images = cf10_data[random_10.transpose()]
        images = np.vstack((images,first_ten_images))
        
    indices = np.linspace(0,99,100).astype(int)
    fig, axes = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(10,10),)
    for i, axis in zip(indices, axes.flatten()):
        axis.imshow(images[i]/255.0, aspect='equal')    
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig
    

def plot_cf10_data(train_labels, test_labels=None):
    labels = np.array(["Airplanes","Automobiles","Birds","Cats","Deers","Dogs","Frogs","Horses","Ships","Trucks"])
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    width = 0.8
    fig = plt.figure()
    ind = np.arange(len(unique_train))
    p1 = plt.bar(ind, counts_train, width, color='red')
    global p2
    if(test_labels!=None):
        p2 = plt.bar(ind, counts_test, width,color='blue')
    plt.ylabel('Frequency')
    plt.xlabel('Category')
    plt.title('Bar Chart of Training and/or Test Data')
    plt.xticks(ind, labels)
    if(test_labels!=None):
        plt.legend((p1[0], p2[0]), ('Train', 'Test'))
        
    fig.autofmt_xdate(bottom=0.2, rotation=80, ha='right')
    return fig



def augment_data(data,labels,fraction):
    """Augment data with the three augmentation methods.
    Args:
        data: ndarray of data to augment.
        labels: ndarray of labels.
        fraction: the amount of data to be added to produced as a fraction of the `data` argument. 
    Returns: 
        ndarray of new data
        ndarray of new labels
    """
    new_data = np.empty((0,32,32,3)).astype(np.uint8)
    new_labels = np.empty((0))
    for i in range(0, len(data), fraction):
        index = np.random.choice(3)
        global new_image
        if(index==1):
            new_image = rotate(data[i])
        elif(index==2):
            new_image = translate(data[i])
        else:
            new_image = affine_transform(data[i])
        
        reshaped = np.reshape(new_image,(1,32,32,3))
        new_data = np.vstack((new_data,reshaped))
        new_labels = np.hstack((new_labels,labels[i]))
    return new_data, new_labels



def plot_color_dist(data):
    """Plot color distribution of data
    Args:
        data: the data to be plotted
    """
    figure, axes = plt.subplots(nrows=3, ncols=1)
    colors = ["red","green", "blue"]
    for i,axis,color in zip(range(3),axes.flatten(),colors):
        channel = data[:,:,:,i].ravel()
        axis.hist(channel, 256, histtype='bar', color=color, label=color)
        axis.legend(prop={'size': 10})
        axis.set_title('color distribution, channel {}'.format(i))

    figure.tight_layout()
    return figure
    
def preprocess_data(data):
    """Final pipeline for image preprocessing.
    Args:
        data: ndarray of input data
    Returns: ndarray of preprocessed data
    """
    new = np.empty((0,32,32,1))
    for i in range(len(data)):
        gaussian_blu = gaussian_blur(data[i]).astype(np.uint8)
        gray = cv2.cvtColor(gaussian_blu, cv2.COLOR_RGB2GRAY)
        normalised = normalise(gray)
        reshaped = np.reshape(normalised,(1,32,32,1))
        new = np.vstack((new,reshaped))
    return new



def image_generator(data, labels, batch_size=128):
    """Obtains a batch of  images from a list of image paths
    Args: 
        all_image_path: path to all input images
        batch_size: batch size to generate
        min: minimum normalisaiton value
        max: maximum normalisation value
        output_size: tuple of desired image output size 
        
    """
    num_samples = len(data)
    X_data, y_data = shuffle(data, labels)
    for offset in range(0, num_samples, batch_size):
        batch_images = preprocess_data(X_data[offset:offset+batch_size])
        batch_labels = y_data[offset:offset+batch_size]
        yield batch_images, batch_labels



def train_model(data,reuse=False, dropout_rate=0.5):
    
    """Model for training data
    Args: 
        data: Tensor of input data to model
        reuse: Boolean of variable reuse
        dropout_rate: dropout rate for dropout layers
    Returns: logits
    """
    with tf.variable_scope('ns_train5', reuse=reuse):
        # input shape will depend on image shapes supplied
        
        conv1_1 = tf.layers.conv2d(inputs=data,filters=32,kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu, name="conv1_1")
        conv1_2 = tf.layers.conv2d(inputs=conv1_1,filters=32,kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu,name="conv1_2")
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=(2,2), strides=(1,1), name="pool1")
        
        
        
        #want to keep as many features as possible while limiting parameter size
        conv2_1 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu, name ="conv2_1")
        conv2_2 = tf.layers.conv2d(inputs=conv2_1,filters=64,kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu, name ="conv2_2")
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=(2,2), strides=(1,1), name="pool2")
        
        
        
        
        
        conv3_1 = tf.layers.conv2d(inputs=pool2,filters=128,kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu, name="conv3_1")
        
        conv3_2 = tf.layers.conv2d(inputs=conv3_1,filters=128,kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu, name="conv3_2")
            
        pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=(2,2), strides=(1,1), name="pool3")

        
        
        
        
        conv4_1 = tf.layers.conv2d(inputs=pool3,filters=256,kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu, name="conv4_1")
        
        conv4_2 = tf.layers.conv2d(inputs=conv4_1,filters=256,kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu, name="conv4_2")
            
        pool4 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=(2,2), strides=(1,1), name="pool4")
            
        
        #flatten image data for each image
        flatten_layer = tf.contrib.layers.flatten(pool4)
        #connected layers
        connected1 = tf.layers.dense(flatten_layer, 1000,name="dens1")
        dropout1 = tf.layers.dropout(connected1,rate=dropout_rate)
        activation1 = tf.nn.relu(dropout1)
        
        
        connected2 = tf.layers.dense(activation1, 500,name="dens2")
        dropout2 = tf.layers.dropout(connected2,rate=dropout_rate)
        activation2 = tf.nn.relu(dropout2)
        
        
        connected3 = tf.layers.dense(activation2, 200, name="dens3")
        dropout3 = tf.layers.dropout(connected3,rate=dropout_rate)
        activation3 = tf.nn.relu(dropout3)
        
        #return  logits of the prediction
        logits = tf.layers.dense(activation3, 10, name="dens4")
        return logits
    

                          
    