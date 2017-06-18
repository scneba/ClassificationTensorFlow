
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import math
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
import argparse
from classification_functions import *


def load_preprocess_and_visualize(base_path,fraction):
    #Load data
    train_data,train_labels = load_cf10_data(base_path, type="train")
    test_data, test_labels =  load_cf10_data(base_path, type="test")
    assert(len(train_data) == len(train_labels))
    assert(len(test_data) == len(test_labels))
    print("Train data shape: {}".format(train_data.shape))
    print("Test data shape: {} ".format(test_data.shape))
    #Visualize data
    
    print("Visualizing data......")
    fig = visualize_cf10(train_data,train_labels)
    fig.suptitle('Samples of cfar10 dataset', fontsize=20)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")

    #Plot data
    print("Plotting histogram..........")
    fig =  plot_cf10_data(train_labels,test_labels)
    fig.suptitle('Bar Chart of Train and Test Data', fontsize=20)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    


    ### Augment training data and display augmented data
    print("Running data augmentation........")
    data,labels = augment_data(train_data,train_labels,fraction)
    train_data = np.vstack((train_data, data))
    train_labels = np.hstack((train_labels,labels))
    assert(len(train_data) == len(train_labels))
    print("Augmented Train data shape: {}".format(train_data.shape))
    print("Augemented Train labels  shape: {}".format(train_labels.shape))

    print("Visualizing  augmented data")
    fig = visualize_cf10(train_data[50000:], train_labels[50000:])
    fig.suptitle('Samples of Augmented Data', fontsize=20)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    
    
    print("Generating Bar Chart of Augmented Data")
    fig = plot_cf10_data(train_labels[50000:])
    fig.suptitle('Bar Chart of Augmented Data', fontsize=20)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")

    #Plot colour distribution of all the data

    print("Plotting color distribution of Train data")
    fig = plot_color_dist(train_data)
    fig.suptitle('Color Histogram of Training Data', fontsize=20)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    
    print("Plotting color distribution of Test data")
    fig = plot_color_dist(test_data)
    fig.suptitle('Color Histogram of Test Data', fontsize=20)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")
  

    return train_data, train_labels, test_data, test_labels
   


def evaluate(accuracy_operation,x,y,X_data, y_data,batch_size):
    """Evaluate accuracy of training
    Args:
        accuracy_operation: Accuracy operation
        x: TF Placeholder for x data
        y: TF Placeholder for y data
        X_data: ndarray of training X data
        y_data: ndarray of labels
    Returns: Accuracy
    """
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    generator = image_generator(X_data,y_data,batch_size)
    for offset in range(math.floor(num_examples/batch_size)):
        batch_x, batch_y = next(generator)
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



def train(train_data, train_labels,epochs,batch_size,cross_entropy_count, validation_count):
    
    x = tf.placeholder(tf.float32, (None, 32, 32,1))
    y = tf.placeholder(tf.int32, (None))
    dropout_rate = tf.placeholder(tf.float32,[], (None))
    one_hot_y = tf.one_hot(y, 10)

    learning_rate = 0.001
    dropout = 0.5
    
    #logits, cross entropy loss and optimization functions
    logits = train_model(x,dropout_rate=dropout_rate)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(loss_operation)


    correct_prediction = tf.equal(tf.argmax(logits, 1, name="logits"), tf.argmax(one_hot_y, 1, name="labels"))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.1, random_state=42)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        count=1
        for i in range(epochs):
            generator = image_generator(X_train,y_train,batch_size)
            print("Epoch {}/{}".format(i+1,epochs))

            for offset in range(math.floor(num_examples/batch_size)):
                batch_x, batch_y = next(generator)
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, dropout_rate: dropout})
                if(count%cross_entropy_count==0):
                    cross_entropy_loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y, dropout_rate: dropout}) 
                    print("Cross Entropy loss count {}: {}".format(count,cross_entropy_loss))
                count+=1
                if(count%validation_count==0):
                    validation_accuracy = evaluate(accuracy_operation,x,y,X_valid, y_valid,batch_size)
                    print("EPOCH {} ...".format(i+1))
                    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                    print()




        saver.save(sess, './mynetwork')
        print("Model saved")

def test_accuracy(test_data, test_labels):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        test_accuracy = evaluate(test_data, test_labels)
        print("Test Accuracy = {:.3f}".format(test_accuracy))


        

def main():
    parser = argparse.ArgumentParser(description='Run a training of cfar10 dataset.')
    parser.add_argument(
        'images_path',
        type=str,
        help='Path to folder containing extracted cfar10 batches and labels for instance "cfar10/"'
    )
    parser.add_argument(
        '--b',
        type=int,
        default=128,
        help='Batch size: default=128.')
    parser.add_argument(
        '--e',
        type=int,
        default=10,
        help='Number of Epochs: default=10.')
    
    parser.add_argument(
        '--c',
        type=int,
        default=10,
        help='Number of Steps to run before showing Cross Entropy Loss: default=10')
        
        
    parser.add_argument(
        '--v',
        type=int,
        default=100,
        help='Number of Steps to run before showing validation loss: default=100')
    
    parser.add_argument(
        '--f',
        type=int,
        default=5,
        help='Fraction of total dataset to add through data augmentation: default=5')
        
        
        
    args = parser.parse_args()

    #run algorithm
    train_data, train_labels, test_data, test_labels = load_preprocess_and_visualize(args.images_path,args.f)
    train(train_data, train_labels, args.e, args.b, args.c, args.v)
    test_accuracy(test_data, test_labels)


if __name__ == '__main__':
    main()





