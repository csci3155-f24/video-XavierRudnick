[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/skJdUf3s)
# Principles of Programming and Programming Languages
# Mini-Project

See [instructions.md](instructions.md) for submission instructions.

# CSCI 3155 extra credit project: TensorFlow

## Description

This project covers goggles TensorFlow which is an open-source machine learning framework. TensorFlow supports a wide range of machine learning tasks, including deep learning, natural language processing, and computer vision, with high-level APIs. One of these API’s is keras, which is used for image data processing, and many other applications. 
The framework relates to this class by using functions as values, code as data, and abstract data types. Specifically Tensor flow relies heavily on defining computations as a series of operations. With so many mathematical operations, the framework must be efficient in data usage. The code defines the computation like neural network layers and data preprocessing as data for the goal of optimizing the process. TensorFlow’s Tensor is an example of an abstract data type because it abstracts numerical computations on multi-dimensional arrays.
My project trains a convolutional neural network using TensorFlow and Keras to classify 
lung disease images into four categories: COVID-19, Normal, Pneumonia, and Tuberculosis. The Classifier uses EfficientNetB0 architecture with pre-trained ImageNet weights as the base model, which is fine-tuned on the custom lung disease dataset. I used a pretrained model to speed up my training with medical images. The model is compiled with the Adam optimizer 
and trained with early stopping and learning rate reduction callbacks to prevent overfitting. The model's performance is evaluated with a test dataset by a confusion matrix along with 
a classification report generated to visualize and assess the model's metrics.
	I learned a lot doing this project, in total I’ve rewritten it 4 times. Each time I rewrote it I improved the project which taught me how to feel prideful of a project.




 

## Repository Organization

TODO: I follow the basic outline given in instructions and readme about repository organization. I give a link to my youtube upload and mp4 file on google drive because the video is too big to upload to github. I uploaded my slides and scrpit outline to the repository. I have all the information written down here.

## Building and Testing Instructions

TODO: To use Tensorflow framework, update terminal to have current version of python downloaded. Next pip instal tensorflow. Thats it, tensorflow is very simple to download and use.

## Presentation

TODO: Update the following links and remove this line.

- YouTube: [https://youtu.be/project.](https://youtu.be/bMfQ7vTA9cs)
- Script: (Pushed to repository)
- Recording: https://drive.google.com/file/d/1b8jPpUBAXHKkNFYmHn_SpV9G4Dcj_1qe/view?usp=drive_link
- Slides (Pushed to repository)
