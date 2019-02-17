## Follow me Project Overview ##

In this project I implemented a segmentation deep neural network to detect the 'hero' (a lady in the red dress). This network is used to navigate the drone in street evironment. The network performs segmentation of 3 types of objects: background, people, the hero. 

## Data

As training data I was using the images provided by Udacity: [train.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip),  [validation.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip), and [sample_evaluation_data.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip).


## Network overview

The network we use consists of the following main components: input, encoder, 1x1 convolution, decoder, output.

### Input

The original images were resized to 160x160 with 3 color components. As result, the images are stored in <Batch size>x160x160x3 tensors.
  
### Encoder

Conceptually, this layer encodes high level information about what's on the image in a short vector representaton. Each value in the encoder layer output was calculated based on inputs from different image regions. That will help us to produce more stable segmentation results as we can not reliably classify a pixel based on its local neighborhood only. 

In our specific model the encoder layer consists of 4 separable 2d convolution layers with batch normalization and dropout, each layer with 32 filters.

Convolution networks is a standard component of image processing networks nowadays. In comparison with fully connected  layers convolutions require less memory due to parameter sharing betwenen input pixels and can be used to classify an object on images regardless of object position. 

Separable convolutions require less memory than standard convolutions. That's because in the separable convolutions are applied in 2 steps:
1. On the first step each channel of the input layer is processed by a separate filter. So, number of filters is always equal to number of channels in the input layer. For example, input layer of AxBxC is transformed to DxExC by using a C FxH filters (which requires CxFxH memory for the params). 
2. On the second step we apply G filters of 1x1xC size to the output of the first step. This results in the output of size DxExG (which requires GxC memory for the params). 

In total filters from the first and the second steps require CxFxH + GxC memory which is usually much less than in case of the standard convolutions which require GxCcFxH memory). Less memory for the parameters often results in faster training and less overfitting.

### Full model graph

This is Keras visualization of the graph:
model.png


## Experiments


a
