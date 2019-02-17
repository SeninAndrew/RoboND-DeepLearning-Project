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

**Convolution networks** is a standard component of image processing networks nowadays. In comparison with fully connected  layers convolutions require less memory due to parameter sharing betwenen input pixels and can be used to classify an object on images regardless of object position. 

**Separable convolutions** require less memory than standard convolutions. That's because in the separable convolutions are applied in 2 steps:
1. On the first step each channel of the input layer is processed by a separate filter. So, number of filters is always equal to number of channels in the input layer. For example, input layer of AxBxC is transformed to DxExC by using a C FxH filters (which requires CxFxH memory for the params). 
2. On the second step we apply G filters of 1x1xC size to the output of the first step. This results in the output of size DxExG (which requires GxC memory for the params). 

In total filters from the first and the second steps require CxFxH + GxC memory which is usually much less than in case of the standard convolutions which require GxCxFxH memory). Less memory for the parameters often results in faster training and less overfitting.

**Batch normalization** is a technique to normalize inputs of each layer by using mean and variance of the current mini-batchd passed through the network. It results in faster training and acts as a regularizer.

**Dropout** is one of the most popular regularization method. When enabled during training, it randomly disables some of connections between 2 consequtive layers. As result the model trains to use all connections in the network and less prone to stuck in local minimums. 

### 1x1 convolution

1x1 convolutions are used to reduce dimensionality while preserving location information. For each pixel in the input layer they calculate a separate output vector. For example, an image of 20x20x50 size might be converter to 20x20x25 by the 1x1 convolution layer if the latter has 25 filters. In a way, it is a form of pooling and might help neurol network to generalize better to new examples.

Interestingly, in my model I found that increasing number of filters in the 1x1 convolution layer (vs the previous layer) results in better segmentation results. So, in my model the 1x1 convolution actually increases dimensionality, rather than decreases. I believe that's because my best model is still relastively small and can benefit from additional parameters.

### Decoder

The decoder components restores information encoded in the previous layers and produces the final segmentation. That's achieved by using the upsampled layer with additional skip links from the previous layers. Precise segmentation can only be achieved by using a mix of local (from the skip layer) and global (from the previous layer) features. 

Number of layers is the same as in the encoder.

### Full model graph

This is Keras visualization of the graph:
![Keras model visualization](https://github.com/SeninAndrew/RoboND-DeepLearning-Project/blob/master/model.png)


## Experiments

