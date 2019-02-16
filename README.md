## Follow me Project Overview ##

In this project I implemented a segmentation deep neural network to detect the 'hero' (a lady in the red dress). This network is used to navigate the drone in street evironment. The network performs segmentation of 3 types of objects: background, people, the hero. 

## Data

As training data I was using the images provided by Udacity: [train.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip),  [validation.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip), and [sample_evaluation_data.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip).


## Network overview

The network we use consists of the following main components: input, encoder, 1x1 convolution, decoder, output.

### Input

The original images were resized to 160x160 with 3 color components. As result, the images are stored in <Batch size>x160x160x3 tensors.
  
### Encoder

Conceptually, this layer encodes high level information about what's on the image in a short vector representaton. Each point in the encoder layer output was calculated based on inputs from different image regions. That will help us to produce more stable segmentation results as we can not reliably classify a pixel based on its local neighborhood  only. 

### Full model graph


## Experiments


