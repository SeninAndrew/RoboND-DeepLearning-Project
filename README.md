## Follow me Project Overview ##

In this project I implemented a segmentation deep neural network to detect the 'hero' (a lady in the red dress). This network is used to navigate the drone in street environment. The network performs segmentation of 3 types of objects: background, people, the hero. 

## Data

As training data I was using the images provided by Udacity: [train.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip),  [validation.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip), and [sample_evaluation_data.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip).

## Network overview

The network we use consists of the following main components: input, encoder, 1x1 convolution, decoder, output.

### Input

The original images were resized to 160x160 with 3 color components. As result, the images are stored in <Batch size>x160x160x3 tensors.
  
### Encoder

Conceptually, this layer encodes high level information about what's on the image in a short vector representation. Each value in the encoder layer output was calculated based on inputs from different image regions. That will help us to produce more stable segmentation results as we can not reliably classify a pixel based on its local neighborhood only. 

In our specific model the encoder layer consists of 4 separable 2d convolution layers with batch normalization and dropout, each layer with 32 filters.

**Convolution networks** is a standard component of image processing networks nowadays. In comparison with fully connected  layers convolutions require less memory due to parameter sharing between input pixels and can be used to classify an object on images regardless of object position. 

**Separable convolutions** require less memory than standard convolutions. That's because in the separable convolutions are applied in 2 steps:
1. On the first step each channel of the input layer is processed by a separate filter. So, number of filters is always equal to number of channels in the input layer. For example, input layer of AxBxC is transformed to DxExC by using a C FxH filters (which requires CxFxH memory for the params). 
2. On the second step we apply G filters of 1x1xC size to the output of the first step. This results in the output of size DxExG (which requires GxC memory for the params). 

In total filters from the first and the second steps require CxFxH + GxC memory which is usually much less than in case of the standard convolutions which require GxCxFxH memory). Less memory for the parameters often results in faster training and less overfitting.

**Batch normalization** is a technique to normalize inputs of each layer by using mean and variance of the current mini-batch passed through the network. It results in faster training and acts as a regularizer.

**Dropout** is one of the most popular regularization method. When enabled during training, it randomly disables some of connections between 2 consecutive layers. As result the model trains to use all connections in the network and less prone to stuck in local minimums. 

### 1x1 convolution

1x1 convolutions are used to reduce dimensionality while preserving location information (as opposite to the fully connected layer). For each pixel in the input layer they calculate a separate output vector. For example, an image of 20x20x50 size might be converter to 20x20x25 by the 1x1 convolution layer if the latter has 25 filters. In a way, it is a form of pooling and might help neural network to generalize better to new examples.

Interestingly, in my model I found that increasing number of filters in the 1x1 convolution layer (vs the previous layer) results in better segmentation results. So, in my model the 1x1 convolution actually increases dimensionality, rather than decreases. I believe that's because my best model is still relatively small and can benefit from additional parameters.

### Decoder

The decoder components restores information encoded in the previous layers and produces the final segmentation. That's achieved by using the upsampled layer with additional skip links from the previous layers. Precise segmentation can only be achieved by using a mix of local (from the skip layer) and global (from the previous layer) features. 

Number of layers is the same as in the encoder.

### Hyperparameters

There are 6 parameters we tune during the training:
- Learning rate. This parameter defines how fast the algorithm updates the trainable parameters based on calculated gradients on the next iteration of the learning process. In most cases it is advices to change learning rate during training (the later training stage is the lower learning rate is recommended to use). This can be achieved by setting the decay parameter for the Adam optimizer we use in this notebook. However, in this task we use a single learning rate over the course of training (since it was required not to modify content of the cell with the optimizer settings).

- Batch size. This parameter defines size of the training batch - number of training examples used to calculate gradient update per iteration. We can not use all the examples on every iteration as it would require too much of memory and computations. Also, using of all images will likely result in a worse model since the update would be less "stochastic" and will likely result in the same local minima every run.

- Num epochs. Number of times all images in the training dataset are visited before we stop training. The longer we train the better results on the training set we usually end up with. However, longer training might result in overfitting and lowering result on the validation and test sets. So, in most cases finishing training early (10-50 epochs) will result in better results on the test set. 

- Steps per epoch. Number of steps in an epoch. It can be estimated as <Number of training examples> / <Batch size>. In case we set a higher value of steps it results in certain images in a train set being used multiple times per epoch (which is similar to increasing number of epochs).

- Validation steps. Total number of steps (batches of samples) to yield from validation_data generator before stopping at the end of every epoch. In our case we have 4x times less validation images than the training ones. So, number of validation steps is about 4 times less than number of train steps.

- Workers. Maximum number of processes to spin up when using process-based threading. 

After many iterations of manual tuning I ended up with the following hyperparameters:
learning_rate = 0.001
batch_size = 40
num_epochs = 50
steps_per_epoch =200
validation_steps = 50
workers = 2

In my experience the learning parameter was the most important for achieving the target performance results.

## Other scenarios
If we want to apply segmentation to other classes of images (such dogs, cats or cars), we would have to collect training data (which includes images and ground truth masks). We can start with the same model and just tune the hyperparameters. It might work as well as for the "Hero" scenario as the basic principle remains the same. But as it usually happens in real work scenario tuning of the model (adding more layers and tuning number of filter) will likely help to improve results as well.

### Full model graph

This is Keras visualization of the graph:
![Keras model visualization](https://github.com/SeninAndrew/RoboND-DeepLearning-Project/blob/master/model.png)

This model gives the final score of 0.4254. You can download the corresponding [model](https://github.com/SeninAndrew/RoboND-DeepLearning-Project/blob/master/model/model_weights_v32) and [config](https://github.com/SeninAndrew/RoboND-DeepLearning-Project/blob/master/model/config_model_weights_v32) files.

Unfortunately, I forgot to save a copy of the notebook for that training run before running out of GPU quota. So, here is [html of the notebook](https://github.com/SeninAndrew/RoboND-DeepLearning-Project/blob/master/code/model_training.html) with another run which gives 0.405 of the final score. In that run I used a 2 layer encoder with 64 and 128 filters, no dropout, 1x1 convolutions with 256 filters. You can also download the corresponding [model](https://github.com/SeninAndrew/RoboND-DeepLearning-Project/blob/master/model/model_weights_v39) and [config](https://github.com/SeninAndrew/RoboND-DeepLearning-Project/blob/master/model/config_model_weights_v39) files.
