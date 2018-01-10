# Semantic Segmentation
### Introduction
In this project, I'll label the pixels of a road in images using a [Fully Convolutional Network (FCN)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). The components of a FCN includes a pretrained neural network (ie. VGG-16) as an encoder and transpose convolution layers as a decoder. The role of the encoder is capturing the features present at different depths (layer 3, 4, 7), while the decoder adds the upsampled (transposed) final layer 7 with the skip connections produced by putting layers 3 and 4 through 1x1 convolutions. Helpful animations of convolutional operations, including transposed convolutions, can be found [here](https://github.com/vdumoulin/conv_arithmetic). The final sum has the same size of the original image and it is used to predict whether each pixel belongs to a labeled class.

[architecture1]: ./images/architecture1.png
[bike]: ./images/bike.png

Note: There are two options for transfer learning in semantic segmentation, depending on whether the pretrained network parameters are going to be re-trained by the new data. If we want to fix the pretrained network, we need to freeze the backpropagation at all skip connections (layer 3, 4 and 7 of VGG).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Goal

The project labels most pixels of roads close to the best solution. The model doesn't have to predict correctly all the images, just most of them. A solution that is close to best would label at least 80% of the road and label no more than 20% of non-road pixels as road.

### Implementation

The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip). This model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. I extract them by name.

```python
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()
    
    # get the layers of vgg to use in Fully Convolutional Network FCN-8s
    input_ = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')
```

The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 

```python
layer_3 = tf.multiply(layer_3, 0.0001)
layer_4 = tf.multiply(layer_4, 0.01)
```    

When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
