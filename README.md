# Image Based Relighting Using Neural Networks
<img style="float: center;" src=./documents/waldorf_example.png>
This is a simple implementation of image based relighting using deep networks.

The goal is to predict how a single scene looks under different lighting conditions given a few sample lighting conditions.  We have a sparse training set (ideally, 200+ images) with no augmentation and want to predict the scene under unseen lighting conditions.

This work is inspired by Ren et al. [Image Based Relighting Using Neural Networks](https://www.microsoft.com/en-us/research/video/image-based-relighting-using-neural-networks-2/). When we tried to implement a version of their system as a baseline, we found that that it wasn't easy to reproduce. Their hierarchical divide and conquer method of breaking the learning problem down into small digestible chunks, the ensemble reconstruction model, and the distributed learning framework was a challenge to reproduce when you don't have a computer cluster for training.  To their credit, they probably didn't have fast GPUs at the time.

Xu et al. [Deep Image-Based Relighting from Optimal Sparse Samples](https://dl.acm.org/citation.cfm?doid=3197517.3201313) is a great approach taking the idea much further utilizing synthetic data.


The main idea behind this work is that we can implement a deep network with approximately the same number of parameters as Ren et al., using a simpler structure, and be able train the model quickly on a single machine.  Initially we tried the Ren et al approach on architectual images, and it did not work for a variety of reasons.  [Computing Long-term Daylighting Simulations from High Dynamic Range Imagery Using Deep Neural Networks](https://www.ashrae.org/File%20Library/Conferences/Specialty%20Conferences/2018%20Building%20Performance%20Analysis%20Conference%20and%20SimBuild/Papers/C018.pdf) details some of the reasons and provides interesting solutions.

## Network structure
<img style="float: center;" src=./documents/network_structure.png>

The network structure in this example is simply several Dense layers with relu activations.  The input to the network, like Ren et al. is a vector consisting of the average image (this could be several salient images as in other work, both work just fine), each pixel position, and the light position in normalized coordinates.  The output is the predicted RGB pixel color.

The key to success of this approach is due to three factors: 
1. Loss functions
2. Data normalizaton
3. Data sampling during training  

### Loss
Normaly one might just use an MSE loss on the RGB image, and call it a day.  In reality, this tends to cause problems where one channel gets stuck in a local minima resulting in zero-value predictions, resulting in an entirely black prediction channel.  This problem can be solved by using the natural cross-correlation betweeen RGB channels to form a joint loss term.  I did this via a luminance channel loss. Another usefull loss for this sort of image data is the relative error, as it is a typical metric used to validate image reconstruction. 

### Data Normalizaton
The original data is calibrated HDR imagery. It has a high dynamic range, with most of the values falling in a very small range.  In order to make the distribution learnable, we need to tranform the data more evenly over a range.  A log transform works well for this purpose,  I did this by initially scaling the data to a 0 to 1 range, applying an exposure compensation, and rescaling.  

### Data sampling
This is where things become more interesting. At this point, one might be asking why use a Dense layers rather than a typical conv net architecture?

 The answer lies in induced data bias while training.  The images are rather large, at 696x464 pixels, a batch size on my machine is 5 images.  Each batch consists of only 5 lighting positions, and this doesn't represent the training data very well, both in terms of input values such as lighting position, but also the resulting image.  When we compare the feature distribution of two batches of equal data size, the batch with a smaller window size much better represents the range of input values.  The plots below compare feature distribution of a single batch of window sizes of 64x64 and 1x1. The features 0 and 2 correpond to light and pixel positions respectively.

<img style="float: center;" src=./documents/features_0_per_batch.png>
<img style="float: center;" src=./documents/features_2_per_batch.png>

The sample problem occurs for the image data we are trying to predict.  The 1x1 window has a much different distribution than the 64x64 window, and much more closely matches the overall
sample distribution.

<img style="float: center;" src=./documents/yfeatures_0_per_batch.png>


In a dense architecture, both the lighting positions and pixel positions are randomly distributed from the entire training set, leading to better convergence and prediction results.

An identical architecture formulated as 1x1 2D kernels fails to produce good results with the entire image as an input.  As we change the input to smaller windows, the loss becomes similar to the dense net architecture as the window size is reduced to a 3x3 window.  




## Example results from the dense network architecture

<img style="float: center;" src=./documents/waldorf_example.png>

<img style="float: center;" src=./documents/bull_example.png>

This data was captured from photographs, so you do see camera sensor noise and focus effects.  The network reproduces these effects as well as find detail off shadows quite well.  The relative error of 5% or 6% is consistent with Ren et al. and is probably the noise floor for this exposure setting.

The Waldorf and Bull models are Mathew O'Toole's, and used in [Optical Computing for Fast Light Transport Analysis](http://www.cs.cmu.edu/~motoole2/opticalcomputing.html).  The git repo contains these two data sets broken up into smaller sized binary chunks.

[The demo notebook](demo.ipynb") shows shows how to train a model and view the results.  I reccomend experimenting with Spyder, trying out different network architectures, losses, and data normalization.

## Implimentation notes

The training procedure is the standard reduce learning on platuea.  I added the additional iteration of reducing the batch size becuase I found that I could increase the overall accuracy of the predictions. 

