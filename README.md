#Image Based Relighting Using Neural Networks

This project is a simple implementation of image based relighting using deep networks.

The goal is to predict how the scene looks under difference lighting conditions based
based on a few samples.

Much of this work is inspired by Ren et al. [Image Based Relighting Using Neural Networks](https://www.microsoft.com/en-us/research/video/image-based-relighting-using-neural-networks-2/).

However their work is somewhat complicated to implement due to their hierarchical divide and conquer method,
ensemble model, and distributed learning framework.  To give credit, they probably didn't have fast GPUs.

The main idea behind this work is that we can implement a deep network with approximately the same number
of parameters, a simpler structure, and train the model much faster on a single machine. 




[The demo notebook](demo.ipynb") shows shows how to train a model and view the results.



The Waldorf and Bull models are Mathew O'Toole's used from Optical Computing for [Fast Light Transport Analysis](http://www.cs.cmu.edu/~motoole2/opticalcomputing.html).

<img style="float: right;" src="docments/waldorf_example.png">


![Waldof example](https://github.com/alexcolburn/neural_relighting/blob/master/documents/waldorf_example.png)

![Bull example](https://github.com/alexcolburn/neural_relighting/blob/master/documents/bull_example.png)
