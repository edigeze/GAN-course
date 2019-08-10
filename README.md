# imortant paper to read in deep learning.


⭐️⭐️⭐️: You must read

⭐️⭐️ : You should read

⭐️ : you may read if you have time

🔥 : hot paper, new paper a the state of the art

### paper you must read in deep learning

#### Neural network

**Neural Network**⭐️⭐️⭐️

This paper introduce the neural network concept and these applications in NLP and computer vision

Y. LeCun, Y. Bengio, and G. Hinton. [Deep learning](https://creativecoding.soe.ucsc.edu/courses/cs523/slides/week3/DeepLearning_LeCun.pdf), Nature, 2015.

**Dropout**⭐️⭐️

Dropout is a technique widely used for regularization in order to reduce overfitting in neural network

N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov. [Dropout: A simple way to prevent neural networks from overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer). The Journal of Machine Learning Research, 2014

**Batch normalization**⭐️⭐️

batch normalization is a technique used in neural network, We normalize input layer by adjusting and scaling activations. This technique allow to improve speed, performance and stability of neural network

S. Ioffe and C. Szegedy. [Batch normalization: Accelerating deep network training by reducing internal covariate shift](https://arxiv.org/pdf/1502.03167.pdf). 2015.

**Gradients Descent**⭐️⭐️

This paper is a overview of the gradient descent based optimization algorithms for learning deep neural network models.

J. Zhang [Gradient Descent based Optimization Algorithms for Deep Learning Models Training](https://arxiv.org/pdf/1903.03614.pdf) , 2019

**Adam**⭐️⭐️

Adam is a specific gradient descent algorithm widely use for the backpropagation of an neural network.

Kingma, Diederik P and Ba, Jimmy Lei. [Adam: A method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf). 2014.

#### CNN

##### Image Classification

**LeNet**⭐️

One on the first convolutional neural network train on minsit classification. It popularize deep neural netwok

Y. LeCun et al.  [Gradient Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), 1998.

**AlexNet**⭐️⭐️⭐️

This CNN fot image classification significantly outperformed all the prior competitors and won the challenge by reducing the top-5 error from 26% to 15.3%.

First Neural network to won imagenet chalenge

A. Krizhevsky, I. Sutskever, and G. Hinton. [Imagenet classification with deep convolutional neural networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). In NIPS, 2012.

**Inception**⭐️⭐️

CNN for classification perform imageNet competion  with a top-5 error rate of 6.67% (very close to human level performance) for V1

Inception v1 :   C. Szegedy et al. [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf). 2014.

Inception v2,3 :C. Szegedy et al. [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf),2015

Inception v4 :   C. Szegedy et al. [Inception-v4, inception-resnet and the impact of residual connections on learning](https://arxiv.org/pdf/1602.07261.pdf)

**VGG Net**⭐️⭐️

CNN for classification, Similar to AlexNet, only 3x3 convolutions and increasing the nuber of filter and the number of hidden layer.

Simonyan, K., Zisserman, A.: [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf), 2014

**ResNet**⭐️⭐️

CNN for image classification, introduce skip connection, batch normalisation. the increas the  depth of 152 layers it achieves 3.57% error on th ImageNet challenge.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. [Deep residual learning for image recognition](https://arxiv.org/pdf/1512.03385.pdf). IEEE, 2015.

**DenseNet**⭐️⭐️

This CNN used dense connection, fewer parameters than the others and we find a high accuracy.

G. Huang, Z. Liu, and K.Q. Weinberger. [Densely connected convolutional networks](https://arxiv.org/pdf/1608.06993.pdf).2016

##### Object Detection

**R-CNN**⭐️⭐️

CNN for object detection propose methode in three steps: extract region proposals, compute CNN features and Classify regions

R. Girshick, J. Donahue, T. Darrell, and J. Malik. [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf). 2014.

**Fast R-CNN** ⭐️⭐️

Same that R-CNN but it build faster object detection algorithm

CNN for object detection, similar to R-CNN

R. Girshick. [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf),2015

**Faster R-CNN**⭐️⭐️

CNN for object detection, this one don't use selective search to find out the region proposals. it use RPN Region Proposal Network give it less time consuming.

S. Ren, K. He, R. Girshick, and J. Sun. [Faster r-cnn: Towards real-time object detection with region proposal networks](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf). 2015.

**YOLO**⭐️⭐️

All of the previous object detection algorithms use regions to localize the object within the image. The network does not look at the complete image.

J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. [You only look once: Unified, real-time object detection](https://arxiv.org/pdf/1506.02640.pdf). In CVPR, 2016.

**SSD**⭐️⭐️

It is a CNN designed for object detection in real-time. SSD eliminate the need of the region proposal network to be faster. It applies a few improvements including multi-scale features and default boxes

W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.Y. Fu, and A. C. Berg. [Ssd: Single shot multibox detector](https://arxiv.org/pdf/1512.02325.pdf). 2016.

**Mask R-CNN**⭐️⭐️

CNN for object detection and segmentation,it extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.

K. He, G. Gkioxari, P. Dollar, and R. Girshick. [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf). 2017.



##### Segmentation

**U-Net**⭐️⭐️

CNN for image segmentation initialy used for biomedical image, This CNN add a convolution layers and deconvolution layer that give it a U shape.

O. Ronneberger, P. Fischer, and T. Brox. [U-net: Convolutional networks for biomedical image segmentation](https://arxiv.org/pdf/1505.04597.pdf). 2015.

**FCN**⭐️⭐️

Constitute of two parts, Downsampling and Upsampling, this CNN it use in addition skip connection to segmentate the image.

J. Long, E. Shelhamer, and T. Darrell, [Fully convolutional networks for semantic segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf), 2016

#### Other Neural network


**Variational autoencoders**⭐️⭐️

D P. Kingma and M. Welling. [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf). CLR, 2013

**AlphaGo**⭐️

D. Silver et al, [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)

AlphaGo Zero : D. Silver et al, [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)

**Deep Reinforcement Learning**⭐️⭐️

V. Mnih, et al, [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

**Seq To Seq**⭐️⭐️

I. Sutskever, O. Vinyals and Q V. Le,. [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), 2014

**Neural Machine Translation **⭐️⭐️

D. Bahdanau, K. Cho and Y. Bengio. [Neural machine translation by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf), ICLR 2015

**RNN Encoder-Decoder**⭐️⭐️

K. Cho, B. van Merrienboer, C. Gulcehre, F. Bougares, H. Schwenk, and Y. Bengio, [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf), 2014

**BERT**⭐️

J. Cevlin et al. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf),2019

**LSTM**⭐️
S. Hochreiter and J. Schmilhuber. [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)


**Visualization CNN**⭐️

M D. Zeiler and R. Fergus. [Visualizing and understanding convolutional networks](https://arxiv.org/pdf/1311.2901.pdf), 2013

**Show attend and tell**⭐️

K. Xu et al. [Show, attend and tell: Neural image caption generation with visual attention](https://arxiv.org/pdf/1502.03044.pdf)

Y. Kim. [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181), 2014

**Delving Deep into Rectifiers**⭐️

K. He et al. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf)

**LRCN**⭐️

J. Donahue, et al.[Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/pdf/1411.4389.pdf),2016



**Adversarial Autoencoders**

A. Makhzani et al. [Adversarial Autoencoders](https://arxiv.org/pdf/1511.05644.pdf),2016

