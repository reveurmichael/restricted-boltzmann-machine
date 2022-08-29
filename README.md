# restricted-boltzmann-machine

## Restricted Boltzmann Machine and Its Application

https://www.latentview.com/blog/restricted-boltzmann-machine-and-its-application/

As mentioned earlier Restricted Boltzmann Machine is an **unsupervised learning** algorithm , so how does it learn without using any output data?  We have a visible layer of neurons that receives input data which is multiplied by some weights and added to a bias value at the hidden layer neuron to generate output. Then the output value generated at the hidden layer neuron will become a new input which is then multiplied with the same weights and then bias of the visible layer will be added to regenerate input. This process is called reconstruction or backward pass. Then the regenerated input will be compared with the original input if it matches or not. This process will keep on happening until the regenerated input is aligned with the original input.


### Applications of RBM 

HandWritten Digit Recognition is a very common problem these days and  is used in a variety of  applications like criminal evidence, office computerization, check verification, and data entry applications. It also comes with  challenges like different writing style, variations in shape and size as well as image noise, which leads to changes in numeral topology. In this a hybrid RBM-CNN methodology is used for digit recognition. First, features are extracted using RBM deep learning algorithms. Then extracted features are fed to the CNN deep learning algorithm for classification. RBMs are highly capable for extracting features from input data. It is designed in such a way that it can extract the discriminative features from large and complex datasets by introducing hidden units in an unsupervised manner. 

## Why is the restricted boltzmann machine both unsupervised and generative?

https://stats.stackexchange.com/questions/110706/why-is-the-restricted-boltzmann-machine-both-unsupervised-and-generative


## 2.9. Neural network models (unsupervised)
https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html

### 2.9.1. Restricted Boltzmann machines

Restricted Boltzmann machines (RBM) are unsupervised nonlinear feature learners based on a probabilistic model. The features extracted by an RBM or a hierarchy of RBMs often give good results when fed into a linear classifier such as a linear SVM or a perceptron.

The model makes assumptions regarding the distribution of inputs. At the moment, scikit-learn only provides BernoulliRBM, which assumes the inputs are either binary values or values between 0 and 1, each encoding the probability that the specific feature would be turned on.


**The method gained popularity for initializing deep neural networks with the weights of independent RBMs. This method is known as unsupervised pre-training.**



#### 2.9.1.2. Bernoulli Restricted Boltzmann machines

In the [**BernoulliRBM**](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#sklearn.neural_network.BernoulliRBM), all units are binary stochastic units. This means that the input data should either be binary, or real-valued between 0 and 1 signifying the probability that the visible unit would turn on or off. This is a good model for character recognition, where the interest is on which pixels are active and which aren’t. For images of natural scenes it no longer fits because of background, depth and the tendency of neighbouring pixels to take the same values.


## Beginner's Guide to Boltzmann Machines in PyTorch

https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/

Unlike other neural network models that we have seen so far, the architecture of Boltzmann Machines is quite different. There is no clear demarcation between the input and output layer. In fact, there is no output layer. The nodes in Boltzmann Machines are simply categorized as visible and hidden nodes. The visible nodes take in the input. The same nodes which take in the input will return back the reconstructed input as the output. This is achieved through bidirectional weights which will propagate backwards and render the output on the visible nodes.

### Implementation of RBMs in PyTorch

Step 1: Importing the required libraries


```python
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
%matplotlib inline
import matplotlib.pyplot as plt
```


Step 2: Loading the MNIST Dataset


```python
batch_size = 64
train_loader = torch.utils.data.DataLoader(
datasets.MNIST('./data',
    train=True,
    download = True,
    transform = transforms.Compose(
        [transforms.ToTensor()])
     ),
     batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(
datasets.MNIST('./data',
    train=False,
    transform=transforms.Compose(
    [transforms.ToTensor()])
    ),
    batch_size=batch_size)
```

Step 3: Defining the Model

```python
class RBM(nn.Module):
   def __init__(self,
               n_vis=784,
               n_hin=500,
               k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
    
   def sample_from_p(self,p):
       return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
   def v_to_h(self,v):
        p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
   def h_to_v(self,h):
        p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
   def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_
    
   def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

```

Step 4: Initialising and Training the Model


```python
rbm = RBM(k=1)
train_op = optim.SGD(rbm.parameters(),0.1)

for epoch in range(10):
    loss_ = []
    for _, (data,target) in enumerate(train_loader):
        data = Variable(data.view(-1,784))
        sample_data = data.bernoulli()
        
        v,v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data)
        train_op.zero_grad()
        loss.backward()
        train_op.step()

    print("Training loss for {} epoch: {}".format(epoch, np.mean(loss_)))

```

```python
def show_adn_save(file_name,img):
    npimg = np.transpose(img.numpy(),(1,2,0))
    f = "./%s.png" % file_name
    plt.imshow(npimg)
    plt.imsave(f,npimg)
```

Step 5: Visualising the Outputs
```python
show_adn_save("real",make_grid(v.view(32,1,28,28).data))
```

```python
show_adn_save("generate",make_grid(v1.view(32,1,28,28).data))
```

## Google Colab: RECOMMENDATION SYSTEM WITH A RESTRICTED BOLTZMANN MACHINE

https://colab.research.google.com/github/Gurubux/CognitiveClass-DL/blob/master/2_Deep_Learning_with_TensorFlow/DL_CC_2_4_RBM/4.2-Review-CollaborativeFilteringwithRBM.ipynb#scrollTo=MWptXXhiGZDJ

Welcome to the Recommendation System with a Restricted Boltzmann Machine notebook. In this notebook, we study and go over the usage of a Restricted Boltzmann Machine (RBM) in a Collaborative Filtering based recommendation system. This system is an algorithm that recommends items by trying to find users that are similar to each other based on their item ratings. By the end of this notebook, you should have a deeper understanding of how Restricted Boltzmann Machines are applied, and how to build one using TensorFlow.

Next, let's start building our RBM with TensorFlow. We'll begin by first determining the number of neurons in the hidden layers and then creating placeholder variables for storing our visible layer biases, hidden layer biases and weights that connects the hidden layer with the visible layer. We will be arbitrarily setting the number of neurons in the hidden layers to 20. You can freely set this value to any number you want since each neuron in the hidden layer will end up learning a feature.

We then move on to creating the visible and hidden layer units and setting their activation functions. In this case, we will be using the tf.sigmoid and tf.relu functions as nonlinear activations since it is commonly used in RBM's.

```python
#Phase 1: Input Processing
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
#Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb) 
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)
```

## Restricted Boltzmann machines and pretraining

Restricted Boltzmann machines are a simple but non-trivial type of neural network. RBMs are interesting because they can be used with a larger neural network to perform pre-training: finding "good default" weights for a neural network without having to do more expensive backpropogation work, by lifting weights from a simpler network that we train beforehand that we teach to approximately reconstruct the input data. Pre-training has historically been important in the development of neural networks because it allows training large neural networks faster and more reliably than would otherwise be possible. We will see how they do so in this notebook.

Note that pre-training has fallen somewhat out of favor on the bleeding edge because it has been somewhat obviated by better optimization algorithms. Additionally RBMs have been somewhat displaced by more complicated but more robust autoencoders (topic for the future).

The RBM learns on each pass. On the forward (projection) pass, the reconstruction is treated as the target, and gradient descent is used to adjust the weights on the input layer. On the backwards pass, the original dataset is treated as the target, and gradient descent is used to adjust the weights on the hidden layer. Since both backwards and forwards weights are the same, each pass simultaneously adjusts both forward and backwards weights.

We are effectively managing two distributions. One distribution is a function of the input data,  f(X)=Xp . The other distribution is a function of that output,  g(Xp)=Xq . By making the weights that control  f(X)  and  g(X)  the same, and by performing backpropogration on each pass through the network, we are effectively making  f(Xp)  and  g(Xp)  converge to roughly the same distribution. In other words,  f(X)≈g−1(X)  and vice versa: the forward and backwards functions become roughly inverses of one another.

To measure the distance between its estimated probability distribution and the ground-truth distribution of the input, RBMs use Kullback Leibler Divergence.



RBMs are not implemented in keras, because as mentioned in the lead they have fallen out of favor. You may implement them yourself using caffe or tensorflow or a similar low level fraomework (or copy one of a number of code snippets floating around online that do just that), or you can use the scikit-learn implementation.



```python
import pandas as pd
import numpy as np

X_train = pd.read_csv('../input/train.csv').values[:,1:]  # exclude the target
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # rescale to (0, 1)


from sklearn.neural_network import BernoulliRBM
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=42, verbose=True)
rbm.fit(X_train)


def gen_mnist_image(X):
    return np.rollaxis(np.rollaxis(X[0:200].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)

xx = X_train[:40].copy()
for _ in range(1000):
    for n in range(40):
        xx[n] = rbm.gibbs(xx[n])
        
import matplotlib.pyplot as plt
plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(xx), cmap='gray')



plt.figure(figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.RdBu,
               interpolation='nearest', vmin=-2.5, vmax=2.5)
    plt.axis('off')


```

The work that the RBM does in finding a sparse representation of the original dataset actually falls into a general class of techniques known as dimensionality reduction techniques. The most famous reduction technique is PCA, which I demonstrate in the notebook "Dimensionality Reduction and PCA for Fashion-MIST. RBMs are not very useful for dimensionality reduction because they are too slow.


## Restricted Boltzmann Machine as Recommendation System for Movie Review (part 2)

https://towardsdatascience.com/restricted-boltzmann-machine-as-a-recommendation-system-for-movie-review-part-2-9a6cab91d85b


## What are the practical difference between an RBM and autoencoder?
https://www.reddit.com/r/MachineLearning/comments/20ue0z/what_are_the_practical_difference_between_an_rbm/

RBMs are generative models, AEs are not. However recent work has bridged even that gap. See the following: Denoising Autoencoders as Generative Models, NADE models. AEs support a recursive architecture useful for text (and other domains lending to recursion) that I don't think has an RBM equivalent. In other cases, I think cross-validation is the only way to gauge which pre-training method you should use.

## Difference between Autoencoders & RBMs

https://medium.com/edureka/restricted-boltzmann-machine-tutorial-991ae688c154

Autoencoder is a simple 3-layer neural network where output units are directly connected back to input units. Typically, the number of hidden units is much less than the number of visible ones. The task of training is to minimize an error or reconstruction, i.e. find the most efficient compact representation for input data.

RBM shares a similar idea, but it uses stochastic units with particular distribution instead of deterministic distribution. The task of training is to find out how these two sets of variables are actually connected to each other.

One aspect that distinguishes RBM from other autoencoders is that it has two biases.

- The hidden bias helps the RBM produce the activations on the forward pass, while
- The visible layer’s biases help the RBM learn the reconstructions on the backward pass.

## What is the difference between convolutional neural networks, restricted Boltzmann machines, and auto-encoders?

**Autoencoder** is a simple 3-layer neural network where output units are directly connected back to input units. E.g. in a network like this:

output[i] has edge back to input[i] for every i. Typically, number of hidden units is much less then number of visible (input/output) ones. As a result, when you pass data through such a network, it first compresses (encodes) input vector to "fit" in a smaller representation, and then tries to reconstruct (decode) it back. The task of training is to minimize an error or reconstruction, i.e. find the most efficient compact representation (encoding) for input data.

**RBM** shares similar idea, but uses stochastic approach. Instead of deterministic (e.g. logistic or ReLU) it uses stochastic units with particular (usually binary of Gaussian) distribution. Learning procedure consists of several steps of Gibbs sampling (propagate: sample hiddens given visibles; reconstruct: sample visibles given hiddens; repeat) and adjusting the weights to minimize reconstruction error.

Intuition behind RBMs is that there are some visible random variables (e.g. film reviews from different users) and some hidden variables (like film genres or other internal features), and the task of training is to find out how these two sets of variables are actually connected to each other (more on this example may be found here).

### Dimensionality reduction
When we represent some object as a vector of n elements, we say that this is a vector in n-dimensional space. Thus, dimensionality reduction refers to a process of refining data in such a way, that each data vector x is translated into another vector x′ in an m-dimensional space (vector with m elements), where m<n. Probably the most common way of doing this is PCA. Roughly speaking, PCA finds "internal axes" of a dataset (called "components") and sorts them by their importance. First m most important components are then used as new basis. Each of these components may be thought of as a high-level feature, describing data vectors better than original axes.

Both - autoencoders and RBMs - do the same thing. Taking a vector in n-dimensional space they translate it into an m-dimensional one, trying to keep as much important information as possible and, at the same time, remove noise. If training of autoencoder/RBM was successful, each element of resulting vector (i.e. each hidden unit) represents something important about the object - shape of an eyebrow in an image, genre of a film, field of study in scientific article, etc. You take lots of noisy data as an input and produce much less data in a much more efficient representation.

### All of these architectures can be interpreted as a neural network
In contrast, Autoencoders almost specify nothing about the topology of the network. They are much more general. The idea is to find good neural transformation to reconstruct the input. They are composed of encoder (projects the input to hidden layer) and decoder (reprojects hidden layer to output). The hidden layer learns a set of latent features or latent factors. Linear autoencoders span the same subspace with PCA. Given a dataset, they learn number of basis to explain the underlying pattern of the data.

RBMs are also a neural network. But interpretation of the network is totally different. RBMs interpret the network as not a feedforward, but a bipartite graph where the idea is to learn joint probability distribution of hidden and input variables. They are viewed as a graphical model. Remember that both AutoEncoder and CNN learns a deterministic function. RBMs, on the other hand, is generative model. It can generate samples from learned hidden representations. There are different algorithms to train RBMs. However, at the end of the day, after learning RBMs, you can use its network weights to interpret it as a feedforward network.


