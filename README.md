Monte Carlo (MC) estimator is the core concept used in Bayesian Variational Inference.
Higher number of MC samples lead to lower variance of the MC estimator and 
higher accuracy. However, with the direct implementation of MC estimator for KL term,
increasing number of MC sampels results in the GPU memory explosion in
Deep Bayesian Neural Networks.

We present the new scheme to compute MC estimator of KL term in Bayesian VI 
settings with almost no memory cost in GPU, regardles of the number of samples (even 1000+),
and significantly improves run time (Figure below). 
Our method is described in the paper (UAI2021):
["Graph Reparameterizations for Enabling 1000+ Monte Carlo Iterations in Bayesian Deep Neural Networks"](paper.pdf).

![timega](images/Batch_Time_GA_comparison.png)

In addition, we provide an implementation framework to make your deterministic
network Bayesian in `PyTorch`. 

If you like our work, please click on a star. If you use our code in your research projects,
please cite our paper above.

# Bayesify your Neural Network

There are 3 main files which help you to `Bayesify` your deterministic network:

1. `bayes_layers.py` - file contains a bayesian implementation of convolution(1d, 2d, 3d, transpose)
and linear layers, according to approx posterior from `Location-Scale` family, i.e. which has 2 parameters
mu and sigma. This file contains general definition, *independent* of specific distribution,
as long as distribution contains 2 parameters mu and sigma. 
It uses forward method defined in `vi_posteriors.py` file. 
One of the main arguments for redefined classes is `approx_post`,
which defined which posterior class to use from `vi_posteriors.py`.
Please, specify this name same way as defined class in `vi_posteriors.py`.
For example, if `vi_posteriors.py` contains class Gaus, then `approx_post='Gaus'`.

2. `vi_posteriors.py` - file describes forward method, including kl term, for different
approximate posterior distributions. Current implementation contains following 
disutributions:
- Radial 
- Gaus

If you would like to implement your own class of distrubtions, in `vi_posteriors.py`
copy one of defined classes
and redefine following functions: `forward(obj, x, fun="")`, `get_kl(obj, n_mc_iter, device)`.

It also contains usefull Utils class which provides 
* definition of loss functions:
  - get_loss_categorical
  - get_loss_normal,
* different beta coefficients: `get_beta` for KL term and
* allows to turn on/off computing the KL term, with function `set_compute_kl`. 
this is useful, when you perform testing/evaluation, and kl term is not required
to be computed. In that case it accelerates computations.



Below is an example to bayesify your own network. Note the forward method, 
which handles situations if a layer is not of a Bayesian type, and thus, 
does not return kl term, e.g. ReLU(x).

```python

import bayes_layers as bl # important for defining bayesian layers
class YourBayesNet(nn.Module):
    def __init__(self, num_classes, in_channels, 
                 **bayes_args):
        super(YourBayesNet, self).__init__()
        self.conv1 = bl.Conv2d(in_channels, 64,
                               kernel_size=11, stride=4,
                               padding=5,
                               **bayes_args)
        self.classifier = bl.Linear(1*1*128,
                                    num_classes,
                                    **bayes_args)
        self.layers = [self.conv1, nn.ReLU(), self.classifier]
        
    def forward(self, x):
        kl = 0
        for layer in self.layers:
            tmp = layer(x)
            if isinstance(tmp, tuple):
                x, kl_ = tmp
                kl += kl_
            else:
                x = tmp

        x = x.view(x.size(0), -1)
        logits, _kl = self.classifier.forward(x)
        kl += _kl
        
        return logits, kl
```

Then later in the main file during training, you can either use one of the loss functions, defined in utils as following:
``` python

output, kl = model(inputs)
kl = kl.mean()  # if several gpus are used to split minibatch

loss, _ = vi.Utils.get_loss_categorical(kl, output, targets, beta=beta) 
#loss, _ = vi.Utils.get_loss_normal(kl, output, targets, beta=beta) 
loss.backward()
```
or design your own, e.g. 
```python 
loss = kl_coef*kl - loglikelihood
loss.backward()
```



3. `uncertainty_estimate.py` - file describes set of functions to perform uncertainty
estimation, e.g. 
- get_prediction_class - function which return the most common class in iterations
- summary_class - function creates a summary file with statistics

# Current implementation of networks for different problems
## Classification
Script bayesian_dnn_class/main.py is the main executable code and 
all standard DNN models are located in bayesian_dnn_class/models, and are:
- AlexNet
- Fully Connected
- DenseNet
- ResNet
- VGG

