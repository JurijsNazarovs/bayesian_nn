import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import numpy as np
import math

import vi_posteriors as vip
import importlib
importlib.reload(vip)

minvar = -3
maxvar = -2

# -------------------------------------------------------------------------------
# Main building blocks
# -------------------------------------------------------------------------------


class BayesLocScaleConvBaseBlock(nn.Module):
    """
    Necessary to define this block, despite that it looks like _ConvNd
    from pyTorch, because we are training posterior parameters to sample weights
    and not weights by itself. Thus, self.weights are not presented here.

    We assume that approximate posterior of weights can be parametrized by
    2 parameters: mu and logsigmasq.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 transposed,
                 groups,
                 prior_mu=0,
                 prior_logsigmasq=0,
                 is_mixed_prior=False,
                 prior_p_mixed=1 / 2,
                 bias=False):
        # Default description of module _ConvNd from pytorch
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.bias = bias

        # Save parameters to fill prior
        self.prior_params = {}
        self.prior_params['mu'] = prior_mu
        self.prior_params['logsigmasq'] = prior_logsigmasq
        self.prior_params['is_mixed'] = is_mixed_prior
        self.prior_params['p_mixed'] = prior_p_mixed

        # Posterior parameters - trained: mu and logsigmasq
        if transposed:
            self.post_mu = Parameter(
                torch.zeros(
                    (in_channels, out_channels // groups, *kernel_size)))

            self.post_logsigmasq = Parameter(
                torch.zeros(
                    (in_channels, out_channels // groups, *kernel_size)))
        else:
            self.post_mu = Parameter(
                torch.zeros(
                    (out_channels, in_channels // groups, *kernel_size)))

            self.post_logsigmasq = Parameter(
                torch.zeros(
                    (out_channels, in_channels // groups, *kernel_size)))

        if self.bias:
            self.post_mu_bias = Parameter(
                torch.zeros((out_channels, ) + (1, ) * len(kernel_size)))
            self.post_logsigmasq_bias = Parameter(
                torch.zeros((out_channels, ) + (1, ) * len(kernel_size)))

        # Initialize parameters

        self.reset_parameters()
        self.reset_priors()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        self.post_mu.data.uniform_(-stdv, stdv)
        self.post_logsigmasq.data.uniform_(minvar, maxvar)  # (0, 1)

        if self.bias:
            self.post_mu_bias.data.uniform_(-stdv, stdv)
            self.post_logsigmasq_bias.data.uniform_(minvar, maxvar)  # (0, 1)

    def reset_priors(self):
        if self.prior_params['is_mixed']:
            # Scale mixture prior
            logsigmasq1 = -2
            logsigmasq2 = -8
            u = torch.rand(self.post_logsigmasq.shape,
                           dtype=self.post_logsigmasq.dtype,
                           requires_grad=False).to("cpu")
            mod_ind = torch.tensor(u <= self.prior_params['p_mixed'],
                                   dtype=self.post_logsigmasq.dtype)

            self.prior_mu = Parameter(torch.zeros_like(self.post_mu) +
                                      self.prior_params['mu'],
                                      requires_grad=False)
            self.prior_logsigmasq = Parameter(mod_ind * logsigmasq1 +
                                              (1 - mod_ind) * logsigmasq2,
                                              requires_grad=False)

            if self.bias:
                self.prior_mu_bias = Parameter(
                    torch.zeros_like(self.post_mu_bias) +
                    self.prior_params['mu'],
                    requires_grad=False)

                u = torch.rand(self.post_logsigmasq_bias.shape,
                               dtype=self.post_logsigmasq.dtype).to("cpu")
                mod_ind = torch.tensor(u <= self.prior_params['p_mixed'],
                                       dtype=self.post_logsigmasq.dtype)
                self.prior_logsigmasq_bias = Parameter(
                    mod_ind * logsigmasq1 + (1 - mod_ind) * logsigmasq2,
                    requires_grad=False)
        else:
            self.prior_mu = Parameter(torch.zeros_like(self.post_mu) +
                                      self.prior_params['mu'],
                                      requires_grad=False)
            self.prior_logsigmasq = Parameter(
                torch.zeros_like(self.post_logsigmasq) +
                self.prior_params['logsigmasq'],
                requires_grad=False)

            if self.bias:
                self.prior_mu_bias = Parameter(
                    torch.zeros_like(self.post_mu_bias) +
                    self.prior_params['mu'],
                    requires_grad=False)
                self.prior_logsigmasq_bias = Parameter(
                    torch.zeros_like(self.post_logsigmasq_bias) +
                    self.prior_params['logsigmasq'],
                    requires_grad=False)

    def extra_repr(self):
        # Displays arguments for a block
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}, '
             'stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'

        s += ', bias = {bias}'
        if hasattr(self, "activation"):
            s += ', activation={activation}'
        if hasattr(self, "approx_post"):
            s += ', approx_post={approx_post}'
        if hasattr(self, "kl_method"):
            s += ', kl_method={kl_method}'
        if hasattr(self, "n_mc_iter"):
            s += ', n_mc_iter={n_mc_iter}'
        if hasattr(self, "compute_kl"):
            s += ', compute_kl={compute_kl}'

        return s.format(**self.__dict__)


class Conv1d(BayesLocScaleConvBaseBlock):
    """
    Bayesian Convolution 1d layer.
    Learned parameters are mu and logsigmasq.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 dilation=1,
                 activation=None,
                 is_mixed_prior=False,
                 bias=False,
                 approx_post="Radial",
                 kl_method="repar",
                 n_mc_iter=20,
                 **kwargs):

        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        self.activation = activation
        self.approx_post = approx_post
        self.kl_method = kl_method
        self.n_mc_iter = n_mc_iter
        self.compute_kl = True

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         False,
                         groups,
                         is_mixed_prior=is_mixed_prior,
                         bias=bias)

    def forward(self, x):
        post = eval("vip." + self.approx_post)
        output, kl = post.forward(self, x, fun="conv1d")

        return output, kl


class Conv2d(BayesLocScaleConvBaseBlock):
    """
    Bayesian Convolution 2d layer.
    Learned parameters are mu and logsigmasq.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 dilation=1,
                 activation=None,
                 is_mixed_prior=False,
                 bias=False,
                 approx_post="Radial",
                 kl_method="repar",
                 n_mc_iter=20,
                 **kwargs):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.activation = activation
        self.approx_post = approx_post
        self.kl_method = kl_method
        self.n_mc_iter = n_mc_iter
        self.compute_kl = True

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         False,
                         groups,
                         is_mixed_prior=is_mixed_prior,
                         bias=bias)

    def forward(self, x):
        post = eval("vip." + self.approx_post)
        output, kl = post.forward(self, x, fun="conv2d")
        return output, kl


class Conv3d(BayesLocScaleConvBaseBlock):
    """
    Bayesian Convolution 3d layer.
    Learned parameters are mu and logsigmasq.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 dilation=1,
                 activation=None,
                 is_mixed_prior=False,
                 bias=False,
                 approx_post="Radial",
                 kl_method="repar",
                 n_mc_iter=20,
                 **kwargs):

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        self.activation = activation
        self.approx_post = approx_post
        self.kl_method = kl_method
        self.n_mc_iter = n_mc_iter
        self.compute_kl = True

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         False,
                         groups,
                         is_mixed_prior=is_mixed_prior,
                         bias=bias)

    def forward(self, x):
        post = eval("vip." + self.approx_post)
        output, kl = post.forward(self, x, fun="conv3d")
        return output, kl


# Transpose classes
class ConvTranspose1d(BayesLocScaleConvBaseBlock):
    """
    Bayesian Convolution 1d layer.
    Learned parameters are mu and logsigmasq.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 dilation=1,
                 activation=None,
                 is_mixed_prior=False,
                 bias=False,
                 approx_post="Radial",
                 kl_method="repar",
                 n_mc_iter=20,
                 **kwargs):

        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        self.activation = activation
        self.approx_post = approx_post
        self.kl_method = kl_method
        self.n_mc_iter = n_mc_iter
        self.compute_kl = True

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         True,
                         groups,
                         is_mixed_prior=is_mixed_prior,
                         bias=bias)

    def forward(self, x):
        post = eval("vip." + self.approx_post)
        output, kl = post.forward(self, x, fun="conv_transpose1d")
        return output, kl


class ConvTranspose2d(BayesLocScaleConvBaseBlock):
    """
    Bayesian Convolution 2d layer.
    Learned parameters are mu and logsigmasq.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 dilation=1,
                 activation=None,
                 is_mixed_prior=False,
                 bias=False,
                 approx_post="Radial",
                 kl_method="repar",
                 n_mc_iter=20,
                 **kwargs):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.activation = activation
        self.approx_post = approx_post
        self.kl_method = kl_method
        self.n_mc_iter = n_mc_iter
        self.compute_kl = True

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         True,
                         groups,
                         is_mixed_prior=is_mixed_prior,
                         bias=bias)

    def forward(self, x):
        post = eval("vip." + self.approx_post)
        output, kl = post.forward(self, x, fun="conv_transpose2d")
        return output, kl


class ConvTranspose3d(BayesLocScaleConvBaseBlock):
    """
    Bayesian Convolution 3d layer.
    Learned parameters are mu and logsigmasq.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 dilation=1,
                 activation=None,
                 is_mixed_prior=False,
                 bias=False,
                 approx_post="Radial",
                 kl_method="repar",
                 n_mc_iter=20,
                 **kwargs):

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        self.activation = activation
        self.approx_post = approx_post
        self.kl_method = kl_method
        self.n_mc_iter = n_mc_iter
        self.compute_kl = True

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         True,
                         groups,
                         is_mixed_prior=is_mixed_prior,
                         bias=bias)

    def forward(self, x):
        post = eval("vip." + self.approx_post)
        output, kl = post.forward(self, x, fun="conv_transpose3d")
        return output, kl


class Linear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mu=0,
                 prior_logsigmasq=0,
                 bias=False,
                 activation=None,
                 approx_post="Radial",
                 kl_method="repar",
                 n_mc_iter=20,
                 **kwargs):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.activation = activation
        self.approx_post = approx_post
        self.kl_method = kl_method
        self.n_mc_iter = n_mc_iter
        self.compute_kl = True

        # Save parameters to fill prior
        self.prior_params = {}
        self.prior_params['mu'] = prior_mu
        self.prior_params['logsigmasq'] = prior_logsigmasq

        # Posterior parameters - trained: mu and logsigmasq
        self.post_mu = Parameter(torch.zeros(out_features, in_features))
        self.post_logsigmasq = Parameter(torch.zeros(out_features,
                                                     in_features))

        if self.bias:
            self.post_mu_bias = Parameter(torch.zeros((out_features)))
            self.post_logsigmasq_bias = Parameter(torch.zeros((out_features)))

        self.reset_parameters()
        self.reset_priors()

    def reset_parameters(self):
        stdv = 1 / 2  # 1. / math.sqrt(self.post_mu.size(1))
        self.post_mu.data.uniform_(-stdv, stdv)
        self.post_logsigmasq.data.uniform_(minvar, maxvar)  # (0, 1)

        if self.bias:
            self.post_mu_bias.data.uniform_(-stdv, stdv)
            self.post_logsigmasq_bias.data.uniform_(minvar, maxvar)  # (0, 1)

    def reset_priors(self):
        self.prior_mu = Parameter(torch.zeros_like(self.post_mu) +
                                  self.prior_params['mu'],
                                  requires_grad=False)
        self.prior_logsigmasq = Parameter(
            torch.zeros_like(self.post_logsigmasq) +
            self.prior_params['logsigmasq'],
            requires_grad=False)
        if self.bias:
            self.prior_mu_bias = Parameter(
                torch.zeros_like(self.post_mu_bias) + self.prior_params['mu'],
                requires_grad=False)
            self.prior_logsigmasq_bias = Parameter(
                torch.zeros_like(self.post_logsigmasq_bias) +
                self.prior_params['logsigmasq'],
                requires_grad=False)

    def extra_repr(self):
        # Displays arguments for a block
        s = ('{in_features}, {out_features}')

        s += ', bias = {bias}'
        if hasattr(self, "activation"):
            s += ', activation={activation}'
        if hasattr(self, "approx_post"):
            s += ', approx_post={approx_post}'
        if hasattr(self, "kl_method"):
            s += ', kl_method={kl_method}'
        if hasattr(self, "n_mc_iter"):
            s += ', n_mc_iter={n_mc_iter}'
        if hasattr(self, "compute_kl"):
            s += ', compute_kl={compute_kl}'

        return s.format(**self.__dict__)

    def forward(self, x):
        post = eval("vip." + self.approx_post)
        output, kl = post.forward(self, x, fun="linear")
        return output, kl
