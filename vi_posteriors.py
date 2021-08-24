"""
This file contains description of classes for different approximate posteriors.
Every class has a corresponding forward and get_kl method
List of classes (approximate posterior, prior):
 1. normal, normal
 2. radial normal, normal
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import re
numer_eps = 1e-8  # number to fix numerical issues with 0. e,g sometimes value
# which has to be non-negative represents 0 as -1e-17, then sqrt returns NaN


class Radial():
    def __init__(self):
        pass

    @staticmethod
    def forward(obj, x, fun=""):
        """
        Forward method based on probability theory
        Input:
         - torch.tensor
        Output:
         - output of layer based on local reparameterization trick
         - kl - divergence
        """

        # Get sample from posterior distribution - local reparameterization trick
        if re.match(r"conv(_transpose)?[1-3]d", fun):
            fun_ = eval("F." + fun)
            conv_post_mu = fun_(input=x,
                                weight=obj.post_mu,
                                stride=obj.stride,
                                padding=obj.padding,
                                dilation=obj.dilation,
                                groups=obj.groups,
                                bias=None).to(x.device)

            conv_post_sigmasq = fun_(input=x**2,
                                     weight=torch.exp(obj.post_logsigmasq).to(
                                         x.device),
                                     stride=obj.stride,
                                     padding=obj.padding,
                                     dilation=obj.dilation,
                                     groups=obj.groups,
                                     bias=None).to(x.device) + numer_eps
        elif fun == "linear":
            conv_post_mu = F.linear(input=x, weight=obj.post_mu,
                                    bias=None).to(x.device)

            conv_post_sigmasq = F.linear(input=x**2,
                                         weight=torch.exp(
                                             obj.post_logsigmasq).to(x.device),
                                         bias=None).to(x.device) + numer_eps
        else:
            raise NotImplementedError(
                "Forward function is not implemented: %s" % fun)

        if obj.bias:
            conv_post_mu += obj.post_mu_bias.\
                unsqueeze(0).expand_as(conv_post_mu)
            conv_post_sigmasq += torch.exp(obj.post_logsigmasq_bias).to(x.device).\
                unsqueeze(0).expand_as(conv_post_sigmasq)

        conv_post_sigmasq[conv_post_sigmasq < numer_eps] = numer_eps
        conv_post_sigma = torch.sqrt(conv_post_sigmasq)
        epsilon = torch.randn(conv_post_mu.size(),
                              dtype=x.dtype,
                              device=x.device)

        epsilon_norm = torch.norm(epsilon).item()
        radius = torch.abs(torch.randn(1, dtype=x.dtype)).item()

        output = conv_post_mu + conv_post_sigma * epsilon / epsilon_norm * radius
        # foo = conv_post_sigma * epsilon / epsilon_norm * radius
        # foo = foo[1]

        # print((conv_post_sigma / epsilon_norm).mean())
        # print("foo.max: ", foo.max())
        # print("foo.mean: ", foo.mean())
        # print("foo.std: ", foo.std())
        # print("foo.min: ", foo.min())
        # import pdb
        # pdb.set_trace()

        if torch.isnan(output).any():
            print("output is nan ---> debug")
            import pdb
            pdb.set_trace()

        if torch.isinf(output).any():
            print("output is inf ---> debug")
            import pdb
            pdb.set_trace()

        if obj.activation:
            output = obj.activation(output)

        if obj.compute_kl or not hasattr(obj, "compute_kl"):
            kl = Radial.get_kl(obj, n_mc_iter=obj.n_mc_iter, device=x.device)
            if torch.isnan(kl):
                print("kl is nan ---> debug")
                import pdb
                pdb.set_trace()
        else:
            # no kl is calculated
            kl = torch.tensor(0, device=x.device, dtype=x.dtype)

        return output, kl

    @staticmethod
    def get_kl(obj, n_mc_iter=20, device="cpu"):
        if obj.kl_method == "repar":
            get_kl_ = Radial.get_kl_repar
        elif obj.kl_method == "direct":
            get_kl_ = Radial.get_kl_direct
        else:
            raise NotImplementedError("Current kl is not implemented: %s" %
                                      obj.kl_method)

        kl = get_kl_(obj.post_mu,
                     obj.post_logsigmasq,
                     obj.prior_mu,
                     obj.prior_logsigmasq,
                     n_mc_iter=n_mc_iter,
                     device=device)

        # Repeat the same for bias
        if obj.bias:
            kl += get_kl_(obj.post_mu_bias,
                          obj.post_logsigmasq_bias,
                          obj.prior_mu_bias,
                          obj.prior_logsigmasq_bias,
                          n_mc_iter=n_mc_iter,
                          device=device)
        return kl

    @staticmethod
    def get_kl_repar(post_mu,
                     post_logsigmasq,
                     prior_mu=0,
                     prior_logsigmasq=0,
                     n_mc_iter=20,
                     device="cpu"):
        # KL divergence based on analytical solution and MCMC
        post_var = torch.exp(post_logsigmasq).to(device)
        prior_var = torch.exp(prior_logsigmasq).to(device)

        # (1) KL of approximate posterior
        kl_entropy = -1 / 2 * torch.sum(torch.log(2 * math.pi * post_var) + 1)

        # (2) KL of prior, MCMC to approximate kl_cross
        sampler_normal = torch.distributions.Normal(
            loc=torch.zeros_like(post_mu),
            scale=torch.ones_like(post_logsigmasq))
        coef1 = 0
        coef2 = 0
        for i in range(n_mc_iter):
            w_sample = sampler_normal.sample((1, )).data
            w_ratio = w_sample / torch.norm(w_sample).data
            radius = torch.abs(torch.randn(1, dtype=post_var.dtype)).item()

            coef1 += w_ratio * radius / n_mc_iter
            coef2 += (w_ratio * radius)**2 / n_mc_iter

        # Approach with cpu
        # w_sample = sampler_normal.sample((n_mc_iter, )).data
        # w_norm = torch.norm(w_sample.reshape(n_mc_iter, -1), p=2, dim=1).data
        # w_ratio = w_sample / w_norm[:, None, None]
        # radius = torch.abs(torch.randn(
        #     n_mc_iter, dtype=post_var.dtype)).data[:, None, None].to(post_mu.device)

        # coef1 = (w_ratio * radius).mean(axis=0)
        # coef2 = ((w_ratio * radius)**2).mean(axis=0)

        mu_ = post_mu - prior_mu

        kl_cross = torch.sum(-1 / 2 * torch.log(2 * math.pi * prior_var) - 1 /
                             (2 * prior_var) *
                             (mu_**2 + 2 * mu_ * torch.sqrt(post_var) * coef1 +
                              post_var * coef2))
        kl = kl_entropy - kl_cross

        # print(kl_cross)
        # exit(1)
        return kl

    @staticmethod
    def get_kl_direct(post_mu,
                      post_logsigmasq,
                      prior_mu=0,
                      prior_logsigmasq=0,
                      n_mc_iter=20,
                      device="cpu"):
        # KL divergence based on analytical solution and MCMC
        post_var = torch.exp(post_logsigmasq).to(device)
        prior_var = torch.exp(prior_logsigmasq).to(device)

        # (1) KL of approximate posterior
        kl_entropy = -1 / 2 * torch.sum(torch.log(2 * math.pi * post_var) + 1)

        # (2) KL of prior, MCMC to approximate kl_cross
        sampler_normal = torch.distributions.Normal(
            loc=torch.zeros_like(post_mu),
            scale=torch.ones_like(post_logsigmasq))
        kl_cross = 0
        for i in range(n_mc_iter):
            radius = torch.randn(1, dtype=post_var.dtype).item()
            weights = sampler_normal.sample((1, ))
            weights = post_mu + \
                torch.sqrt(post_var) * weights / \
                torch.norm(weights).item() * radius

            kl_cross += (torch.sum(-1 / 2 * torch.log(2 * np.pi * prior_var) +
                                   -1 / 2 / prior_var *
                                   (weights - prior_mu)**2) / n_mc_iter)

        kl = kl_entropy - kl_cross
        return kl


class Gaus():
    def __init__(self):
        pass

    @staticmethod
    def forward(obj, x, fun=""):
        """
        Forward method based on probability theory
        Input:
         - torch.tensor
        Output:
         - output of layer based on local reparameterization trick
         - kl - divergence
        """

        # Get sample from posterior distribution - local reparameterization trick
        if re.match(r"conv(_transpose)?[1-3]d", fun):
            fun_ = eval("F." + fun)
            conv_post_mu = fun_(input=x,
                                weight=obj.post_mu,
                                stride=obj.stride,
                                padding=obj.padding,
                                dilation=obj.dilation,
                                groups=obj.groups,
                                bias=None).to(x.device)

            conv_post_sigmasq = fun_(input=x**2,
                                     weight=torch.exp(obj.post_logsigmasq).to(
                                         x.device),
                                     stride=obj.stride,
                                     padding=obj.padding,
                                     dilation=obj.dilation,
                                     groups=obj.groups,
                                     bias=None).to(x.device) + numer_eps

        if fun == "linear":
            conv_post_mu = F.linear(input=x, weight=obj.post_mu,
                                    bias=None).to(x.device)

            conv_post_sigmasq = F.linear(input=x**2,
                                         weight=torch.exp(
                                             obj.post_logsigmasq).to(x.device),
                                         bias=None).to(x.device) + numer_eps

        conv_post_sigmasq[conv_post_sigmasq < numer_eps] = numer_eps

        if obj.bias:
            conv_post_mu += obj.post_mu_bias.\
                unsqueeze(0).expand_as(conv_post_mu)
            conv_post_sigmasq += torch.exp(obj.post_logsigmasq_bias).to(x.device).\
                unsqueeze(0).expand_as(conv_post_sigmasq)

        conv_post_sigma = torch.sqrt(conv_post_sigmasq)
        epsilon = torch.randn(conv_post_mu.size(),
                              dtype=x.dtype,
                              device=x.device)

        output = conv_post_mu + conv_post_sigma * epsilon

        if torch.isnan(output).any():
            print("output is nan ---> debug")
            import pdb
            pdb.set_trace()

        if torch.isinf(output).any():
            print("output is inf ---> debug")
            import pdb
            pdb.set_trace()

        if obj.activation:
            output = obj.activation(output)

        if obj.compute_kl or not hasattr(obj, "compute_kl"):
            kl = Gaus.get_kl(obj, n_mc_iter=obj.n_mc_iter, device=x.device)
            if torch.isnan(kl):
                print("kl is nan ---> debug")
                import pdb
                pdb.set_trace()
        else:
            # no kl is calculated
            kl = torch.tensor(0, device=x.device, dtype=x.dtype)

        return output, kl

    @staticmethod
    def get_kl(obj, n_mc_iter=20, device="cpu"):
        if obj.kl_method == "repar":
            get_kl_ = Gaus.get_kl_repar
        elif obj.kl_method == "direct":
            get_kl_ = Gaus.get_kl_direct
        elif obj.kl_method == "closed":
            get_kl_ = Gaus.get_kl_closed
        else:
            raise NotImplementedError("Current kl is not implemented: %s" %
                                      obj.kl_method)
        kl = get_kl_(obj.post_mu,
                     obj.post_logsigmasq,
                     obj.prior_mu,
                     obj.prior_logsigmasq,
                     n_mc_iter=n_mc_iter,
                     device=device)

        # kl_closed = Gaus.get_kl_closed(obj.post_mu,
        #                                obj.post_logsigmasq,
        #                                obj.prior_mu,
        #                                obj.prior_logsigmasq,
        #                                n_mc_iter=n_mc_iter,
        #                                device=device)

        # Repeat the same for bias
        if obj.bias:
            kl += get_kl_(obj.post_mu_bias,
                          obj.post_logsigmasq_bias,
                          obj.prior_mu_bias,
                          obj.prior_logsigmasq_bias,
                          n_mc_iter=n_mc_iter,
                          device=device)
        return kl

    @staticmethod
    def get_kl_repar(post_mu,
                     post_logsigmasq,
                     prior_mu=0,
                     prior_logsigmasq=0,
                     n_mc_iter=20,
                     device="cpu"):
        # KL divergence based on analytical solution and MCMC
        post_var = torch.exp(post_logsigmasq).to(device)
        prior_var = torch.exp(prior_logsigmasq).to(device)

        # (1) KL of approximate posterior
        kl_entropy = -1 / 2 * torch.sum(torch.log(2 * math.pi * post_var) + 1)

        # (2) KL of prior, MCMC to approximate kl_cross
        sampler_normal = torch.distributions.Normal(
            loc=torch.zeros_like(post_mu),
            scale=torch.ones_like(post_logsigmasq))
        coef1 = 0
        coef2 = 0
        for i in range(n_mc_iter):
            w_sample = sampler_normal.sample((1, )).data

            coef1 += w_sample / n_mc_iter
            coef2 += (w_sample)**2 / n_mc_iter

        # Approach with cpu
        # w_sample = sampler_normal.sample((n_mc_iter, )).data
        # coef1 = w_sample.mean(axis=0)
        # coef2 = ((w_sample)**2).mean(axis=0)

        mu_ = post_mu - prior_mu

        kl_cross = torch.sum(-1 / 2 * torch.log(2 * math.pi * prior_var) - 1 /
                             (2 * prior_var) *
                             (mu_**2 + 2 * mu_ * torch.sqrt(post_var) * coef1 +
                              post_var * coef2))
        kl = kl_entropy - kl_cross

        # print(kl_cross)
        # exit(1)
        return kl

    @staticmethod
    def get_kl_direct(post_mu,
                      post_logsigmasq,
                      prior_mu=0,
                      prior_logsigmasq=0,
                      n_mc_iter=20,
                      device="cpu"):
        # KL divergence based on analytical solution and MCMC
        post_var = torch.exp(post_logsigmasq).to(device)
        prior_var = torch.exp(prior_logsigmasq).to(device)

        # (1) KL of approximate posterior
        kl_entropy = -1 / 2 * torch.sum(torch.log(2 * math.pi * post_var) + 1)

        # (2) KL of prior, MCMC to approximate kl_cross
        sampler_normal = torch.distributions.Normal(
            loc=torch.zeros_like(post_mu),
            scale=torch.ones_like(post_logsigmasq))
        kl_cross = 0
        for i in range(n_mc_iter):
            weights = sampler_normal.sample((1, ))
            weights = post_mu + torch.sqrt(post_var) * weights

            kl_cross += (
                torch.sum(-1 / 2 * torch.log(2 * math.pi * prior_var) +
                          -1 / 2 / prior_var * (weights - prior_mu)**2) /
                n_mc_iter)

        kl = kl_entropy - kl_cross
        return kl

    @staticmethod
    def get_kl_closed(post_mu,
                      post_logsigmasq,
                      prior_mu=0,
                      prior_logsigmasq=0,
                      device="cpu",
                      **kwargs):
        # KL divergence based on analytical solution
        post_var = torch.exp(post_logsigmasq).to(device)
        prior_var = torch.exp(prior_logsigmasq).to(device)

        kl = -1 / 2 * torch.sum(
            torch.log(2 * math.pi * post_var) + 1 +
            -torch.log(2 * math.pi * prior_var) + -1 / prior_var *
            (post_var + (post_mu - prior_mu)**2))

        return kl


class IG():
    def __init__(self):
        pass

    @staticmethod
    def forward(obj, x, fun=""):
        """
        Forward method based on probability theory
        Input:
         - torch.tensor
        Output:
         - output of layer based on local reparameterization trick
         - kl - divergence
        """

        # Get sample from posterior distribution - local reparameterization trick
        if re.match(r"conv(_transpose)?[1-3]d", fun):
            fun_ = eval("F." + fun)
            conv_post_mu = fun_(input=x,
                                weight=torch.exp(obj.post_mu).to(x.device),
                                stride=obj.stride,
                                padding=obj.padding,
                                dilation=obj.dilation,
                                groups=obj.groups,
                                bias=None).to(x.device)

            conv_post_sigma = fun_(
                input=torch.sqrt(x),
                weight=torch.sqrt(torch.exp(obj.post_logsigmasq)).to(x.device),
                stride=obj.stride,
                padding=obj.padding,
                dilation=obj.dilation,
                groups=obj.groups,
                bias=None).to(x.device) + numer_eps
        elif fun == "linear":
            conv_post_mu = F.linear(input=x,
                                    weight=torch.exp(obj.post_mu),
                                    bias=None).to(x.device)

            conv_post_sigma = F.linear(
                input=torch.sqrt(x),
                weight=torch.sqrt(torch.exp(obj.post_logsigmasq)).to(x.device),
                bias=None).to(x.device) + numer_eps
        else:
            raise NotImplementedError(
                "Forward function is not implemented: %s" % fun)

        conv_post_sigmasq = conv_post_sigma**2
        if obj.bias:
            conv_post_mu += torch.exp(obj.post_mu_bias).to(x.device).\
                unsqueeze(0).expand_as(conv_post_mu)
            conv_post_sigmasq += torch.exp(obj.post_logsigmasq_bias).to(x.device).\
                unsqueeze(0).expand_as(conv_post_sigmasq)

        conv_post_sigmasq[conv_post_sigmasq < numer_eps] = numer_eps

        output = IG.sample(conv_post_mu, conv_post_sigmasq, x.dtype, x.device)

        if torch.isnan(output).any():
            print("output is nan ---> debug")
            import pdb
            pdb.set_trace()

        if torch.isinf(output).any():
            print("output is inf ---> debug")
            import pdb
            pdb.set_trace()

        if obj.activation:
            output = obj.activation(output)

        if obj.compute_kl or not hasattr(obj, "compute_kl"):
            kl = IG.get_kl(obj, n_mc_iter=obj.n_mc_iter, device=x.device)
            if torch.isnan(kl):
                print("kl is nan ---> debug")
                import pdb
                pdb.set_trace()
        else:
            # no kl is calculated
            kl = torch.tensor(0, device=x.device, dtype=x.dtype)

        return output, kl

    @staticmethod
    def sample(mu, lam, dtype, device):
        # sample 1 observation from IG
        nu = torch.randn(mu.size(), dtype=dtype, device=device)**2
        epsilon = mu +\
            mu**2 * nu / (2 * lam)\
            - mu / (2 * lam) * torch.sqrt(4 * mu * lam * nu + mu**2 * nu**2)

        u = torch.rand(mu.size(), dtype=dtype, device=device)

        output = torch.zeros_like(epsilon)
        ind = u >= mu / (mu + epsilon)
        output[ind] = (mu**2 / epsilon)[ind]
        output[~ind] = epsilon[~ind]

        return output

    @staticmethod
    def get_kl(obj, n_mc_iter=20, device="cpu"):
        if obj.kl_method == "direct":
            get_kl_ = IG.get_kl_direct
        else:
            raise NotImplementedError("Current kl is not implemented: %s" %
                                      obj.kl_method)

        kl = get_kl_(obj.post_mu,
                     obj.post_logsigmasq,
                     obj.prior_mu,
                     obj.prior_logsigmasq,
                     n_mc_iter=n_mc_iter,
                     device=device)

        # Repeat the same for bias
        if obj.bias:
            kl += get_kl_(obj.post_mu_bias,
                          obj.post_logsigmasq_bias,
                          obj.prior_mu_bias,
                          obj.prior_logsigmasq_bias,
                          n_mc_iter=n_mc_iter,
                          device=device)
        return kl

    @staticmethod
    def get_kl_direct(post_mu,
                      post_logsigmasq,
                      prior_mu=0,
                      prior_logsigmasq=0,
                      n_mc_iter=20,
                      device="cpu"):
        # KL divergence based on analytical solution and MCMC
        post_mu = torch.exp(post_mu).to(device)
        post_var = torch.exp(post_logsigmasq).to(device)
        prior_var = torch.exp(prior_logsigmasq).to(device)

        kl_entropy = 0
        kl_cross = 0

        for i in range(n_mc_iter):
            # (1) KL of approximate posterior
            weights = IG.sample(post_mu, post_var, post_mu.dtype, device)
            kl_entropy += (
                torch.sum(1 / 2 * torch.log(post_var) -
                          1 / 2 * torch.log(2 * np.pi * weights**3) +
                          -(post_var * (weights - post_mu)**2) /
                          (2 * post_mu**2 * weights)) / n_mc_iter)

            # (2) KL of prior, MCMC to approximate kl_cross
            # weights = IG.sample(post_mu, post_logsigmasq, post_mu.dtype,
            #                     device)
            kl_cross += (
                torch.sum(-1 / 2 * torch.log(2 * math.pi * prior_var) +
                          -1 / 2 / prior_var * (weights - prior_mu)**2) /
                n_mc_iter)

        kl = kl_entropy - kl_cross
        return kl


class Utils():
    """
    Class contains differenet utility functions relevant to Bayesian VI
    """
    def __init__():
        pass

    @staticmethod
    def get_beta(beta_type="original", n_batches=1, batch_idx=1):
        """
        Function returns beta for VI inference
        """

        if beta_type == "blundell":
            # https://arxiv.org/abs/1505.05424
            beta = 2**(n_batches - (batch_idx + 1)) / (2**n_batches - 1)
        elif beta_type == "graves":
            # https://papers.nips.cc/paper/2011/file/7eb3c8be3d411e8ebfab08eba5f49632-Paper.pdf
            # eq (18)
            beta = 1 / n_batches
        elif beta_type == "original":
            beta = 1
        else:
            beta = 0

        return beta

    @staticmethod
    def get_loss_categorical(kl, logits, y, beta=1, N=1):
        """
        Calculates the loss of Bayesian Temporal network, according to VI theory,
        given the assumption about categorical loglikehood
        Input:
         - kl is kl divergency
         - logits - predicted scores for all classes
         - y is real observed y
        Output: loss with shape = (Minibatches, 1)
        """
        criterion = nn.CrossEntropyLoss(reduction="sum")
        loglike = -criterion(logits, y.long())  # / y.shape[0]
        loss = beta * kl - loglike * N

        return loss, loglike

    @staticmethod
    def get_loss_normal(kl, params, y, beta=1, N=1):
        """
        Calculates the loss of Bayesian Temporal network, according to VI theory,
        given the assumption about normal loglikehood
        Input:
         - kl is kl divergency
         - params - predicted mu and logsigmasq:
           mu = params[:, 0]
           var = exp(params[:, 1]) or 1, if params.shape[1] != 2
         - y is real observed y
        Output: loss with shape = (Minibatches, 1)
        """

        # y = y.double()
        mu = params[:, 0]
        if mu.shape[1] == 2:
            var = torch.exp(params[:, 1])
        else:
            var = torch.ones(mu.shape).to(y.device)

        loglike = torch.sum(-1 / 2 * torch.log(2 * math.pi * var) - 1 /
                            (2 * var) * (y - mu)**2)  #/ y.shape[0]

        loss = beta * kl - loglike * N
        return loss, loglike

    @staticmethod
    def set_compute_kl(net, compute_kl=False, quite=False):
        '''
        Function set compute_kl attribute to specified value in all models
        '''
        for module in net.modules():
            if hasattr(module, "compute_kl"):
                module.compute_kl = compute_kl
        if not quite:
            print("Compute KL divergence set to: %s" % compute_kl)

    @staticmethod
    def get_summary(net):
        '''
        Function return dictionary - summary of the network,
        including variational part
        '''
        count_compute_kl = 0
        approx_posts = []
        kl_methods = []
        n_mc_iter = []
        summary_info = {}
        n_par_kl = 0

        for module in net.modules():
            if hasattr(module, "compute_kl"):
                count_compute_kl += 1
                n_par_kl += sum(p.numel() for p in module.parameters()
                                if p.requires_grad)

                approx_posts.append(module.approx_post)
                kl_methods.append(module.kl_method)
                n_mc_iter.append(module.n_mc_iter)

        summary_info["approx_posts"] = ','.join(set(approx_posts))
        summary_info["kl_methods"] = ','.join(set(kl_methods))
        summary_info["n_mc_iter"] = ','.join(map(str, set(n_mc_iter)))
        summary_info["count_compute_kl"] = str(count_compute_kl)
        summary_info["n_par_requires_kl"] = str(n_par_kl)
        summary_info["n_parameters"] = str(
            sum(p.numel() for p in net.parameters() if p.requires_grad))

        return summary_info

    @staticmethod
    def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
        # From here:  github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
        L = np.ones(n_epoch)
        period = n_epoch / n_cycle
        step = (stop - start) / (period * ratio)  # linear schedule

        for c in range(n_cycle):

            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_epoch):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L
