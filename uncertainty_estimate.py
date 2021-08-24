import torch
import numpy as np
import os
from collections import Counter


def summary_class(data,
                  n_classes=10,
                  out_dir="./stat",
                  note="",
                  write_mode='a+',
                  targets=None):
    '''
    Function creates/append a summary file based on the input data.
    File is created for every value of the minibatch, thus,
    order of data should be fixed and not shuffled across iterations.
    File header:  Median; Mean; Var; Lb, Rb; frequencies 1:n_classes
    input:
      data - first dimension: number of testing iterations = samples
             second dimension: minibtach
             values: ind of the class with maximum score
    '''

    os.makedirs(out_dir, exist_ok=True)
    predicted = []
    data = data.data.cpu().numpy()
    for i in range(data.shape[1]):
        summary_file_path = "%s/%d%s.txt" % (out_dir, i, note)
        data_ = data[:, i]  # data for 1 example

        # Statistics
        mean = data_.mean()
        median = np.median(data_)
        std = data_.std()
        lb, rb = mean - 1.96 * std, mean + 1.96 * std
        counter = Counter(data_.tolist())
        freq = [0] * n_classes
        for j in range(n_classes):
            freq[j] = counter[j]  #/ len(data_)

        # Summary to the file
        target_str = str(
            targets[i].data.cpu().numpy()) if targets[i] is not None else ''

        summary_file = open(summary_file_path, write_mode)
        summary_file.write(
            "%.6f;%.6f;%.6f;%.6f,%.6f;%s;%s\n" %
            (median, mean, std, lb, rb, ','.join(map(str, freq)), target_str))
        summary_file.close()

        # Predicted value for this example
        predicted.append(np.argmax(freq))

    return torch.tensor(predicted, dtype=torch.int64).reshape(1, data.shape[1])


def get_prediction_class(data, n_classes=10, is_score=False, threshold=0.5):
    '''
    Function the most common class in data per entry among n_test_iter
    data - first dimension: number of testing iterations = samples
           second dimension: minibtach
           values: ind of the class with maximum score
    '''

    predicted = []
    if is_score:
        data = torch.round(data - (threshold - 0.5))
    data = data.data.cpu().numpy()
    if data.shape[0] == 1:
        return torch.tensor(data, dtype=torch.int64).reshape(1, data.shape[1])

    for i in range(data.shape[1]):
        data_ = data[:, i]  # data for 1 example

        # Statistics
        counter = Counter(data_.tolist())
        freq = [0] * n_classes

        for i in range(n_classes):
            freq[i] = counter[i]

        # Predicted value for this example
        predicted.append(np.argmax(freq))
    return torch.tensor(predicted, dtype=torch.int64).reshape(1, data.shape[1])


def get_norm_parameters(net):
    '''
    Function returns l2 norm of parameters
    '''
    vals = {}
    uniq_par = set([
        name.split('.')[-1] for name, p in net.named_parameters()
        if p.requires_grad and "post_" in name
    ])
    for key in uniq_par:
        vals[key] = 0

    for name, par in net.named_parameters():
        if par.requires_grad:
            var = name.split(".")[-1]  # make it post_mean, post_var and so on
            if "post_" not in var:
                continue

            val = par.detach().cpu().numpy()
            vals[var] += (val**2).sum()

    for key in vals.keys():
        vals[key] = np.sqrt(vals[key])

    return (vals)


def summary_parameters(net, out_dir="./stat", note="", write_mode='a+'):
    os.makedirs(out_dir, exist_ok=True)
    w_norm = get_norm_parameters(net)
    for param in w_norm.keys():
        summary_file_path = "%s/%s%s.txt" % (out_dir, param, note)

        # Summary to the file
        summary_file = open(summary_file_path, write_mode)
        summary_file.write("%f\n" % w_norm[param])
        summary_file.close()
