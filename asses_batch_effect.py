import torch
import pandas as pd
import statsmodels.api as sm
import numpy as np

from statsmodels.formula.api import ols
from scipy.stats import f as fisher_dist



## corrected_data should be tensor of size (n_features, n_samples)
## Main function assesing the batch effect present in a dataset. This compares the
## F statistic distribution to the fisher distribution.
# def fisher_kldiv_detailed(corrected_data, n_batches, batch_size, batchless_entropy):
#     y = corrected_data
#     length = len(y)
#     y_mean = torch.mean(y, 1).view(length, 1).repeat_interleave(n_batches * batch_size, 1)

#     y_batch_mean = y.view(length, n_batches, batch_size)
#     y_batch_mean = torch.mean(y_batch_mean, 2).repeat_interleave(batch_size, 1)

#     exp_var = torch.sum(torch.square(y_batch_mean - y_mean), 1)
#     unexp_var = torch.sum(torch.square(y - y_batch_mean), 1)

#     N = batch_size * n_batches
#     K = n_batches

#     F_stat = (exp_var/unexp_var) * ((N-K) / (K-1))
#     p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
#     log_F = p.log_prob(F_stat)
#     return(log_F - batchless_entropy)

def compute_F_stat(corrected_data, n_batches, batch_size):
    y = corrected_data
    y_dim = y.dim()
    # length = y.size(y_dim-2)
    view_args = []
    for index in range(0, y_dim-1):
        view_args = view_args + [y.size(index)]
    view_args = view_args + [1]
    y_mean = torch.mean(y, y_dim-1).view(*view_args).repeat_interleave(n_batches * batch_size, y_dim-1)

    view_args = []
    for index in range(0, y_dim-1):
        view_args = view_args + [y.size(index)]
    view_args = view_args + [n_batches, batch_size]
    y_batch_mean = y.view(*view_args)
    y_batch_mean = torch.mean(y_batch_mean, y_dim).repeat_interleave(batch_size, y_dim-1)

    exp_var = torch.sum(torch.square(y_batch_mean - y_mean), y_dim-1)
    unexp_var = torch.sum(torch.square(y - y_batch_mean), y_dim-1)

    N = batch_size * n_batches
    K = n_batches

    F_stat = (exp_var/unexp_var) * ((N-K) / (K-1))
    return F_stat

def fisher_kldiv(corrected_data, n_batches, batch_size, batchless_entropy):
    N = batch_size * n_batches
    K = n_batches
    F_stat = compute_F_stat(corrected_data, n_batches, batch_size)

    p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
    log_F = p.log_prob(F_stat)
    return -(log_F - batchless_entropy)

# def fisher_kldiv(corrected_data, n_batches, batch_size, batchless_entropy):
#     distance = fisher_kldiv_detailed(corrected_data, n_batches, batch_size, batchless_entropy)
#     f_dim = distance.dim()

#     loss_kl = torch.sum(distance, f_dim-1)
#     return loss_kl


def abs_effect_estimate(corrected_data, n_batches, batch_size, batchless_entropy):
    y = corrected_data
    y_dim = y.dim()
    length = y.size(y_dim-2)
    distance = fisher_kldiv(corrected_data, n_batches, batch_size, batchless_entropy)
    f_dim = distance.dim()

    loss_kl = torch.sum(abs(distance), f_dim-1)/length
    return loss_kl


## y is a tensor of size (k, n_batches * batch_size)
def test_batch_effect_fast(y, n_batches, batch_size):
    length = len(y)
    y_mean = torch.mean(y, 1).view(length, 1).repeat_interleave(n_batches * batch_size, 1)

    y_batch_mean = y.view(length, n_batches, batch_size)
    y_batch_mean = torch.mean(y_batch_mean, 2).repeat_interleave(batch_size, 1)

    exp_var = torch.sum(torch.square(y_batch_mean - y_mean), 1)
    unexp_var = torch.sum(torch.square(y - y_batch_mean), 1)

    N = batch_size * n_batches
    K = n_batches

    F_stat = (exp_var/unexp_var) * ((N-K) / (K-1))
    # df2test = 10000
    # p_values = 1 - fisher_dist.cdf(F_stat, dfn = K-1, dfd = df2test)
    p_values = 1 - fisher_dist.cdf(F_stat, dfn = K-1, dfd = N-K)

    return(p_values) 
    

## Functions for testing batch effect
## Slow method for testing batch effect. Here for sanity checks.
def test_batch_effect(y, n_batches, batch_size):
    p_values = list()

    for yy in y:
        d = {'value' : yy, 'batch' : [format(b) for b in range(n_batches) for i in range(batch_size)]}
        dff = pd.DataFrame(data = d)
        model = ols('value ~ batch', data = dff).fit()
        aov_table = sm.stats.anova_lm(model, typ = 2)
        p_value = aov_table.iloc[0,3]
        p_values.append(p_value)

    return(p_values)


def batchless_entropy_estimate(n_batches, batch_size, sample_size = 7000000):
    N = batch_size * n_batches
    K = n_batches

    p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
    F_stat = np.random.f(K-1, N-K, sample_size)
    log_F = p.log_prob(torch.tensor(F_stat))
    return float(torch.mean(log_F)), float(torch.std(log_F, unbiased = True))


def batchless_entropy_distribuions(n_batches, batch_size, n_divisions, n_overlap, sample_size = 10000000):  
    natural_distributions = []

    div_size = sample_size//n_divisions
    step_size = div_size//n_overlap
    natural_distributions = []
    for j in range(n_overlap):
        natural_distributions = natural_distributions + [(i*div_size + j*step_size, (i + 1)*div_size + j*step_size) for i in range(n_divisions)]

    N = batch_size * n_batches
    K = n_batches
    p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
    F_stat = np.random.f(K-1, N-K, sample_size)
    log_F = p.log_prob(torch.tensor(F_stat))
    F_df = pd.DataFrame({'index' : range(0, len(F_stat)), 'F_stat' : F_stat, 'batch_dist' : log_F})

    A = F_df.sort_values(by = 'F_stat', ascending = False)
    A = pd.concat([A,A])

    for index, window in enumerate(natural_distributions):
        natural_distributions[index] = A.iloc[window[0]:window[1]]['batch_dist'].to_list()

    for index, dist in enumerate(natural_distributions):
        dist = torch.tensor(dist)
        natural_mean = float(torch.mean(dist))
        natural_std = float(torch.std(dist, unbiased = True))
        natural_distributions[index] = (natural_mean, natural_std)
    return natural_distributions


    # div_size = sample_size//n_div
    # N = batch_size * n_batches
    # K = n_batches

    # p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
    # F_stat = np.random.f(K-1, N-K, sample_size)
    # log_F = p.log_prob(torch.tensor(F_stat))
    # individual_distance = pd.DataFrame({'index' : range(0, len(F_stat)),
    #                                 'F_stat' : F_stat,
    #                                 'batch_dist' : log_F})

    # C = individual_distance.sort_values(by = 'F_stat', ascending = False)
    # natural_distributions = [C['batch_dist'].to_list()[i*div_size:(1+i)*div_size] for i in range(0, n_div)]
    # for index, dist in enumerate(natural_distributions):
    #     dist = torch.tensor(dist)
    #     natural_mean = float(torch.mean(dist))
    #     natural_std = float(torch.std(dist, unbiased = True))
    #     natural_distributions[index] = (natural_mean, natural_std)
    # return natural_distributions
    








