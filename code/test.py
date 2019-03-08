#!/home/aks249/anaconda3/envs/py27/bin/python
import os
import sys

from ipdb import launch_ipdb_on_exception
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from bptf import BPTF
from bpptf import BPPTF
from utils import parafac


def two_sided_geometric(p, size=()):
    """Generate 2-sided geometric noise"""
    if p == 0:
        return np.zeros(size)

    rte = (1 - p) / p
    Lambda_2 = np.random.gamma(1, 1. / rte, size=(2,) + size)
    G_2 = np.random.poisson(Lambda_2)
    return G_2[0] - G_2[1]


def main(n_docs, n_words, alpha, beta, rank, priv, n_iters=200, out_file='test_out.npz'):
    try:
        dat_file = np.load('test_data.npz')
        data_DV = dat_file['Y_DV']
        assert(data_DV.shape == (n_docs, n_words))
        noisy_data_DV = dat_file['noisy_data_DV']
        phi_KV = dat_file['phi_KV']
        assert(phi_KV.shape == (rank, n_words))
        theta_DK = dat_file['theta_DK']
        poisson_priors_DV = parafac((theta_DK, phi_KV.T))
    except:
        output_data_shape = (n_docs, n_words)
        theta_DK = np.random.gamma(alpha, 1./beta, (n_docs, rank))
        phi_KV = np.random.gamma(alpha, 1./beta, (rank, n_words))
        poisson_priors_DV = parafac((theta_DK, phi_KV.T))
        # Sample true data and noisy data
        data_DV = np.random.poisson(poisson_priors_DV, output_data_shape)
        noisy_data_DV = data_DV + two_sided_geometric(priv, size=output_data_shape)
        np.savez_compressed('test_data.npz', Y_DV=data_DV, noisy_data_DV=noisy_data_DV, phi_KV=phi_KV, theta_DK=theta_DK, mu_DV=poisson_priors_DV)

    assert(poisson_priors_DV.shape == (n_docs, n_words))
    bpptf_model = BPPTF(n_modes=2, n_components=rank, verbose=True, max_iter=n_iters, true_mu=poisson_priors_DV, tol=1e-6)
    (new_theta, new_phi) = bpptf_model.fit_transform(noisy_data_DV, priv, version='arithmetic')
    new_mu = parafac((new_theta, new_phi))

    np.savez_compressed(out_file, inferred_mu_DV=new_mu, inferred_theta_DK=new_theta, inferred_phi_KV=new_phi.T)

    if n_docs > 100 or n_words > 100:
        return

    naive_model = BPPTF(n_modes=2, n_components=rank, verbose=True, max_iter=n_iters, true_mu=poisson_priors_DV, tol=1e-6)
    (naive_theta, naive_phi) = naive_model.fit_transform(noisy_data_DV, 0.0, version='arithmetic')
    naive_mu = parafac((naive_theta, naive_phi))

    eager_model = BPPTF(n_modes=2, n_components=rank, verbose=True, max_iter=n_iters, true_mu=poisson_priors_DV, tol=1e-6)
    (eager_theta, eager_phi) = eager_model.fit_transform(noisy_data_DV, priv / 0.9, version='arithmetic')
    eager_mu = parafac((eager_theta, eager_phi))

    sns.set(context='poster', style='white', font='serif')
    data_max = np.max(noisy_data_DV)
    kwargs = {'cmap': 'bwr', 'vmin': -data_max, 'vmax': data_max, 'square': True, 'cbar': False, 'xticklabels': False, 'yticklabels': False}
    plt.figure(figsize=(22,17))

    plt.subplot(2, 3, 1)
    sns.heatmap(poisson_priors_DV, **kwargs)
    plt.title('(a) True parameters')
    plt.subplot(2, 3, 2)
    sns.heatmap(data_DV, **kwargs)
    plt.title('(b) Actual data')
    plt.subplot(2, 3, 3)
    sns.heatmap(noisy_data_DV, **kwargs)
    plt.title('(c) Observed data')
    plt.subplot(2, 3, 4)
    sns.heatmap(naive_mu, **kwargs)
    plt.title('(d) Naive method, \nmae = {:.3f}'.format(
        np.mean(np.abs(naive_mu - poisson_priors_DV))))
    plt.subplot(2, 3, 5)
    sns.heatmap(new_mu, **kwargs)
    plt.title('(e) Our method, \nmae = {:.3f}'.format(
        np.mean(np.abs(new_mu - poisson_priors_DV))))
    plt.subplot(2, 3, 6)
    sns.heatmap(np.abs(eager_mu - poisson_priors_DV), **kwargs)
    plt.title('(f) Overestimated method, \nmae = {:.3f}'.format(
        np.mean(np.abs(eager_mu - poisson_priors_DV))))

    plt.savefig('test_output.pdf', bbox_inches='tight')

if __name__ == '__main__':
    n_docs = 50
    n_words = 50
    alpha = 0.5
    beta = 1
    rank = 5
    priv = float(sys.argv[1])
    outfile = sys.argv[2]

    main(n_docs, n_words, alpha, beta, rank, priv, n_iters=100, out_file=outfile)
