import os

from ipdb import launch_ipdb_on_exception
import numpy as np

from bpptf import BPPTF


def two_sided_geometric(p, size=()):
    """Generate 2-sided geometric noise"""
    if p == 0:
        return np.zeros(size)

    rte = (1 - p) / p
    Lambda_2 = np.random.gamma(1, 1. / rte, size=(2,) + size)
    G_2 = np.random.poisson(Lambda_2)
    return G_2[0] - G_2[1]


def main(n_docs, n_words, alpha, beta, rank, priv):
    try:
        dat_file = np.load('test_data.npz')
        data_DV = dat_file['Y_DV']
        assert(Y_DV.shape == (n_docs, n_words))
        noisy_data_DV = dat_file['noisy_data_DV']
        phi_KV = dat_file['phi_KV']
        assert(phi_KV.shape == (rank, n_words))
        theta_DK = dat_file['theta_DK']
        poisson_priors_DV = np.dot(theta_DK, phi_KV)
    except:
        output_data_shape = (n_docs, n_words)
        theta_DK = np.random.gamma(alpha, beta, (n_docs, rank))
        phi_KV = np.random.gamma(alpha, beta, (rank, n_words)) 
        poisson_priors_DV = np.dot(theta_DK, phi_KV)
        # Sample true data and noisy data
        data_DV = np.random.poisson(poisson_priors_DV, output_data_shape)
        noisy_data_DV = data_DV + two_sided_geometric(priv, size=output_data_shape)
        np.savez_compressed('test_data.npz', Y_DV=data_DV, noisy_data_DV=noisy_data_DV, phi_KV=phi_KV, theta_DK=theta_DK, mu_DV=np.dot(theta_DK, phi_KV))

    bpptf_model = BPPTF(n_modes=2, n_components=rank, verbose=True, max_iter=20, true_mu=poisson_priors_DV)
    (theta, phi) = bpptf_model.fit_transform(noisy_data_DV, priv)
    mu = theta.dot(phi.T)
    return np.mean(np.abs(data_DV - mu))


if __name__ == '__main__':
    n_docs = 20
    n_words = 20
    alpha = 0.2
    beta = 1.
    rank = 5
    priv = 0.0
    with launch_ipdb_on_exception():
        main(n_docs, n_words, alpha, beta, rank, priv)