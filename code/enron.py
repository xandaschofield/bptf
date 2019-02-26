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


def main(rank, priv, input_data, output_model, n_top_words=25, max_iter=200):
    dat_file = np.load(input_data)
    data_DV = dat_file['Y_DV']
    vocab = dat_file['types_V']
    output_data_shape = data_DV.shape
    noisy_data_DV = data_DV + two_sided_geometric(priv, size=output_data_shape)

    n_docs, n_words = data_DV.shape

    bpptf_model = BPPTF(n_modes=2, n_components=rank, verbose=True, max_iter=max_iter, true_mu=data_DV)
    (new_theta, new_phi) = bpptf_model.fit_transform(noisy_data_DV, priv)
    new_phi = new_phi.T
    np.savez_compressed(output_model, theta_DK=new_theta, phi_KV=new_phi, alpha=priv)

    print 'Topics:'
    top_words = np.argpartition(new_phi, n_words - n_top_words)[:,-n_top_words:]
    for topic in xrange(rank):
        top_word_vals = zip(-new_phi[topic, top_words[topic]], vocab[top_words[topic]])
        print topic, ' '.join(['{}'.format(wd) for (_, wd) in sorted(top_word_vals)])
        


if __name__ == '__main__':
    data_file = 'enron_8_10_sample.npz'
    out_file = 'enron_test_topic_model.npz'
    rank = 50
    priv = 0.135335
    max_iter=200
    with launch_ipdb_on_exception():
        main(rank, priv, data_file, out_file, max_iter=max_iter)