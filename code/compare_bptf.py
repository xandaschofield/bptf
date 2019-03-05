from ipdb import launch_ipdb_on_exception
import numpy as np
import sktensor as skt

from bptf import BPTF
from bpptf import BPPTF
from utils import parafac, preprocess


def two_sided_geometric(p, size=()):
    """Generate 2-sided geometric noise"""
    if p == 0:
        return np.zeros(size)

    rte = (1 - p) / p
    Lambda_2 = np.random.gamma(1, 1. / rte, size=(2,) + size)
    G_2 = np.random.poisson(Lambda_2)
    return G_2[0] - G_2[1]


def check_equal(bpptf_model, bptf_model, m):
    def check_two_arrs(arr_1, arr_2, name):
        if not np.array_equal(arr_1, arr_2):
            print "{} diverged: bpptf mean {}, bptf mean {}".format(
                name,
                arr_1.mean(),
                arr_2.mean()
            )
            return False
        else:
            return True
        
    ok = [
        check_two_arrs(bpptf_model.theta_shp_DK_M[m], bptf_model.gamma_DK_M[m], 'gamma'),
        check_two_arrs(bpptf_model.theta_rte_DK_M[m], bptf_model.delta_DK_M[m], 'delta'),
        check_two_arrs(bpptf_model.theta_E_DK_M[m], bptf_model.E_DK_M[m], 'E'),
        check_two_arrs(bpptf_model.theta_G_DK_M[m], bptf_model.G_DK_M[m], 'G'),
        bpptf_model.beta_M[m] == bptf_model.beta_M[m]
    ]
    if not np.all(ok):
        raise RuntimeError("Values diverged!")

def main(n_top_words, alpha, beta, rank, priv, n_iters=200):
    # output_data_shape = (n_docs, n_words)
    # theta_DK = np.random.gamma(alpha, beta, (n_docs, rank))
    # phi_KV = np.random.gamma(alpha, beta, (rank, n_words)) 
    # poisson_priors_DV = parafac((theta_DK, phi_KV.T))
    # data_DV = np.random.poisson(poisson_priors_DV, output_data_shape)
    
    with np.load('sotu_years.npz') as dat_file:
        data_DV = dat_file['Y_DV']
        vocab = dat_file['types_V']
    n_docs, n_words = data_DV.shape
    bpptf_model = BPPTF(n_modes=2, n_components=rank, verbose=True, max_iter=1)
    bptf_model = BPTF(n_modes=2, n_components=rank, verbose=True, max_iter=1)
    
    # initialize both models
    modes = (0, 1)
    data_usable = preprocess(data_DV)
    if isinstance(data_usable, skt.dtensor):
        bpptf_model.data_DIMS = data_usable.copy()
    else:
        bpptf_model.data_DIMS = skt.sptensor(
            tuple((np.copy(ds) for ds in data_usable.subs)),
            data_usable.vals.copy())

    bpptf_model._init_all_components(data_usable.shape)
    bptf_model._init_all_components(data_usable.shape) 
    
    bpptf_model.y_E_DIMS = data_usable
    if isinstance(data_usable, skt.sptensor):
        bpptf_model.y_E_DIMS = bpptf_model.y_E_DIMS.toarray()
    for i in range(n_iters):
        print i
        for m in modes:
                # check_equal(bpptf_model, bptf_model, m)

                bpptf_model._update_theta_gamma(m)
                bptf_model._update_gamma(m, data_usable)
                # check_equal(bpptf_model, bptf_model, m)

                bpptf_model._update_theta_delta(m, None)
                bptf_model._update_delta(m, None)
                # check_equal(bpptf_model, bptf_model, m)

                bpptf_model._update_cache(m)
                bptf_model._update_cache(m)
                # check_equal(bpptf_model, bptf_model, m)
                
                bpptf_model._update_beta(m)  # must come after cache update!
                bptf_model._update_beta(m)
                # check_equal(bpptf_model, bptf_model, m)

                bpptf_model._check_component(m)
                bptf_model._check_component(m)
                # check_equal(bpptf_model, bptf_model, m)

    print "Old topics"
    new_phi = bptf_model.E_DK_M[1].T
    top_words = np.argpartition(new_phi, n_words - n_top_words)[:,-n_top_words:]
    for topic in xrange(rank):
        top_word_vals = zip(-new_phi[topic, top_words[topic]], vocab[top_words[topic]])
        print topic, ' '.join(['{}'.format(wd) for (_, wd) in sorted(top_word_vals)])
    
    print "\nNew topics"
    new_phi = bpptf_model.theta_E_DK_M[1].T
    top_words = np.argpartition(new_phi, n_words - n_top_words)[:,-n_top_words:]
    for topic in xrange(rank):
        top_word_vals = zip(-new_phi[topic, top_words[topic]], vocab[top_words[topic]])
        print topic, ' '.join(['{}'.format(wd) for (_, wd) in sorted(top_word_vals)])

    

if __name__ == '__main__':
    n_top_words = 20
    alpha = 10
    beta = 1
    rank = 20
    priv = 0
    # priv = 0.367879
    with launch_ipdb_on_exception():
        main(n_top_words, alpha, beta, rank, priv, n_iters=200)