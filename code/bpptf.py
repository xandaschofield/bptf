"""
Bayesian Poisson tensor factorization with variational inference.

TODO: make this work for noisy data
- add a function for adding noise
- add data structures for representing latent noise:
    - lam (shape parameters for drawing) 2 x *shape(data)
    - g (noise samples) 2 x *shape(data)
- add inference for lambdas and gs based on computed Poisson expectations
  with the existing data (in theta_E_DK_M), implementing appendix E
    - use delta method for approximating the intractible expectations
"""
import time
import numpy as np
import numpy.random as rn
#pylint: disable=E0611
from scipy.special import gammaln, iv, psi
import sktensor as skt
from sklearn.base import BaseEstimator, TransformerMixin

from pathlib import Path as path
from argparse import ArgumentParser
from utils import is_binary, parafac, preprocess, serialize_bptf, sp_uttkrp


def _gamma_bound_term(pa, pb, qa, qb):
    return gammaln(qa) - pa * np.log(qb) + \
        (pa - qa) * psi(qa) + qa * (1 - pb / qb)


class BPPTF(BaseEstimator, TransformerMixin):
    def __init__(self, n_modes=4, n_components=100,  max_iter=200, tol=0.0001,
                 smoothness=100, verbose=True, alpha=0.1, debug=False, true_mu=None,
                 beta_burnin=0):
        """Bayesian Private Poisson Tensor Factorization.

        Arguments
        ---------

        n_modes : int
            Number of latent factor matrices (should match the order of the
            input tensor)

        n_components : int
            Number of latent components, K in the paper

        max_iter : int
            Maximum number of iterations of inference to perform if no iteration produces an ELBO difference of less than tol.

        tol : float
            Threshold of change in ELBO below which iteration stops.

        smoothness : int
            Describes how smoothed the draws from the gamma distribution for
            initialization are; higher means more smoothing.

        verbose : bool
            Print out information each iteration about timing and the ELBO.

        alpha : float
            The shape parameter for the Gamma prior on all the latent factors.

        debug : bool
            Whether or not to run tests on the code, fixing parameters and
            testing the ELBO.

        true_mu : np.array
            If data was generated synthetically, provides the true value of the Poisson priors.

        beta_burnin : int
            Number of iterations before beta starts to be inferred.
        """

        self.n_modes = n_modes
        self.n_components = n_components
        self.min_iter = 1
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.verbose = verbose
        self.debug = debug

        self.alpha = alpha                                      # shape hyperparameter
        self.beta_M = np.ones(self.n_modes, dtype=float)        # rate hyperparameter (inferred)
        self.beta_burnin = beta_burnin

        # Initialize the list of variational parameter matrices and
        # expectations for latent factors. Because these matrices have
        # different shapes for each mode, these are initialized as arrays
        # of dim x K arrays for whatever dimension dim exists for that factor.
        self.theta_shp_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
        self.theta_rte_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates

        self.theta_E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
        self.theta_V_DK_M = np.empty(self.n_modes, dtype=object)      # variances
        self.theta_G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations

        # Inference cache
        self.sum_theta_E_MK = np.empty((self.n_modes, self.n_components), dtype=float)
        self.nz_recon_I = None

        # For synthetic testing
        self.true_mu = true_mu

    def _reconstruct_nz(self, subs_I_M):
        """Computes the reconstruction for only non-zero entries.

        Arguments
        ---------

        subs_I_M : 2d array-like (n_modes x n_nonzeros)
            Lists the subscripts for each of the non-zero entries in the data.
            Used to index into the nonzero elements in the data arrays.
        """
        I = subs_I_M[0].size
        K = self.n_components
        nz_recon_IK = np.ones((I, K))
        for m in xrange(self.n_modes):
            nz_recon_IK *= self.theta_G_DK_M[m][subs_I_M[m], :]
        self.nz_recon_I = nz_recon_IK.sum(axis=1)
        return self.nz_recon_I

    def _test_elbo(self, data):
        """Copies code from pmf.py.  Used for debugging."""
        raise NotImplementedError

        # assert data.ndim == 2
        # if isinstance(data, skt.sptensor):
        #     X = data.toarray()
        # else:
        #     X = np.array(data)
        # Et = self.theta_E_DK_M[0]
        # Eb = self.theta_E_DK_M[1].T
        # Elogt = np.log(self.theta_G_DK_M[0])
        # Elogb = np.log(self.theta_G_DK_M[1].T)
        # gamma_t = self.theta_shp_DK_M[0]
        # gamma_b = self.theta_shp_DK_M[1].T
        # rho_t = self.theta_rte_DK_M[0]
        # rho_b = self.theta_rte_DK_M[1].T
        # Z = np.dot(np.exp(Elogt), np.exp(Elogb))
        # bound = np.sum(X * np.log(Z) - Et.dot(Eb))
        # a = self.alpha
        # c = self.beta_M[0]
        # bound += _gamma_bound_term(a, a * c, gamma_t, rho_t).sum()
        # bound += self.n_components * X.shape[0] * a * np.log(c)
        # bound += _gamma_bound_term(a, a, gamma_b, rho_b).sum()
        # return bound

    def _elbo(self, data, mask=None):
        """Computes the Evidence Lower Bound (ELBO).

        Arguments
        ---------

        data : tensor-like
            Input data against which the ELBO is tested.

        mask : tensor-like containing 1s and 0s
            Masks sections of the data, with 1s indicating data regions that
            are kept.
        """
        # raise NotImplementedError
        if mask is None:
            uttkrp_K = self.sum_theta_E_MK.prod(axis=0)
        elif isinstance(mask, skt.dtensor):
            uttkrp_DK = mask.uttkrp(self.theta_E_DK_M, 0)
            uttkrp_K = (self.theta_E_DK_M[0] * uttkrp_DK).sum(axis=0)
        elif isinstance(mask, skt.sptensor):
            uttkrp_DK = sp_uttkrp(mask.vals, mask.subs, 0, self.theta_E_DK_M)
            uttkrp_K = (self.theta_E_DK_M[0] * uttkrp_DK).sum(axis=0)

        bound = -uttkrp_K.sum()

        if isinstance(data, skt.dtensor):
            subs_I_M = data.nonzero()
            vals_I = data[subs_I_M]
        elif isinstance(data, skt.sptensor):
            subs_I_M = data.subs
            vals_I = data.vals
        nz_recon_I = self._reconstruct_nz(subs_I_M)

        bound += (vals_I * np.log(nz_recon_I)).sum()

        K = self.n_components
        for m in range(self.n_modes):
            bound += _gamma_bound_term(pa=self.alpha,
                                       pb=self.alpha * self.beta_M[m],
                                       qa=self.theta_shp_DK_M[m],
                                       qb=self.theta_rte_DK_M[m]).sum()
            bound += K * self.mode_dims[m] * self.alpha * np.log(self.beta_M[m])
        return bound

    def stochastic_elbo(self):
        # - sample lambdas from Q distribution, then compute
        # ln p and ln q for lambdas and gs (using fixed m)
        # - ydvks have to be poisson sampled one dv at a time to avoid
        # making the full mu_dvk prior tensor
        # - thetas can be computed directly from the previous formula
        pass

    def _init_all_components(self, mode_dims):
        assert len(mode_dims) == self.n_modes
        self.mode_dims = mode_dims
        for m, D in enumerate(mode_dims):
            self._init_component(m, D)
        self.mu_G_DIMS = parafac(self.theta_E_DK_M)

    def _init_component(self, m, dim):
        assert self.mode_dims[m] == dim
        K = self.n_components
        s = self.smoothness
        if not self.debug:
            theta_shp_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
            theta_rte_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
        else:   # When debugging, deterministically set shapes and
                # rates for our variational dists to the smoothness
                # parameter.
            theta_shp_DK = s * np.ones((dim, K))
            theta_rte_DK = s * np.ones((dim, K))
        self.theta_shp_DK_M[m] = theta_shp_DK
        self.theta_rte_DK_M[m] = theta_rte_DK
        self.theta_E_DK_M[m] = theta_shp_DK / theta_rte_DK
        self.theta_V_DK_M[m] = theta_shp_DK / np.square(theta_rte_DK)
        self.sum_theta_E_MK[m, :] = self.theta_E_DK_M[m].sum(axis=0)
        self.theta_G_DK_M[m] = np.exp(psi(theta_shp_DK) - np.log(theta_rte_DK))

        # We can set beta deterministically - it is the inverse of the
        # arithmetic expectation of that latent factor.
        if m == 0 or not self.debug:
            self.beta_M[m] = 1. / self.theta_E_DK_M[m].mean()

    def _init_privacy_variables(self, mode_dims, priv):
        self.y_E_DIMS = np.empty(mode_dims, dtype=int)
        self.min_DIMS = np.random.poisson(lam=priv, size=mode_dims)
        if not self.debug:
            self.lam_shp_2DIMS = 1 + rn.gamma(1, priv/(1 - priv), size=(2,) + mode_dims)
        else:
            self.lam_shp_2DIMS = np.ones((2,) + mode_dims)

        # As it turns out, we never have to update this variational parameter,
        # as it only depends on the observed privacy level.
        self.lam_rte_DIMS = np.ones(mode_dims) / priv

    def _check_component(self, m):
        assert np.isfinite(self.theta_E_DK_M[m]).all()
        assert np.isfinite(self.theta_G_DK_M[m]).all()
        assert np.isfinite(self.theta_shp_DK_M[m]).all()
        assert np.isfinite(self.theta_rte_DK_M[m]).all()

    def _check_privacy_params(self):
        assert np.isfinite(self.y_E_DIMS).all()
        assert np.isfinite(self.min_DIMS).all()
        assert np.isfinite(self.lam_shp_2DIMS).all()
        assert np.isfinite(self.lam_rte_DIMS).all()

    def _update_theta_gamma(self, m):
        subs_I_M = np.where(self.y_E_DIMS > 1e-4)
        y_spt_DIMS = skt.sptensor(
            subs_I_M,
            self.y_E_DIMS[subs_I_M],
            shape=self.y_E_DIMS.shape,
            dtype=float
        )
        tmp_DIMS = y_spt_DIMS.vals / self._reconstruct_nz(y_spt_DIMS.subs)
        uttkrp_nonzero_DK = sp_uttkrp(tmp_DIMS, y_spt_DIMS.subs, m, self.theta_G_DK_M)

        self.theta_shp_DK_M[m][:, :] = self.alpha + self.theta_G_DK_M[m] * uttkrp_nonzero_DK

    def _update_theta_delta(self, m, mask=None):
        if mask is None:
            self.sum_theta_E_MK[m, :] = 1.
            uttkrp_DK = self.sum_theta_E_MK.prod(axis=0)
        else:
            uttkrp_DK = mask.uttkrp(self.theta_E_DK_M, m)

        self.theta_rte_DK_M[m][:, :] = self.alpha * self.beta_M[m] + uttkrp_DK

    def _update_y(self, mask=None):
        lam_pos_G_DIMS = np.exp(psi(self.lam_shp_2DIMS[0]) - np.log(self.lam_rte_DIMS))
        self.y_E_DIMS = self.y_pos_E_DIMS * self.mu_G_DIMS / (self.mu_G_DIMS + lam_pos_G_DIMS)

    def _update_lam_shp(self, mask=None):
        self.lam_shp_2DIMS[0] = self.g_pos_E_DIMS + 1
        self.lam_shp_2DIMS[1] = self.g_neg_E_DIMS + 1

    def _update_mu(self, mask=None):
        parafac_theta = parafac(self.theta_E_DK_M)
        first_order_term = np.log(parafac_theta)
        second_order_term = -parafac(self.theta_V_DK_M) / (2 * np.square(parafac_theta))
        self.mu_G_DIMS = np.exp(first_order_term + second_order_term)
        if mask is not None:
            self.mu_G_DIMS *= mask

    def _update_mins(self, mask=None):
        """Updates the minimum of g- and y+ (m) sampled from a Bessel.
        We estimate this using a point estimate of the mode of the
        Bessel distribution instead of the mean."""
        # Compute geometric expectation and variances
        mu_V_DIMS = (
            parafac(np.square(self.theta_E_DK_M) + self.theta_V_DK_M)
            - parafac(np.square(self.theta_E_DK_M)))
                # Compute geometric expectation of mu (based on theta)
        lam_G_2DIMS = np.exp(psi(self.lam_shp_2DIMS) - np.log(self.lam_rte_DIMS))
        lam_pos_V_DIMS = self.lam_shp_2DIMS[0] / np.square(self.lam_rte_DIMS)

        # Approximating a geometric expectation using the delta method:
        # E[f(x)] ~= f(E[x]) + f"(E[x])V[x]/2
        lam_pos_plus_mu_G_DIMS = (
            # f(E[x])
            (self.mu_G_DIMS + lam_G_2DIMS[0])
            # f"(E[x])V[x]/2
            * np.exp(
                -(mu_V_DIMS + lam_pos_V_DIMS)
                / (2 * np.square(self.mu_G_DIMS + lam_G_2DIMS[0]))
            ))

        # Bessel parameters a and nu
        a_DIMS = 2 * np.sqrt(lam_G_2DIMS[1] * lam_pos_plus_mu_G_DIMS)
        if isinstance(self.data_DIMS, skt.dtensor):
            nu_DIMS = self.data_DIMS.copy()
        else:
            nu_DIMS = self.data_DIMS.toarray()

        # Formula for the mode of the Bessel
        self.min_DIMS = np.floor_divide(
            np.sqrt(np.square(a_DIMS) + np.square(nu_DIMS)) - nu_DIMS,
            2
        )

        if mask is not None:
            self.min_DIMS *= mask

    def _update_cache(self, m):
        theta_shp_DK = self.theta_shp_DK_M[m]
        theta_rte_DK = self.theta_rte_DK_M[m]
        self.theta_E_DK_M[m] = theta_shp_DK / theta_rte_DK
        self.theta_V_DK_M[m] = theta_shp_DK / np.square(theta_rte_DK)
        self.sum_theta_E_MK[m, :] = self.theta_E_DK_M[m].sum(axis=0)
        self.theta_G_DK_M[m] = np.exp(psi(theta_shp_DK) - np.log(theta_rte_DK))

    def _update_beta(self, m):
        self.beta_M[m] = 1. / self.theta_E_DK_M[m].mean()

    def _print_arr_stats(self, arr, name):
        print "{}: mean {}, std {}, row mean {}".format(
            name,
            arr.mean(),
            arr.std(),
            arr.sum(axis=tuple(range(1, len(arr.shape)))).mean()
        )

    def _update(self, data, priv=None, mask=None, modes=None):
        if modes is not None:
            modes = list(set(modes))
        else:
            modes = range(self.n_modes)
        assert all(m in range(self.n_modes) for m in modes)

        for m in range(self.n_modes):
            if m not in modes:
                self._clamp_component(m)

        if priv == 0:
            self.y_E_DIMS = data
            if isinstance(data, skt.sptensor):
                self.y_E_DIMS = self.y_E_DIMS.toarray()

        if not self.debug and self.true_mu is not None:
            mu_diff = np.abs(self.mu_G_DIMS - self.true_mu).mean()
        else:
            mu_diff = -1

        if self.verbose:
            print('ITERATION %d:\t'\
                  'Time: %f\t'\
                  'Objective: %.2f\t'\
                  'Change: %.5e\t'\
                % (0, 0.0, mu_diff, np.nan))

        for itn in range(self.max_iter):
            s = time.time()
            if priv is not None and priv > 0:
                self._update_mins(mask=mask)
                self._update_y(mask=mask)
                self._update_lam_shp(mask=mask)
                if self.verbose:
                    print "Privacy values:"
                    self._print_arr_stats(self.min_DIMS, "m")
                    self._print_arr_stats(self.y_E_DIMS, "y")
                    self._print_arr_stats(self.lam_shp_2DIMS[0], "lam+")
                    self._print_arr_stats(self.lam_shp_2DIMS[1], "lam-")

            self._update_mu(mask=mask)
            for m in modes:
                self._update_theta_gamma(m)
                self._update_theta_delta(m, mask)
                self._update_cache(m)
                if self.verbose:
                    self._print_arr_stats(self.theta_E_DK_M[m], 'theta_%d' % m)

                if (m == 0 or not self.debug) and itn >= self.beta_burnin:
                    self._update_beta(m)  # must come after cache update!
                self._check_component(m)

            if not self.debug and self.true_mu is not None:
                old_mu_diff = mu_diff
                mu_diff = np.abs(self.mu_G_DIMS - self.true_mu).mean()
                delta = (old_mu_diff - mu_diff)
            else:
                delta = np.infty
            e = time.time() - s
            if self.verbose:
                print('ITERATION %d:\t'\
                      'Time: %f\t'\
                      'Objective: %.2f\t'\
                      'Change: %.5e\t'\
                      % (itn+1, e, mu_diff, delta))

            if abs(delta) < self.tol:
                break

    def set_component(self, m, theta_E_DK, theta_G_DK, theta_shp_DK, theta_rte_DK):
        assert theta_E_DK.shape[1] == self.n_components
        self.theta_E_DK_M[m] = theta_E_DK.copy()
        self.sum_theta_E_MK[m, :] = theta_E_DK.sum(axis=0)
        self.theta_G_DK_M[m] = theta_G_DK.copy()
        self.theta_shp_DK_M[m] = theta_shp_DK.copy()
        self.theta_rte_DK_M[m] = theta_rte_DK.copy()
        self.beta_M[m] = 1. / theta_E_DK.mean()

    def _clamp_component(self, m, version='geometric'):
        """Make a component a constant.
        This amounts to setting the expectations under the
        Q-distribution to be equal to a single point estimate.
        """
        assert (version == 'geometric') or (version == 'arithmetic')
        if version == 'geometric':
            self.theta_E_DK_M[m][:, :] = self.theta_G_DK_M[m]
        else:
            self.theta_G_DK_M[m][:, :] = self.theta_E_DK_M[m]
        self.sum_theta_E_MK[m, :] = self.theta_E_DK_M[m].sum(axis=0)
        self.beta_M[m] = 1. / self.theta_E_DK_M[m].mean()

    def set_component_like(self, m, model, subs_D=None):
        assert model.n_modes == self.n_modes
        assert model.n_components == self.n_components
        D = model.theta_E_DK_M[m].shape[0]
        if subs_D is None:
            subs_D = np.arange(D)
        assert min(subs_D) >= 0 and max(subs_D) < D
        theta_E_DK = model.theta_E_DK_M[m][subs_D, :].copy()
        theta_G_DK = model.theta_G_DK_M[m][subs_D, :].copy()
        theta_shp_DK = model.theta_shp_DK_M[m][subs_D, :].copy()
        theta_rte_DK = model.theta_rte_DK_M[m][subs_D, :].copy()
        self.set_component(m, theta_E_DK,   theta_G_DK, theta_shp_DK, theta_rte_DK)

    def fit(self, data, priv, mask=None):
        assert data.ndim == self.n_modes
        data = preprocess(data)
        if isinstance(data, skt.dtensor):
            self.data_DIMS = data.copy()
        else:
            self.data_DIMS = skt.sptensor(
                tuple((np.copy(ds) for ds in data.subs)),
                data.vals.copy())

        if mask is not None:
            mask = preprocess(mask)
            assert data.shape == mask.shape
            assert is_binary(mask)
            assert np.issubdtype(mask.dtype, int)
        self._init_all_components(data.shape)
        if priv > 0:
            self._init_privacy_variables(data.shape, priv)
        self._update(data, priv=priv, mask=mask)
        return self

    @property
    def y_pos_E_DIMS(self):
        if isinstance(self.data_DIMS, skt.dtensor):
            return self.min_DIMS + self.data_DIMS.clip(0, None)
        else:
            return self.min_DIMS + self.data_DIMS.toarray().clip(0, None)

    @property
    def g_neg_E_DIMS(self):
        if isinstance(self.data_DIMS, skt.dtensor):
            return self.min_DIMS - self.data_DIMS.clip(None, 0)
        else:
            return self.min_DIMS - self.data_DIMS.toarray().clip(None, 0)

    @property
    def g_pos_E_DIMS(self):
        return self.y_pos_E_DIMS - self.y_E_DIMS

    def transform(self, modes, data, mask=None, version='geometric'):
        """Transform new data given a pre-trained model."""
        assert all(m in range(self.n_modes) for m in modes)
        assert (version == 'geometric') or (version == 'arithmetic')

        assert data.ndim == self.n_modes
        data = preprocess(data)
        if mask is not None:
            mask = preprocess(mask)
            assert data.shape == mask.shape
            assert is_binary(mask)
            assert np.issubdtype(mask.dtype, int)
        self.mode_dims = data.shape
        for m, D in enumerate(self.mode_dims):
            if m not in modes:
                if self.theta_E_DK_M[m].shape[0] != D:
                    raise ValueError('Pre-trained components dont match new data.')
            else:
                self._init_component(m, D)
        self._update(data, mask=mask, modes=modes)

        if version == 'geometric':
            return [self.theta_G_DK_M[m] for m in modes]
        elif version == 'arithmetic':
            return [self.theta_E_DK_M[m] for m in modes]

    def fit_transform(self, data, priv, mask=None, modes=None, version='geometric'):
        if modes is None:
            modes = range(self.n_modes)
        else:
            assert all(m in range(self.n_modes) for m in modes)
        assert (version == 'geometric') or (version == 'arithmetic')

        self.fit(data, priv, mask=mask)

        if version == 'geometric':
            return [self.theta_G_DK_M[m] for m in modes]
        elif version == 'arithmetic':
            return [self.theta_E_DK_M[m] for m in modes]

    def reconstruct(self, mask=None, version='geometric', drop_diag=False):
        """Reconstruct data using point estimates of latent factors.
        Currently supported only up to 5-way tensors.
        """
        assert (version == 'geometric') or (version == 'arithmetic')
        if version == 'geometric':
            tmp = [G_DK.copy() for G_DK in self.theta_G_DK_M]
        elif version == 'arithmetic':
            tmp = [E_DK.copy() for E_DK in self.theta_E_DK_M]

        Y_pred = parafac(tmp)
        if drop_diag:
            diag_idx = np.identity(Y_pred.shape[0]).astype(bool)
            Y_pred[diag_idx] = 0
        return Y_pred


def main():
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=path, required=True)
    p.add_argument('-priv', type=float, default=0)
    p.add_argument('-o', '--out', type=path, required=True)
    p.add_argument('-m', '--mask', type=path, default=None)
    p.add_argument('-k', '--n_components', type=int, required=True)
    p.add_argument('-n', '--max_iter', type=int, default=200)
    p.add_argument('-t', '--tol', type=float, default=1e-4)
    p.add_argument('-s', '--smoothness', type=int, default=100)
    p.add_argument('-a', '--alpha', type=float, default=0.1)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('--debug', action="store_true", default=False)
    args = p.parse_args()

    args.out.makedirs_p()
    assert args.data.exists() and args.out.exists()
    if args.data.ext == '.npz':
        data_dict = np.load(args.data)
        if 'data' in data_dict.files:
            data = data_dict['data']
        elif 'Y' in data_dict.files:
            data = data_dict['Y']
        if data.dtype == 'object':
            assert data.size == 1
            data = data[0]
    else:
        data = np.load(args.data)

    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(data, vt) for vt in valid_types)

    mask = None
    if args.mask is not None:
        if args.mask.ext == '.npz':
            mask = np.load(args.mask)['mask']
            if mask.dtype == 'object':
                assert mask.size == 1
                mask = mask[0]
        else:
            mask = np.load(args.mask)

        assert any(isinstance(mask, vt) for vt in valid_types)
        assert mask.shape == data.shape

    bptf = BPPTF(n_modes=data.ndim,
                n_components=args.n_components,
                max_iter=args.max_iter,
                tol=args.tol,
                smoothness=args.smoothness,
                verbose=args.verbose,
                alpha=args.alpha,
                debug=args.debug)

    bptf.fit(data, args.priv, mask=mask)
    serialize_bptf(bptf, args.out, num=None, desc='trained_model')


if __name__ == '__main__':
    main()
