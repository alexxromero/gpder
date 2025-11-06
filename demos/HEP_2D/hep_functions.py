import numpy as np
from scipy.special import expit


def at_least_2d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 0:
        return X.reshape(1, 1)
    elif X.ndim == 1:
        return X.reshape(1, -1)
    else:
        return X


def sigmoid(x, c=0, a=1):
    return expit(a * (x - c))


def der_sigmoid(x, c=0, a=1):
    return a * (1.0 - sigmoid(x, c, a)) * sigmoid(x, c, a)


def MET(nu_J1, nu_J23, threeM):
    """Missing transverse energy (MET)."""
    # J2 and J3 share the same jet energy scale
    nu_J_arr = np.array((nu_J1, nu_J23, nu_J23)).T
    pT_x = threeM[:, :, 0] * np.cos(threeM[:, :, 2]) / nu_J_arr
    pT_y = threeM[:, :, 0] * np.sin(threeM[:, :, 2]) / nu_J_arr
    return np.sqrt(np.sum(pT_x, axis=1) ** 2 + np.sum(pT_y, axis=1) ** 2)


def efficiency(X, events_threeM, pT_cut=200, MET_cut=50, normalized=True):
    """Efficiency of the three-jet events with an MET of less than MET_cut GeVs.
    X is the array of nuisance parameters. The first column contains the
    nuisance parameter for the first (hardest) jet, and the second column the
    nuisance parameter for the second and third jets.
    """
    X = np.atleast_2d(X)
    nu_J1, nu_J23 = X[:, 0], X[:, 1]
    ix_pT_cut = (events_threeM[:, 0, 0] / nu_J1 > pT_cut) & (
        events_threeM[:, 1, 0] / nu_J23 < pT_cut
    )
    MET_vals = MET(nu_J1, nu_J23, events_threeM[ix_pT_cut])
    ix_MET_cut = MET_vals < MET_cut
    eff = np.sum(ix_MET_cut) / np.sum(ix_pT_cut)
    if normalized:
        # normalized by (nu_J1, nu_J23) = (1, 1)
        ix_pT_cut_norm = (events_threeM[:, 0, 0] > pT_cut) & (events_threeM[:, 1, 0] < pT_cut)
        MET_vals_norm = MET(1, 1, events_threeM[ix_pT_cut_norm])
        ix_MET_cut_norm = MET_vals_norm < MET_cut
        norm = np.sum(ix_MET_cut_norm) / np.sum(ix_pT_cut_norm)
        eff /= norm
    return eff


def der_MET(nu_J1, nu_J23, threeM):
    """Derivative of the missing transverse energy (MET) with respect to
    the nuiance parameters (X).
    """
    nu_J_arr = np.array((nu_J1, nu_J23, nu_J23)).T
    pT_x = threeM[:, :, 0] * np.cos(threeM[:, :, 2]) / nu_J_arr
    pT_y = threeM[:, :, 0] * np.sin(threeM[:, :, 2]) / nu_J_arr
    dMET = 1.0 / np.sqrt(np.sum(pT_x, axis=1) ** 2 + np.sum(pT_y, axis=1) ** 2)
    # chain rule terms
    dpT_x_dnu_J = np.sum(pT_x, axis=1).reshape(-1, 1) * threeM[:, :, 0] * np.cos(threeM[:, :, 2])
    dpT_x_dnu_J *= -1 / nu_J_arr**2
    dpT_y_dnu_J = np.sum(pT_y, axis=1).reshape(-1, 1) * threeM[:, :, 0] * np.sin(threeM[:, :, 2])
    dpT_y_dnu_J *= -1 / nu_J_arr**2
    # full derivative
    dMET_dnu_J_arr = dMET.reshape(-1, 1) * (dpT_x_dnu_J + dpT_y_dnu_J)
    return np.stack((dMET_dnu_J_arr[:, 0], np.sum(dMET_dnu_J_arr[:, 1:], axis=1)), axis=-1)


def der_efficiency(X, events_threeM, pT_cut=200, MET_cut=50, normalized=True, a=1):
    """Derivative of the efficiency with respect to the nuiance parameters (X).
    Sigmoids are used to replace the pT and MET cuts.
    """
    X = at_least_2d(X)
    nu_J1, nu_J23 = X[:, 0], X[:, 1]
    # replacing pT cuts with sigmoids
    pT_sig = sigmoid(events_threeM[:, 0, 0] / nu_J1, c=pT_cut, a=a).reshape(-1, 1)
    pT_sig *= sigmoid(events_threeM[:, 1, 0] / nu_J23, c=pT_cut, a=-a).reshape(-1, 1)
    # with respect to nu_J1
    dpT_sig_dnu_J1 = der_sigmoid(events_threeM[:, 0, 0] / nu_J1, c=pT_cut, a=a)
    dpT_sig_dnu_J1 *= sigmoid(events_threeM[:, 1, 0] / nu_J23, c=pT_cut, a=-a)
    dpT_sig_dnu_J1 *= -1 * events_threeM[:, 0, 0] / nu_J1**2
    # with respect to nu_J23
    dpT_sig_dnu_J23 = sigmoid(events_threeM[:, 0, 0] / nu_J1, c=pT_cut, a=a)
    dpT_sig_dnu_J23 *= der_sigmoid(events_threeM[:, 1, 0] / nu_J23, c=pT_cut, a=-a)
    dpT_sig_dnu_J23 *= -1 * events_threeM[:, 1, 0] / nu_J23**2
    # stacking them
    dpT_sig_dnuJ = np.stack((dpT_sig_dnu_J1, dpT_sig_dnu_J23), axis=1)

    # replacing MET cut with a sigmoid
    MET_vals = MET(nu_J1, nu_J23, events_threeM)
    MET_sig = sigmoid(MET_vals, c=MET_cut, a=-a).reshape(-1, 1)
    dMET_sig_dnuJ = -1 * der_sigmoid(MET_vals, c=MET_cut, a=a).reshape(-1, 1)
    dMET_dnuJ = der_MET(nu_J1, nu_J23, events_threeM)

    # combining the MET and pT sigmoids - and their chain rule terms
    MET_pT_sigmoids = pT_sig * MET_sig  # both cuts combined
    dMET_pT_sigmoids_dnuJ = pT_sig * dMET_sig_dnuJ * dMET_dnuJ + dpT_sig_dnuJ * MET_sig

    # final derivative of the efficiency
    deff_dnuJ = np.sum(dMET_pT_sigmoids_dnuJ, axis=0) / np.sum(pT_sig)
    deff_dnuJ -= np.sum(MET_pT_sigmoids) / np.sum(pT_sig) ** 2 * np.sum(dpT_sig_dnuJ, axis=0)

    if normalized:
        # using the regular efficiency values for the normalization.
        ix_pT_cut_norm = (events_threeM[:, 0, 0] > pT_cut) & (events_threeM[:, 1, 0] < pT_cut)
        MET_vals_norm = MET(1, 1, events_threeM[ix_pT_cut_norm])
        ix_MET_cut_norm = MET_vals_norm < MET_cut
        norm = np.sum(ix_MET_cut_norm) / np.sum(ix_pT_cut_norm)
        deff_dnuJ /= norm
    return deff_dnuJ
