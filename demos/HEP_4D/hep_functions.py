import numpy as np
from scipy.special import expit


def sigmoid(x, c=0, a=1):
    return expit((x - c) / a)


def der_sigmoid(x, c=0, a=1):
    return (1.0 / a) * (1.0 - sigmoid(x, c, a)) * sigmoid(x, c, a)


def MET(nu_J1, nu_J23, threeM):
    """Missing transverse energy (MET)."""
    # J2 and J3 share the same jet energy scale
    nu_J_arr = np.array((nu_J1, nu_J23, nu_J23)).T
    pT_x = threeM[:, :, 0] * np.cos(threeM[:, :, 2]) / nu_J_arr
    pT_y = threeM[:, :, 0] * np.sin(threeM[:, :, 2]) / nu_J_arr
    return np.sqrt(np.sum(pT_x, axis=1) ** 2 + np.sum(pT_y, axis=1) ** 2)


def efficiency(X, events_threeM, pT_cut=200, MET_cut=50, normalized=True):
    """Efficiency of the events with an MET of less than MET_cut GeVs.
    X is the array of nuisance parameters. The first two columns contains the
    nuisance parameter for the first (hardest) jet for |eta| < 1 and |eta| > 1.
    Likewise, the third and fourth columns the nuisance parameter for the
    second and third jets for |eta| < 1 and |eta| > 1. See paper for more details.
    """
    X = np.atleast_2d(X)
    nu_J1_in, nu_J1_out, nu_J23_in, nu_J23_out = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    # use nu_J1_in where the jet is in the inner detector; |eta_J1| < 1
    # else nu_J1_out
    nu_J1 = np.where(abs(events_threeM[:, 0, 1]) < 1, nu_J1_in, nu_J1_out)
    # similarly, use nu_J23_in where the average |eta| of J2 and J3 is less than 1
    eta_avg = np.average(events_threeM[:, 1:, 1], weights=events_threeM[:, 1:, 0], axis=1)
    nu_J23 = np.where(abs(eta_avg) < 1, nu_J23_in, nu_J23_out)
    # the rest is just like in the 2D case
    ix_pT_cut = (events_threeM[:, 0, 0] / nu_J1 > pT_cut) & (
        events_threeM[:, 1, 0] / nu_J23 < pT_cut
    )
    MET_vals = MET(nu_J1[ix_pT_cut], nu_J23[ix_pT_cut], events_threeM[ix_pT_cut])
    ix_MET_cut = MET_vals < MET_cut
    if normalized:
        # normalized by (nu_J1, nu_J23) = (1, 1)
        ix_pT_cut_norm = (events_threeM[:, 0, 0] > pT_cut) & (events_threeM[:, 1, 0] < pT_cut)
        MET_vals_norm = MET(1, 1, events_threeM[ix_pT_cut_norm])
        ix_MET_cut_norm = MET_vals_norm < MET_cut
        norm = np.sum(ix_MET_cut_norm) / np.sum(ix_pT_cut_norm)
        return np.sum(ix_MET_cut) / np.sum(ix_pT_cut) / norm
    return np.sum(ix_MET_cut) / np.sum(ix_pT_cut)


def der_MET(nu_J1, nu_J23, threeM):
    """Derivative of the missing transverse energy (MET) with respect to
    the nuiance parameters (X).
    """
    nu_J_arr = np.array((nu_J1, nu_J23, nu_J23)).T
    pT_x = threeM[:, :, 0] * np.cos(threeM[:, :, 2]) / nu_J_arr
    pT_y = threeM[:, :, 0] * np.sin(threeM[:, :, 2]) / nu_J_arr
    dMET = 1.0 / np.sqrt(np.sum(pT_x, axis=1) ** 2 + np.sum(pT_y, axis=1) ** 2)
    # chain rule terms
    dpT_x_dnu_J = (
        -1
        * np.sum(pT_x, axis=1).reshape(-1, 1)
        * threeM[:, :, 0]
        * np.cos(threeM[:, :, 2])
        / nu_J_arr**2
    )
    dpT_y_dnu_J = (
        -1
        * np.sum(pT_y, axis=1).reshape(-1, 1)
        * threeM[:, :, 0]
        * np.sin(threeM[:, :, 2])
        / nu_J_arr**2
    )
    # full derivative
    dMET_dnu_J_arr = dMET.reshape(-1, 1) * (dpT_x_dnu_J + dpT_y_dnu_J)
    return np.stack(
        (
            dMET_dnu_J_arr[:, 0],
            dMET_dnu_J_arr[:, 0],
            np.sum(dMET_dnu_J_arr[:, 1:], axis=1),
            np.sum(dMET_dnu_J_arr[:, 1:], axis=1),
        ),
        axis=-1,
    )


def der_efficiency(X, events_threeM, pT_cut=200, MET_cut=50, normalized=True, a=1):
    """Derivative of the efficiency with respect to the nuiance parameters (X).
    Sigmoids are used to replace the pT and MET cuts, as well as the eta-based jet scale
    criteria.
    """
    X = np.atleast_2d(X)
    nu_J1_in, nu_J1_out, nu_J23_in, nu_J23_out = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    # replacing eta-based selection criteria with sigmoids
    nu_J1 = nu_J1_in * sigmoid(abs(events_threeM[:, 0, 1]), c=1, a=-a)
    nu_J1 += nu_J1_out * sigmoid(abs(events_threeM[:, 0, 1]), c=1, a=a)
    eta_avg = np.average(events_threeM[:, 1:, 1], weights=events_threeM[:, 1:, 0], axis=1)
    nu_J23 = nu_J23_in * sigmoid(abs(eta_avg), c=1, a=-a)
    nu_J23 += nu_J23_out * sigmoid(abs(eta_avg), c=1, a=a)

    # chain rule of the eta-based selection criteria
    dnu_J = np.stack(
        (
            sigmoid(abs(events_threeM[:, 0, 1]), c=1, a=-a),
            sigmoid(abs(events_threeM[:, 0, 1]), c=1, a=a),
            sigmoid(abs(eta_avg), c=1, a=-a),
            sigmoid(abs(eta_avg), c=1, a=a),
        ),
        axis=-1,
    )

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
    # derivatives of nu_J1  and nu_J23 with respect to the four nuisance
    # parameters
    dpT_sig_dnu_J1_in = dpT_sig_dnu_J1 * sigmoid(abs(events_threeM[:, 0, 1]), c=1, a=-a)
    dpT_sig_dnu_J1_out = dpT_sig_dnu_J1 * sigmoid(abs(events_threeM[:, 0, 1]), c=1, a=a)
    dpT_sig_dnu_J23_in = dpT_sig_dnu_J23 * sigmoid(abs(eta_avg), c=1, a=-a)
    dpT_sig_dnu_J23_out = dpT_sig_dnu_J23 * sigmoid(abs(eta_avg), c=1, a=a)
    # stacking them
    dpT_sig_dnuJ = np.stack(
        (
            dpT_sig_dnu_J1_in,
            dpT_sig_dnu_J1_out,
            dpT_sig_dnu_J23_in,
            dpT_sig_dnu_J23_out,
        ),
        axis=1,
    )

    # replacing MET cut with a sigmoid
    MET_vals = MET(nu_J1, nu_J23, events_threeM)
    MET_sig = sigmoid(MET_vals, c=MET_cut, a=-a).reshape(-1, 1)
    dMET_sig_dnuJ = der_sigmoid(MET_vals, c=MET_cut, a=-a).reshape(-1, 1)
    dMET_dnuJ = der_MET(nu_J1, nu_J23, events_threeM)

    # combining the MET and pT sigmoids - and their chain rule terms
    MET_pT_sigmoids = pT_sig * MET_sig  # both cuts combined
    dMET_pT_sigmoids_dnuJ = pT_sig * dMET_sig_dnuJ * dMET_dnuJ * dnu_J + dpT_sig_dnuJ * MET_sig

    # final derivative of the efficiency
    deff_dnuJ = np.sum(dMET_pT_sigmoids_dnuJ, axis=0) / np.sum(pT_sig)
    deff_dnuJ -= np.sum(MET_pT_sigmoids) / np.sum(pT_sig) ** 2 * np.sum(dpT_sig_dnuJ, axis=0)

    if normalized:
        # using the regular efficiency values for the normalization.
        ix_pT_cut_norm = (events_threeM[:, 0, 0] > pT_cut) & (events_threeM[:, 1, 0] < pT_cut)
        MET_vals_norm = MET(1, 1, events_threeM[ix_pT_cut_norm])
        ix_MET_cut_norm = MET_vals_norm < MET_cut
        norm = np.sum(ix_MET_cut_norm) / np.sum(ix_pT_cut_norm)
        return deff_dnuJ / norm
    return deff_dnuJ
