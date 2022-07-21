import numpy as np
import scipy
from scipy.special import expit
import h5py

# ----- Reading the data from file ---- #
f = h5py.File("../datasets/three_jets_30k.h5", 'r')
j1_threeM = np.array(f['j1_threeM'])
j2_threeM = np.array(f['j2_threeM'])
j3_threeM = np.array(f['j3_threeM'])
f.close()
# ------------------------------------- #

def sigmoid(X, s=1):
    return expit(X / s)

def dsigmoid(X, s=1):
    return (1. / s) * sigmoid(X, s) * (1. - sigmoid(X, s))

def MET(nu_J, nu_j, j1_threeM, j2_threeM, j3_threeM):
    # -- pT_x decomp -- #
    j1_pt_x = j1_threeM[0] * np.cos(j1_threeM[2]) / nu_J
    j2_pt_x = j2_threeM[0] * np.cos(j2_threeM[2]) / nu_j
    j3_pt_x = j3_threeM[0] * np.cos(j3_threeM[2]) / nu_j
    # -- pT_y decomp -- #
    j1_pt_y = j1_threeM[0] * np.sin(j1_threeM[2]) / nu_J
    j2_pt_y = j2_threeM[0] * np.sin(j2_threeM[2]) / nu_j
    j3_pt_y = j3_threeM[0] * np.sin(j3_threeM[2]) / nu_j
    # -- MET -- #
    met_x = j1_pt_x + j2_pt_x + j3_pt_x
    met_y = j1_pt_y + j2_pt_y + j3_pt_y
    return np.sqrt(met_x**2 + met_y**2)

def eff_MET50(nu_J_in,   # jet 1 energy scale
              nu_J_out,  # jet 1 energy scale
              nu_j_in,   # jets 2 & 3 energy scale
              nu_j_out,  # jets 2 & 3 energy scale
              j1_threeM=j1_threeM,
              j2_threeM=j2_threeM,
              j3_threeM=j3_threeM):
    count_met, count_pTcut = 0, 0
    for i in range(len(j1_threeM)):
        nu_J = nu_J_in if (abs(j1_threeM[i, 1]) < 1) else nu_J_out
        eta_j_list = [j2_threeM[i, 1], j3_threeM[i, 1]]
        pT_j_list = [j2_threeM[i, 0], j3_threeM[i, 0]]
        eta_avg_j = np.average(eta_j_list, weights=pT_j_list)
        nu_j = nu_j_in if (abs(eta_avg_j) < 1) else nu_j_out
        if (j1_threeM[i, 0]/nu_J > 200.) and (j2_threeM[i, 0]/nu_j < 200.):
            count_pTcut += 1
            met = MET(nu_J, nu_j, j1_threeM[i], j2_threeM[i], j3_threeM[i])
            if met < 50.:
                count_met += 1
    return count_met / count_pTcut

def dMET_dnuJ(nu_J, nu_j, j1_threeM, j2_threeM, j3_threeM):
    # -- pT_x decomp -- #
    j1_pt_x = j1_threeM[0] * np.cos(j1_threeM[2]) / nu_J
    j2_pt_x = j2_threeM[0] * np.cos(j2_threeM[2]) / nu_j
    j3_pt_x = j3_threeM[0] * np.cos(j3_threeM[2]) / nu_j
    # -- pT_y decomp -- #
    j1_pt_y = j1_threeM[0] * np.sin(j1_threeM[2]) / nu_J
    j2_pt_y = j2_threeM[0] * np.sin(j2_threeM[2]) / nu_j
    j3_pt_y = j3_threeM[0] * np.sin(j3_threeM[2]) / nu_j
    # -- MET -- #
    met_x = j1_pt_x + j2_pt_x + j3_pt_x
    met_y = j1_pt_y + j2_pt_y + j3_pt_y
    # Outer term in the chain rule. Factos of 2 cancel out. #
    dmet_dnuJ = 1. / np.sqrt(met_x**2 + met_y**2)

    # Inner terms in the chain rule.
    # -- wrt J1 -- #
    dmet_x_dnu_J  = -met_x * j1_threeM[0] * np.cos(j1_threeM[2])
    dmet_x_dnu_J /= nu_J**2
    dmet_y_dnu_J  = -met_y * j1_threeM[0] * np.sin(j1_threeM[2])
    dmet_y_dnu_J /= nu_J**2
    # -- wrt J23 -- #
    dmet_x_dnu_j = -met_x * j2_threeM[0] * np.cos(j2_threeM[2])
    dmet_x_dnu_j += -met_x * j3_threeM[0] * np.cos(j3_threeM[2])
    dmet_x_dnu_j /= nu_j**2
    dmet_y_dnu_j = -met_y * j2_threeM[0] * np.sin(j2_threeM[2])
    dmet_y_dnu_j += -met_y * j3_threeM[0] * np.sin(j3_threeM[2])
    dmet_y_dnu_j /= nu_j**2
    derivs = dmet_dnuJ * np.array([(dmet_x_dnu_J + dmet_y_dnu_J),
                                   (dmet_x_dnu_j + dmet_y_dnu_j)])
    return derivs

def deff_MET50_sigmoid(nu_J_in,   # jet 1 energy scale
                       nu_J_out,  # jet 1 energy scale
                       nu_j_in,   # jets 2 & 3 energy scale
                       nu_j_out,  # jets 2 & 3 energy scale
                       j1_threeM=j1_threeM,
                       j2_threeM=j2_threeM,
                       j3_threeM=j3_threeM,
                       s=1):
    count_met, count_pTcut = 0, 0
    dcount_pTcut_dnu_J_in, dcount_pTcut_dnu_J_out = 0, 0
    dcount_pTcut_dnu_j_in, dcount_pTcut_dnu_j_out = 0, 0
    dcount_met_dnu_J_in, dcount_met_dnu_J_out = 0, 0
    dcount_met_dnu_j_in, dcount_met_dnu_j_out = 0, 0
    for i in range(len(j1_threeM)):
        nu_J  = nu_J_in * sigmoid(-(abs(j1_threeM[i, 1])-1), s=s)
        nu_J += nu_J_out * sigmoid((abs(j1_threeM[i, 1])-1), s=s)

        eta_j_list = [j2_threeM[i, 1], j3_threeM[i, 1]]
        pT_j_list = [j2_threeM[i, 0], j3_threeM[i, 0]]
        eta_avg_j = np.average(eta_j_list, weights=pT_j_list)
        nu_j = nu_j_in * sigmoid(-(abs(eta_avg_j)-1), s=s)
        nu_j += nu_j_out * sigmoid((abs(eta_avg_j)-1), s=s)

        pT_coeff  = sigmoid((j1_threeM[i, 0]/nu_J - 200.), s=s)
        pT_coeff *= sigmoid(-(j2_threeM[i, 0]/nu_j - 200.), s=s)
        count_pTcut += pT_coeff

        met = MET(nu_J, nu_j, j1_threeM[i], j2_threeM[i], j3_threeM[i])
        met_coeff = sigmoid(-(met-50.), s=s)
        count_met += pT_coeff * met_coeff

        # chain rule terms
        dpT_coeff_dnu_J  = dsigmoid((j1_threeM[i, 0]/nu_J - 200.), s=s)
        dpT_coeff_dnu_J *= sigmoid(-(j2_threeM[i, 0]/nu_j - 200.), s=s)
        dpT_coeff_dnu_J *= -j1_threeM[i, 0] / nu_J**2
        dpT_coeff_dnu_j  = sigmoid((j1_threeM[i, 0]/nu_J - 200.), s=s)
        dpT_coeff_dnu_j *= dsigmoid(-(j2_threeM[i, 0]/nu_j - 200.), s=s)
        dpT_coeff_dnu_j *= j2_threeM[i, 0] / nu_j**2
        dmet_coeff_dmet = -dsigmoid(-(met-50.), s=s)
        dmet_dnu_J, dmet_dnu_j = dMET_dnuJ(nu_J, nu_j, j1_threeM[i], j2_threeM[i], j3_threeM[i])
        dnu_J_dnu_J_in = sigmoid(-(abs(j1_threeM[i, 1])-1), s=s)
        dnu_J_dnu_J_out = sigmoid((abs(j1_threeM[i, 1])-1), s=s)
        dnu_j_dnu_j_in = sigmoid(-(abs(eta_avg_j)-1), s=(s*10))
        dnu_j_dnu_j_out = sigmoid((abs(eta_avg_j)-1), s=(s*10))
        # stitching all the terms together...
        dcount_met_dnu_J_in += pT_coeff * dmet_coeff_dmet * dmet_dnu_J * dnu_J_dnu_J_in
        dcount_met_dnu_J_in += dpT_coeff_dnu_J * met_coeff * dnu_J_dnu_J_in
        dcount_met_dnu_J_out += pT_coeff * dmet_coeff_dmet * dmet_dnu_J * dnu_J_dnu_J_out
        dcount_met_dnu_J_out += dpT_coeff_dnu_J * met_coeff * dnu_J_dnu_J_out
        dcount_met_dnu_j_in += pT_coeff * dmet_coeff_dmet * dmet_dnu_j * dnu_j_dnu_j_in
        dcount_met_dnu_j_in += dpT_coeff_dnu_j * met_coeff * dnu_j_dnu_j_in
        dcount_met_dnu_j_out += pT_coeff * dmet_coeff_dmet * dmet_dnu_j * dnu_j_dnu_j_out
        dcount_met_dnu_j_out += dpT_coeff_dnu_j * met_coeff * dnu_j_dnu_j_out

        dcount_pTcut_dnu_J_in += dpT_coeff_dnu_J * dnu_J_dnu_J_in
        dcount_pTcut_dnu_J_out += dpT_coeff_dnu_J * dnu_J_dnu_J_out
        dcount_pTcut_dnu_j_in += dpT_coeff_dnu_j * dnu_j_dnu_j_in
        dcount_pTcut_dnu_j_out += dpT_coeff_dnu_j * dnu_j_dnu_j_out

        dcount_met_dnuJall = np.array([dcount_met_dnu_J_in,
                                       dcount_met_dnu_J_out,
                                       dcount_met_dnu_j_in,
                                       dcount_met_dnu_j_out])

        dcount_pTcut_dnuJall = np.array([dcount_pTcut_dnu_J_in,
                                        dcount_pTcut_dnu_J_out,
                                        dcount_pTcut_dnu_j_in,
                                        dcount_pTcut_dnu_j_out])

    deff = dcount_met_dnuJall / count_pTcut
    deff -= count_met / count_pTcut**2 * dcount_pTcut_dnuJall
    return deff
