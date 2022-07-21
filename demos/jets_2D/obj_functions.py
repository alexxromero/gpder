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

def MET(nuJ1, nuJ23, j1_threeM, j2_threeM, j3_threeM):
    # -- pT_x decomp -- #
    j1_pt_x = j1_threeM[0] * np.cos(j1_threeM[2]) / nuJ1
    j2_pt_x = j2_threeM[0] * np.cos(j2_threeM[2]) / nuJ23
    j3_pt_x = j3_threeM[0] * np.cos(j3_threeM[2]) / nuJ23
    # -- pT_y decomp -- #
    j1_pt_y = j1_threeM[0] * np.sin(j1_threeM[2]) / nuJ1
    j2_pt_y = j2_threeM[0] * np.sin(j2_threeM[2]) / nuJ23
    j3_pt_y = j3_threeM[0] * np.sin(j3_threeM[2]) / nuJ23
    # -- MET -- #
    met_x = j1_pt_x + j2_pt_x + j3_pt_x
    met_y = j1_pt_y + j2_pt_y + j3_pt_y
    return np.sqrt(met_x**2 + met_y**2)

def eff_MET50(nuJ1,  # jet 1 energy scale
              nuJ23, # jets 2 & 3 energy scale
              j1_threeM=j1_threeM,
              j2_threeM=j2_threeM,
              j3_threeM=j3_threeM):
    count_met, count_pTcut = 0, 0
    for i in range(len(j1_threeM)):
        if (j1_threeM[i, 0]/nuJ1 > 200.) and (j2_threeM[i, 0]/nuJ23 < 200.):
            count_pTcut += 1
            met = MET(nuJ1, nuJ23, j1_threeM[i], j2_threeM[i], j3_threeM[i])
            if met < 50.:
                count_met += 1
    return count_met / count_pTcut

def dMET_dnuJ(nuJ1, nuJ23, j1_threeM, j2_threeM, j3_threeM):
    # -- pT_x decomp -- #
    j1_pt_x = j1_threeM[0] * np.cos(j1_threeM[2]) / nuJ1
    j2_pt_x = j2_threeM[0] * np.cos(j2_threeM[2]) / nuJ23
    j3_pt_x = j3_threeM[0] * np.cos(j3_threeM[2]) / nuJ23
    # -- pT_y decomp -- #
    j1_pt_y = j1_threeM[0] * np.sin(j1_threeM[2]) / nuJ1
    j2_pt_y = j2_threeM[0] * np.sin(j2_threeM[2]) / nuJ23
    j3_pt_y = j3_threeM[0] * np.sin(j3_threeM[2]) / nuJ23
    # -- MET -- #
    met_x = j1_pt_x + j2_pt_x + j3_pt_x
    met_y = j1_pt_y + j2_pt_y + j3_pt_y
    # Outer term in the chain rule. Factos of 2 cancel out. #
    dmet_dnuJ = 1. / np.sqrt(met_x**2 + met_y**2)

    # Inner terms in the chain rule.
    # -- wrt J1 -- #
    dmet_x_dnuJ1  = -met_x * j1_threeM[0] * np.cos(j1_threeM[2])
    dmet_x_dnuJ1 /= nuJ1**2
    dmet_y_dnuJ1  = -met_y * j1_threeM[0] * np.sin(j1_threeM[2])
    dmet_y_dnuJ1 /= nuJ1**2
    # -- wrt J23 -- #
    dmet_x_dnuJ23 = -met_x * j2_threeM[0] * np.cos(j2_threeM[2])
    dmet_x_dnuJ23 += -met_x * j3_threeM[0] * np.cos(j3_threeM[2])
    dmet_x_dnuJ23 /= nuJ23**2
    dmet_y_dnuJ23 = -met_y * j2_threeM[0] * np.sin(j2_threeM[2])
    dmet_y_dnuJ23 += -met_y * j3_threeM[0] * np.sin(j3_threeM[2])
    dmet_y_dnuJ23 /= nuJ23**2
    derivs = dmet_dnuJ * np.array([(dmet_x_dnuJ1 + dmet_y_dnuJ1),
                                   (dmet_x_dnuJ23 + dmet_y_dnuJ23)])
    return derivs

def deff_MET50_sigmoid(nuJ1, nuJ23,
                       j1_threeM=j1_threeM,
                       j2_threeM=j2_threeM,
                       j3_threeM=j3_threeM,
                       s=1):
    count_met, count_pTcut = 0, 0
    dcount_met_dnuJ, dcount_pTcut_dnuJ = 0, 0
    for i in range(len(j1_threeM)):
        # pT_coeff approximates 1 if the pT of j1 is larger than 200 AND
        # the pT of j2 is less than 200
        pT_coeff  = sigmoid((j1_threeM[i, 0]/nuJ1 - 200.), s=s)
        pT_coeff *= sigmoid(-(j2_threeM[i, 0]/nuJ23 - 200.), s=s)
        count_pTcut += pT_coeff
        met = MET(nuJ1, nuJ23, j1_threeM[i], j2_threeM[i], j3_threeM[i])
        # met_coeff approximates 1 if the MET is less than 50
        met_coeff = sigmoid(-(met-50.), s=s)
        count_met += pT_coeff * met_coeff

        # chain rule terms
        dpT_coeff_dnuJ1  = dsigmoid((j1_threeM[i, 0]/nuJ1 - 200.), s=s)
        dpT_coeff_dnuJ1 *= sigmoid(-(j2_threeM[i, 0]/nuJ23 - 200.), s=s)
        dpT_coeff_dnuJ1 *= -j1_threeM[i, 0] / nuJ1**2
        dpT_coeff_dnuJ23  = sigmoid((j1_threeM[i, 0]/nuJ1 - 200.), s=s)
        dpT_coeff_dnuJ23 *= dsigmoid(-(j2_threeM[i, 0]/nuJ23 - 200.), s=s)
        dpT_coeff_dnuJ23 *= j2_threeM[i, 0] / nuJ23**2
        dpT_coeff_dnuJ = np.array([dpT_coeff_dnuJ1, dpT_coeff_dnuJ23])
        dmet_coeff_dnuJ = -dsigmoid(-(met-50.), s=s)
        dmet_dnuJ = dMET_dnuJ(nuJ1, nuJ23, j1_threeM[i], j2_threeM[i], j3_threeM[i])
        # stitching all the terms together...
        dcount_met_dnuJ += pT_coeff * dmet_coeff_dnuJ * dmet_dnuJ
        dcount_met_dnuJ += dpT_coeff_dnuJ * met_coeff
        dcount_pTcut_dnuJ += dpT_coeff_dnuJ

    deff = dcount_met_dnuJ / count_pTcut
    deff -= count_met / count_pTcut**2 * dcount_pTcut_dnuJ
    return deff
