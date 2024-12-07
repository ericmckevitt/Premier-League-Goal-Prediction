## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm

## adaptively generating minority data samples according to their distributions
def over_sampling_adasyn(

        ## arguments / inputs
        data,  ## training set
        label,  ## label for each observation in the dataset
        index,  ## index of input data
        perc,  ## oversampling percentage
        k,  ## num of neighs for over-sampling
        
        ## inputs from preprocessing
        knn_index,  
        feat_list_nom,
        feat_list_num,
        feat_ranges,
        data_var,
        feat_const,
        feat_var,
        feat_non_neg,
        data_orig
):
    """
    generates synthetic observations and is the primary function underlying the
    over-sampling technique utilized in the higher main function 'adasyn()', the
    4 step procedure for generating synthetic observations is:

    1) pre-processing: temporarily removes features without variation, label
    encodes nominal / categorical features, and subsets the training set into
    two data sets by data type: numeric / continuous, and nominal / categorical

    2) distances: calculates the cartesian distances between all observations,
    distance metric automatically determined by data type (euclidean distance
    for numeric only data, heom distance for both numeric and nominal data, and
    hamming distance for nominal only data) and determine k nearest neighbors

    3) over-sampling: 'adasyn' is used to determine the number of new synthetic
    observations to be generated for each observation according to the ratio of
    majority class in its k nearest neighbors.

    'adasyn' only applies to numeric / continuous features,
    for nominal / categorical features, synthetic values are generated at random
    from sampling observed values found within the same feature

    4) post processing: restores original values for label encoded features,
    reintroduces constant features previously removed, converts any interpolated
    negative values to zero in the case of non-negative features

    returns a pandas dataframe containing both new and original observations of
    the training set which are then returned to the higher main function 'adasyn()'

    ref:

    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
    
    Branco, P., Ribeiro, R., Torgo, L. (2017). 
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.

    He, H., Bai, Y., Garcia, E. A., & Li, S. (2008, June).
    ADASYN: Adaptive synthetic sampling approach for imbalanced
    learning. In 2008 IEEE international joint conference on neural
    networks (IEEE world congress on computational intelligence)
    (pp. 1322-1328). IEEE.
    https://www.ele.uri.edu/faculty/he/PDFfiles/adasyn.pdf.

    """

    ## store dimensions of data subset
    n = len(data)
    d = len(data.columns)

    ## find knn_index that belong to the bump
    ## and convert new index list to matrix # ????? check
    temp = []
    for i in range(len(knn_index)):
        if knn_index[i][0] in index: # ????? knn_index[i][0] == i
            temp.append(knn_index[i])

    knn_matrix = np.array(temp)

    ## total number of new synthetic observations to generate
    n_synth = int(len(index) * (perc - 1))

    ## find the ratio ri = #majority class in neighbourhood/k for each observation
    ri = 0
    r = []
    for i in range(len(knn_matrix)):
        count_majority = 0
        for j in range(1, k + 1):
            if label[knn_matrix[i][j]] == -1: # ????? label sorted, knn_matrix[i][j] unsorted UPDATE: solved in adasyn.py
                count_majority += 1
        ri = count_majority / k
        r.append(ri)

    ## normalize ri's so their sum equals to 1
    Rhat_i = []
    for r_value in r:
        rhat_i = r_value / sum(r)
        Rhat_i.append(rhat_i)
    assert (sum(Rhat_i) > 0.99)

    ## calculate the number of new synthetic observations
    ## that will be generated for each observation # ????? n_synth - sum(Gi) discarded
    Gi = []
    for rhat_i in Rhat_i:
        gi = round(rhat_i * n_synth)
        Gi.append(int(gi))

    ## sort index since knn_matrix is stored in ascending order
    index.sort()

    ## create null matrix to store new synthetic observations
    synth_matrix = np.ndarray(shape=((sum(Gi)), d))

    for i in tqdm(range(len(index)), ascii=True, desc="index"):
        if Gi[i] > 0:

            num = sum(Gi[:i])

            # ?????  ----- moved out of looping range(Gi[i])
            no_minority = True # ????? not necessary
            # ????? Modified.
            minority_indices = list()
            ## check if there is at least 1 neighbor belongs to minority class
            for l in range(1, len(knn_matrix[i])):
                if label[knn_matrix[i][l]] == 1: # ????? not necessary
                    if knn_matrix[i][l] in index:
                        no_minority = False
                        minority_indices.append(l)
                    if len(minority_indices) == k:
                        break
                    elif len(minority_indices) > k:
                        raise ValueError("Invalid length of minority_indices.")
            # ????? ----- moved out of looping range(Gi[i])

            for j in range(Gi[i]):

                if no_minority == True: # ????? equivalent to len(minority_indices) == 0
                    synth_matrix[num + j, 0:d] = data.iloc[index[i], 0:d]

                # ????? Modified.
                if no_minority == False:
                    neigh = int(np.random.choice(
                        a=tuple(minority_indices),
                        size=1))
                    # ## check if the selected neighbor belongs to minority class
                    # ## and if not, reselect until this neighbor belongs to minority class
                    # while (label[knn_matrix[i][neigh]] != 1):
                    #     neigh = int(np.random.choice(
                    #         a=tuple(range(1, k + 1)),
                    #         size=1))

                    ## conduct synthetic minority over-sampling
                    ## technique for regression (adasyn)
                    diffs = data.iloc[knn_matrix[i, neigh], 0:(d - 1)] - data.iloc[index[i], 0:(d - 1)]
                    synth_matrix[num + j, 0:(d - 1)] = data.iloc[index[i], 0:(d - 1)] + rd.random() * diffs # ????? SMOGN: boundary 1 excluded

                    ## randomly assign nominal / categorical features from
                    ## observed cases and selected neighbors
                    for x in feat_list_nom:
                        synth_matrix[num + j, x] = [data.iloc[knn_matrix[i, neigh], x],
                                                    data.iloc[index[i], x]][round(rd.random())]

                    ## generate synthetic y response variable by # ????? SMOGN: loop over feat_list_num, only last z matters?
                    ## inverse distance weighted
                    for z in feat_list_num:
                        a = abs(data.iloc[index[i], z] -
                                synth_matrix[num + j, z]) / feat_ranges[z]
                        b = abs(data.iloc[knn_matrix[i, neigh], z] -
                                synth_matrix[num + j, z]) / feat_ranges[z]

                    if len(feat_list_nom) > 0:
                        a = a + sum(data.iloc[index[i], feat_list_nom] !=
                                    synth_matrix[num + j, feat_list_nom])
                        b = b + sum(data.iloc[knn_matrix[i, neigh], feat_list_nom] !=
                                    synth_matrix[num + j, feat_list_nom])

                    if a == b:
                        synth_matrix[num + j, (d - 1)] = data.iloc[index[i], (d - 1)] + data.iloc[
                            knn_matrix[i, neigh], (d - 1)] / 2
                    else:
                        synth_matrix[num + j, (d - 1)] = (b * data.iloc[index[i], (d - 1)] +
                                                          a * data.iloc[knn_matrix[i, neigh], (d - 1)]) / (a + b)

    ## create null matrix to store original observations
    original_matrix = np.ndarray(shape=(len(index), d))

    for i in tqdm(range(len(index)), ascii=True, desc="ori_index"):
        original_matrix[i, 0:d] = data.iloc[index[i], 0:d]

    ## concatenate new generated synthetic observations with original observations
    final_matrix = np.concatenate((synth_matrix, original_matrix), axis=0)

    ## convert final matrix to dataframe
    data_new = pd.DataFrame(final_matrix)

    ## synthetic data quality check
    if sum(data_new.isnull().sum()) > 0:
        raise ValueError("oops! synthetic data contains missing values")

    ## replace label encoded values with original values
    for j in feat_list_nom:
        code_list = data.iloc[:, j].unique()
        cat_list = data_var.iloc[:, j].unique()

        for x in code_list:
            data_new.iloc[:, j] = data_new.iloc[:, j].replace(x, cat_list[x])

    ## reintroduce constant features previously removed
    if len(feat_const) > 0:
        data_new.columns = feat_var

        for j in range(len(feat_const)):
            data_new.insert(
                loc=int(feat_const[j]),
                column=feat_const[j],
                value=np.repeat(
                    data_orig.iloc[0, feat_const[j]], # ????? data or data_orig? UPDATE: changed from data to data_orig, may be problematic to some datasets, need further verification.
                    len(final_matrix))
            )

    ## convert negative values to zero in non-negative features
    for j in feat_non_neg:
        # data_new.iloc[:, j][data_new.iloc[:, j] < 0] = 0
        data_new.iloc[:, j] = data_new.iloc[:, j].clip(lower=0)

    ## return over-sampling results dataframe
    return data_new
