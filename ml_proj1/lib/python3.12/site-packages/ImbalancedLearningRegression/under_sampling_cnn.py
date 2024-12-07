## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


## condense normal observations
def under_sampling_cnn(
    
    ## arguments / inputs
    data,           ## training set
    index,          ## index of input data
    estimator,      ## KNN classifier
    store_indices,  ## original indices of STORE are those in the minority set
    n_seed          ## number of seed samples moved from a mojority set to STORE at the beginning
    
    ):
    
    """
    under-sample observations and is the primary function underlying the
    under-sampling technique utilized in the higher main function 'cnn()', the
    4 step procedure for generating synthetic observations is:
    
    1) pre-processing: label encodes nominal / categorical features, and subsets 
    the training set into two data sets by data type: numeric / continuous, and 
    nominal / categorical
    
    2) under-sampling: CNN, which apply CNN rule to choose a subset of a majority 
    dataset
    
    3) post processing: restores original values for label encoded features, 
    converts any interpolated negative values to zero in the case of non-negative 
    features
    
    returns a pandas dataframe containing synthetic observations of the training
    set which are then returned to the higher main function 'cnn()'
    
    ref:
    
    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
    
    Branco, P., Ribeiro, R., Torgo, L. (2017). 
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.

    Branco, P., Torgo, L., & Ribeiro, R. P. (2019). 
    Pre-processing approaches for imbalanced distributions in regression. 
    Neurocomputing, 343, 76-99. 
    https://www.sciencedirect.com/science/article/abs/pii/S0925231219301638

    Hart, P. (1968). 
    The condensed nearest neighbor rule (corresp.). 
    IEEE transactions on information theory, 14(3), 515-516.
    https://ieeexplore.ieee.org/document/1054155

    Kunz, N., (2019). SMOGN. 
    https://github.com/nickkunz/smogn
    """

    ## initialize grabbag
    grabbag_indices = list()
    
    ## randomly pick one or more sample(s) from majority and add it to STORE
    try:
        normal_seed_index = list(np.random.choice(a = index, size = n_seed, replace = False))
    except ValueError:
        print("n_seed =", n_seed, ">", len(index))
        print("WARNING: n_seed is greater than the number of samples avaiable in a majority bin, used n_seed = 1 instead!")
        normal_seed_index = list(np.random.choice(a = index, size = 1, replace = False))
    store_indices.extend(normal_seed_index)

    ## store dimensions of data subset
    n = len(data)
    d = len(data.columns)
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
    
    ## find non-negative numeric features
    feat_non_neg = [] 
    num_dtypes = ["int64", "float64"]
    
    for j in range(d):
        if data.iloc[:, j].dtype in num_dtypes and any(data.iloc[:, j] > 0):
            feat_non_neg.append(j)
    
    ## create copy of data containing variation
    data_var = data.copy()
    
    ## create global feature list by column index
    feat_list = list(data.columns.values)
    
    ## create nominal feature list and
    ## label encode nominal / categorical features
    ## (strictly label encode, not one hot encode) 
    feat_list_nom = []
    nom_dtypes = ["object", "bool", "datetime64"]
    
    # Unknown warning, may be handled later
    pd.options.mode.chained_assignment = None
    
    for j in range(d):
        if data.dtypes[j] in nom_dtypes:
            feat_list_nom.append(j)
            data.iloc[:, j] = pd.Categorical(pd.factorize(
                data.iloc[:, j])[0])
    
    data = data.apply(pd.to_numeric)
    
    ## create numeric feature list
    feat_list_num = list(set(feat_list) - set(feat_list_nom))
    
    ## calculate ranges for numeric / continuous features
    ## (includes label encoded features)
    feat_ranges = list(np.repeat(1, d))
    
    if len(feat_list_nom) > 0:
        for j in feat_list_num:
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    else:
        for j in range(d):
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    
    ## subset feature ranges to include only numeric features
    ## (excludes label encoded features)
    feat_ranges_num = [feat_ranges[i] for i in feat_list_num]
    
    ## subset data by either numeric / continuous or nominal / categorical
    data_num = data.iloc[:, feat_list_num]
    data_nom = data.iloc[:, feat_list_nom]
    
    ## get number of features for each data type
    feat_count_num = len(feat_list_num)
    feat_count_nom = len(feat_list_nom)

    ## initial training
    train_X = [list(data.iloc[i,:(d-1)].values) for i in store_indices]
    train_y = [(0 if i in index else 1) for i in store_indices]
    min_max_scaler = MinMaxScaler()
    train_X_minmax = min_max_scaler.fit_transform(train_X)
    estimator.fit(train_X_minmax, train_y)

    ## loop through the majority set
    randomized_index = index.copy()
    rd.shuffle(randomized_index)
    for i in randomized_index:
        if i in store_indices:
            continue
        predict_X = min_max_scaler.transform(data.iloc[i,:(d-1)].values.reshape(1,-1))
        predict_y = estimator.predict(predict_X)
        if predict_y == 0:
            grabbag_indices.append(i)
        else:
            store_indices.append(i)
            train_X = [list(data.iloc[j,:(d-1)].values) for j in store_indices]
            train_y = [(0 if j in index else 1) for j in store_indices]
            min_max_scaler = MinMaxScaler()
            train_X_minmax = min_max_scaler.fit_transform(train_X)
            estimator.fit(train_X_minmax, train_y)

    ## loop through the grabbag until empty or no transfer
    while True:
        if len(grabbag_indices) == 0:
            break
        has_transfer = False
        new_grabbag_indices = list()
        for i in grabbag_indices:
            if i in store_indices:
                raise ValueError("index exists in both store and grabbag")
            predict_X = min_max_scaler.transform(data.iloc[i,:(d-1)].values.reshape(1,-1))
            predict_y = estimator.predict(predict_X)
            if predict_y == 0:
                new_grabbag_indices.append(i)
            else:
                has_transfer = True
                store_indices.append(i)
                train_X = [list(data.iloc[j,:(d-1)].values) for j in store_indices]
                train_y = [(0 if j in index else 1) for j in store_indices]
                min_max_scaler = MinMaxScaler()
                train_X_minmax = min_max_scaler.fit_transform(train_X)
                estimator.fit(train_X_minmax, train_y)
        grabbag_indices = new_grabbag_indices
        if not has_transfer:
            break

    ## conduct under sampling and store modified training set
    data_new = pd.DataFrame()
    cond = [i in store_indices and i in index for i in range(n)]
    data_new = pd.concat([data.loc[cond,:], data_new], ignore_index = True)
    
    ## replace label encoded values with original values
    for j in feat_list_nom:
        code_list = data.iloc[:, j].unique()
        cat_list = data_var.iloc[:, j].unique()
        
        for x in code_list:
            data_new.iloc[:, j] = data_new.iloc[:, j].replace(x, cat_list[x])
    
    ## convert negative values to zero in non-negative features
    for j in feat_non_neg:
        # data_new.iloc[:, j][data_new.iloc[:, j] < 0] = 0
        data_new.iloc[:, j] = data_new.iloc[:, j].clip(lower = 0)
    
    ## return over-sampling results dataframe
    return data_new
