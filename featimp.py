import shap
from sklearn.base import is_classifier, is_regressor
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, log_loss, r2_score, mean_absolute_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import xgboost
from tqdm import tqdm
from scipy.stats import t, norm
# plot function


def visualizing(sorted_featimp):
    fig, ax = plt.subplots()
    sns.barplot(y=sorted_featimp.index, x=sorted_featimp.values,
                palette=['#008bfb' if c >= 0 else '#ff0051' for c in sorted_featimp.values], ax=ax)
    ax.bar_label(ax.containers[-1], fmt='%.2f', label_type='center')

# linear coef featimp


def linear_featimp(X, y):
    linear_x = pd.DataFrame(StandardScaler().fit_transform(X), columns=list(X))
    sorted_featimp = pd.Series(LinearRegression().fit(
        linear_x, y).coef_, list(X)).sort_values(ascending=False)
    return sorted_featimp

# Spearman's rank correlation coefficient


def Spearman_featimp(X, y):
    mg_df = X.merge(pd.DataFrame(
        y, columns=['y']), left_index=True, right_index=True)
    sorted_featimp = mg_df.corr(method='spearman')[
        'y'][:-1].sort_values(ascending=False)
    return sorted_featimp

# PCA feature important


def pca_featimp(X, scale_X=True):
    columns = X.columns
    if scale_X:
        X = StandardScaler().fit_transform(X)
    pca = PCA()
    pca.fit(X)
    pca_X = pca.transform(X)
    pcamapdf = pd.DataFrame(pca.components_, index=columns, columns=[
                            'PC-'+str(i) for i in np.arange(pca.components_.shape[0])+1])
    sorted_featimp = pcamapdf['PC-1'].sort_values(ascending=False)
    return sorted_featimp

# mRMR


def mRMR_featimp(X, y):
    mg_df = X.merge(pd.DataFrame(
        y, columns=['y']), left_index=True, right_index=True)
    corr_df = mg_df.corr(method='spearman')
    # corr between each feature and y
    I = corr_df['y'][:-1]
    # sum of corr between each feature and other feaures
    # devided by total number of features
    # -1 to correct each feature's correlation with itself is 1
    # |S| is the length of selected features
    b = (np.sum(corr_df.iloc[:-1, :-1], axis=1)-1)/(X.shape[1])
    sorted_featimp = I-b
    return sorted_featimp.sort_values(ascending=False)

# permutation


def permutation_importances(rf, X_train, y_train):
    '''
    rf = RandomForestRegressor(oob_score=True)
    '''
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)

    if is_regressor(rf):
        metric = r2_score

    elif is_classifier(rf):
        metric = accuracy_score

    baseline = metric(y_train, y_pred)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        y_pred = rf.predict(X_train)

        m = metric(y_train, y_pred)

        X_train[col] = save
        imp.append(baseline - m)
    # can't distinguish positive/negative change
    sorted_featimp = abs(pd.Series(np.array(imp), list(X_train)))
    return sorted_featimp.sort_values(ascending=False)


def dropcol_importances(rf, X_train, y_train, random_state=False):
    # for rf handle, need to set oob_score = True
    rf_ = clone(rf)
    # fix randness for debuging purpose
    if random_state:
        rf_.random_state = 999
    rf_.fit(X_train, y_train)
    # use oob_score
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        if random_state:
            rf_.random_state = 999
        rf_.fit(X.values, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
        data={'Feature': X_train.columns,
              'Importance': imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I['Importance']


def compare_strategies(model, X, y, scale_X=False, random_state=True):
    '''
    INPUT mode : {'OLS','RF','XGBoost'}
    '''
    # collect 6 different strategies results
    spearman_result = abs(Spearman_featimp(X, y)).sort_values(ascending=False)
    pca_result = abs(pca_featimp(X, scale_X=scale_X)
                     ).sort_values(ascending=False)
    mrmr_result = abs(mRMR_featimp(X, y)).sort_values(ascending=False)
    linear_result = abs(linear_featimp(X, y)).sort_values(ascending=False)

    rf = RandomForestRegressor(oob_score=True)
    permut_result = abs(permutation_importances(
        rf, X, y)).sort_values(ascending=False)

    drop_result = abs(dropcol_importances(
        rf, X, y, random_state=random_state)).sort_values(ascending=False)

    results = [spearman_result, pca_result, mrmr_result,
               linear_result, permut_result, drop_result]
    maes = [[] for _ in range(X.shape[1])]
    for i in range(X.shape[1]):
        for r in results:
            # standard scale for liner ols
            scale_x = X[r[:i+1].index]
            if model == 'OLS':
                handle = LinearRegression()
            elif model == 'RF':
                handle = RandomForestRegressor()
            elif model == 'XGBoost':
                handle = xgboost.XGBRegressor()
            y_pred = cross_val_predict(handle, scale_x, y, cv=5)
            mae_score = mean_absolute_error(y, y_pred)
            # append each method's mae with the same amount of features
            maes[i].append(mae_score)

    df = pd.DataFrame(maes, columns=[
                      'Spearman', 'PCA', 'mRMR', 'OLS', 'Permut_col', 'Drop_col'], index=range(1, 15))

    # plotting
    plt.figure(figsize=(8, 5))
    for i in df:
        sns.lineplot(x=df.index, y=df[i], label=i)
        plt.legend(loc=0)
        plt.ylabel('20% 5-fold CV MAE')
        plt.xlabel('Top k most important features')
        plt.title(f'Boston housing prices with {model}')
    return df

# Automatic feature selection algorithm


def auto_feat_select(X, y, rf):
    plotx = []
    ploty = []
    # add noise column
    noise = pd.Series(np.random.normal(0, 1, X.shape[0]), name='noise')
    X_noise = X.join(noise)

    X_noise_scal = pd.DataFrame(
        StandardScaler().fit_transform(X_noise), columns=list(X_noise))

    plotx.append(X_noise_scal.shape[1])
    # baseline loss
    rf_baseline = clone(rf)
    rf_permut = clone(rf)
    rf.fit(X_noise_scal, y)

    if is_regressor(rf):
        metric = mean_squared_error
        y_pred = rf.predict(X_noise_scal)
    elif is_classifier(rf):
        metric = log_loss
        y_pred = rf.predict_proba(X_noise_scal)

    all_feature_pred = metric(y, y_pred)
    ploty.append(all_feature_pred)

    # get permut featimp
    featimp = permutation_importances(rf, X_noise_scal, y)
    # get feature names that above the noise column
    imp_features = list(featimp[:list(featimp.index).index('noise')].index)
    # drop all features below noise and noise
    X_reduce = X_noise_scal.loc[:, imp_features]
    plotx.append(X_reduce.shape[1])
    # recompute loss
    rf_baseline.fit(X_reduce, y)

    if is_regressor(rf_baseline):
        metric = mean_squared_error
        y_pred_baseline = rf_baseline.predict(X_reduce)
    elif is_classifier(rf_baseline):
        metric = log_loss
        y_pred_baseline = rf_baseline.predict_proba(X_reduce)

    baseline = metric(y, y_pred_baseline)
    ploty.append(baseline)

    # drop one feature at a time from the least imp one
    for i in range(1, len(imp_features)):
        X_reduce_copy = X_reduce.copy()
        rf_new = clone(rf_permut)
        X_reduce = X_reduce.loc[:, imp_features[:-i]]
        plotx.append(X_reduce.shape[1])
        # recompute loss
        rf_new.fit(X_reduce, y)

        if is_regressor(rf_new):
            metric = mean_squared_error
            y_pred_new = rf_new.predict(X_reduce)
        elif is_classifier(rf_new):
            metric = log_loss
            y_pred_new = rf_new.predict_proba(X_reduce)

        new_metirc = metric(y, y_pred_new)
        ploty.append(new_metirc)
        selected_features = list(X_reduce_copy)

        if new_metirc > baseline:
            return selected_features, plotx, ploty
        baseline = new_metirc.copy()

# Variance and empirical p-values for feature importances


def empirical_imp(X, y, handle=RandomForestRegressor, n_run=100, shuffle_y=False):
    '''
    shuffle_y = False - Returns actual permuted Feature important with 95% CI
    shuffle_y = True - Returns Feature importance under Ho: feature is not significant in predicting y
    '''
    imp_df = pd.DataFrame()
    for i in tqdm(range(n_run)):
        rf = handle(oob_score=True)
        if shuffle_y:
            permute_imp = permutation_importances(
                rf, X, np.random.permutation(y))
        else:
            permute_imp = permutation_importances(rf, X, y)
        # normalize
        normalized_imp = permute_imp/permute_imp.sum()
        # form to 1*k df
        norm_df = pd.DataFrame(normalized_imp).T
        # Concat the latest importances with the old ones
        imp_df = pd.concat([imp_df, norm_df], axis=0)
    imp_df.reset_index(drop=True, inplace=True)
    return imp_df

# Ho distrition plots


def distimp_plot(null_df, feature, actual_normalized_imp_df, log_trans=False):
    # some of the distributions are skewed, so in defult we apply log_trans
    if log_trans:
        df = np.log(null_df)
        act_vline = np.log(actual_normalized_imp_df[feature].mean(axis=0))
    else:
        act_vline = actual_normalized_imp_df[feature].mean(axis=0)

    sns.displot(null_df[feature])
    plt.axvline(x=act_vline, color='red', label='act result')
    # 95% thresholds
    mean = null_df[feature].mean()
    std = null_df[feature].std()
    lower_threshold = norm.ppf(0.05, loc=mean, scale=std)
    upper_threshold = norm.ppf(0.95, loc=mean, scale=std)
    plt.axvline(x=lower_threshold, color='orange', label='5% threshold')
    plt.axvline(x=upper_threshold, color='orange')
    plt.legend(loc=0)
    plt.title(f'{feature} importance under Ho')

# plot all the features


def plot_all_features(null_df, actual_normalized_imp_df, log_trans=False):
    for feature in list(null_df):
        distimp_plot(null_df, feature, actual_normalized_imp_df,
                     log_trans=log_trans)
