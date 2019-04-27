#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor,BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import validation_curve
from sklearn import metrics, svm
from sklearn.svm import SVR
#from xgboost import XGBRegressor
from scipy.stats import boxcox
from math import exp
from sklearn.svm import SVR
#from mlxtend.regressor import StackingRegressor

from xgboost import XGBRegressor



#import data from kaggle
train_df1_ = pd.read_csv('train.csv')
test_df1_ = pd.read_csv('test.csv')
#combine both data for feature editing
combined_df = pd.concat([train_df1_, test_df1_], axis=0, sort=False)

# Confirm the number of missing values in each column.
combined_df.info()

#missing heat map
sns.heatmap(combined_df.isnull(), cbar=False)
plt.title('Heatmap of missingness')
plt.show()

#isnull
print(combined_df.isna().sum()>0)
#correlation matrix for the train df
corr = train_df1.corr()
fig, ax = plt.subplots(figsize=(18,18))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap= 'BuGn', annot=True, 
        ax=ax, fmt='.0%',
        linewidths = 1.0)

#dealing with the missing values
combined_df.Alley.fillna('No_alley', inplace = True)
combined_df.PoolQC.fillna('No_pool', inplace = True)
combined_df.Fence.fillna('No_fence', inplace = True)
combined_df.Electrical.fillna(combined_df.Electrical.value_counts().idxmax(), inplace=True)
combined_df.GarageQual.fillna('No_garage', inplace = True)
combined_df.MasVnrType.fillna('Not_available', inplace = True)
combined_df.MasVnrArea.fillna(0, inplace = True)
combined_df.BsmtQual.fillna('No_basement', inplace = True)
combined_df.BsmtExposure.fillna('No_basement', inplace = True)
combined_df.BsmtFinType1.fillna('No_basement', inplace = True)
combined_df.MSZoning.fillna('No_zone', inplace = True)
combined_df.Utilities.fillna('No_utilities', inplace = True)
combined_df.Exterior1st.fillna('Not_available', inplace = True)
combined_df.Exterior2nd.fillna('Not_available', inplace = True)
combined_df.BsmtCond.fillna('No_basement', inplace = True)
combined_df.BsmtFinSF1.fillna(0, inplace = True)
combined_df.BsmtFinType2.fillna('No_basement', inplace = True)
combined_df.BsmtFinSF2.fillna(0, inplace = True)
combined_df.BsmtUnfSF.fillna(0, inplace=True)
combined_df.TotalBsmtSF.fillna(0, inplace = True)
combined_df.BsmtFullBath.fillna(0, inplace = True)
combined_df.BsmtHalfBath.fillna(0, inplace = True)
combined_df.KitchenQual.fillna('Not_available', inplace = True)
combined_df.Functional.fillna('Not_available', inplace = True)
combined_df.FireplaceQu.fillna('No_fireplace', inplace = True)
combined_df.GarageType.fillna('No_garage', inplace = True)
combined_df.GarageYrBlt.fillna(combined_df['YearBuilt'], inplace = True)
combined_df.GarageFinish.fillna('No_garage', inplace = True)
combined_df.GarageCars.fillna(0, inplace = True)
combined_df.GarageArea.fillna(0, inplace = True)
combined_df.GarageCond.fillna('No_garage', inplace = True)
combined_df.MiscFeature.fillna('None', inplace = True)
combined_df.SaleType.fillna(combined_df.SaleType.value_counts().idxmax(), inplace = True)
combined_df["LotFrontage"] = combined_df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

#visualization for outliers
sns.scatterplot(train_df1_.GrLivArea, train_df1.SalePrice)
sns.scatterplot(train_df1_.LotFrontage, train_df1.SalePrice)
sns.scatterplot(train_df1_.BsmtFinSF1, train_df1.SalePrice)


#visualization for independent variable distribution
sns.distplot(train_df1_['GrLivArea'], kde=True, rug=False);
sns.distplot(np.log(combined_df['GrLivArea']), kde=True, rug=False).set(xlabel='LogTransformed GrLivArea');

sns.distplot(train_df1_['LotFrontage'], kde=True, rug=False);
sns.distplot(np.log(train_df1_['LotFrontage']), kde=True, rug=False).set(xlabel='LogTransformed 1stFlrSF');

#train_df1['GrLivArea'].sort_values()

#visualization of target variable distribution
sns.distplot(train_df1.SalePrice, kde=True, rug=False);
sns.distplot(np.log(y), kde=True, rug=False).set(xlabel='LogTransformed SalePrice');


#feature engineering
combined_df['BsmtUnSFAbv1'] = combined_df.BsmtUnfSF.apply(lambda x: x > 1)
combined_df['LowQualFinSFAbv1'] = combined_df.LowQualFinSF.apply(lambda x: x > 0)
combined_df['Remodel'] = pd.DataFrame(combined_df.YearBuilt != combined_df.YearRemodAdd)
combined_df['Remodel_years'] = pd.DataFrame(combined_df.YearRemodAdd - combined_df.YearBuilt)
combined_df['GrYrAfter'] = pd.DataFrame(combined_df.YearBuilt != combined_df.GarageYrBlt)
combined_df['LotFrontage'],_ = boxcox(combined_df.LotFrontage)
combined_df['GrLivArea'] = np.log(combined_df.GrLivArea)
combined_df['MSSubClass'] = combined_df['MSSubClass'].astype(object)
combined_df['1stFlrSF'] = np.log(combined_df['1stFlrSF'])
combined_df['after_1980'] = combined_df['YearRemodAdd'] > 1985
combined_df['before_1980'] = combined_df['YearRemodAdd'] < 1960
#dropping one feature
combined_df = combined_df.drop(['GarageYrBlt'], axis=1)

#get dummy variables
combined_dummy = pd.get_dummies(combined_df, drop_first = True)

#splitting df back to traing and test dfs
train_df1 = combined_dummy.iloc[:1460]
test_df1 = combined_dummy.iloc[1460:]

#sorting for outliers
train_df1 = train_df1[train_df1.BsmtFinSF1 < 5000]
train_df1 = train_df1[train_df1.GrLivArea < 8.44]
train_df1 = train_df1[train_df1.LotFrontage < 40]
# =============================================================================
# #get dummy variables
# train_df1 = pd.get_dummies(train_df1, drop_first = True)
# =============================================================================

#separating dependent and independent variables
y = train_df1.SalePrice.astype(float)
X = train_df1.loc[:, train_df1.columns != 'SalePrice'].astype(float)



warnings.filterwarnings('ignore')

#Grid search
# =============================================================================
# grid_para_tree = {
#     "n_estimators": [10,100,400],
#     "min_samples_leaf": [1,2,3],
#     "max_depth": [10,30,70],
#     "min_samples_split":[1.0,2,3],
#     "max_features":['auto','sqrt']}
# =============================================================================
# =============================================================================
# grid_para_tree = {
#     "alpha":[10, 100, 1000],
#     "max_iter":[None, 100, 1000, 10000, 100000],
#     "solver":['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
#     "tol":[0.00001, 0.0001, 0.001, 0.01, 0.1]}
# =============================================================================
   
# grid = RandomizedSearchCV(rf, parm_distribution = grid_para_tree, cv=10,
#                          scoring='neg_mean_squared_error', n_jobs=-1)
# =============================================================================
# grid = GridSearchCV(ridge, grid_para_tree, cv=10,
#                          scoring='neg_mean_squared_error', n_jobs=-1)
# =============================================================================
grid.fit(X,np.log(y))
grid.best_params_

#....................model parameters
eNet = ElasticNet(fit_intercept = True, l1_ratio = 0.6,alpha = .0001,normalize = False,
                  max_iter = 25, precompute = True, selection = 'random')
ridge = Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, solver='svd', tol = 1e-05)

lasso = Lasso(alpha=.006, copy_X=True, fit_intercept=True, max_iter=25,normalize=False,
              positive=False, precompute=False, random_state=None,selection='cyclic', tol=0.107)
gbr = GradientBoostingRegressor(learning_rate = .1, n_estimators = 400,subsample = 0.7,
                                max_depth = 2, min_samples_leaf= 2, min_samples_split = 3)
rf = RandomForestRegressor(bootstrap = True,
                        max_depth = 30, min_samples_leaf = 2, min_samples_split = 3,
                        n_estimators = 100, oob_score = True, warm_start = True)
svr = SVR(gamma = .000001, C = 100)
#higher C correspond to more hard line margin
xgb = XGBRegressor(estimators = 400, gamma = 0, max_depth = 3, subsample = .8, alpha= 0)

models = [eNet,rf,ridge,lasso,gbr, svr, xgb]

#.....................root mean squared error
for model in models:
    model.fit(X,np.log(y))
    mse = cross_val_score(model, X, np.log(y), cv=10, scoring = 'neg_mean_squared_error').mean()
    rmse = (mse*-1)**.5
    strmodel = str(model)
    end_index = strmodel.index('(')
    print('root MSE of {} : '.format(strmodel[:end_index]) + str(round(rmse,6)))

#...................R squared
for model in models:
    R2 = model.score(X,np.log(y))
    strmodel = str(model)
    end_index = strmodel.index('(')
    print('R-squared of {} : '.format(strmodel[:end_index]) + str(round(R2,6)))



#prediction df for kaggle submission
subm2 = pd.read_csv('sample_submission.csv')
subm2.drop('SalePrice', axis=1, inplace=True)
test_df = test_df1.loc[:, test_df1.columns != 'SalePrice'].astype(float)
ridge_pred = pd.Series(ridge.predict(test_df))
subm2['SalePrice'] = ridge_pred.apply(lambda price: exp(price))
subm2.to_csv('subm2.csv',index=False)


#...........feature importance
feature_importance = list(zip(X.columns, rf.feature_importances_))
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_importance = np.array(feature_importance, dtype=dtype)
feature_sort = np.sort(feature_importance, order='importance')[::-1]
name, score = zip(*list(feature_sort))
pd.DataFrame({'name':name,'score':score})[:25].plot.bar(x='name', y='score')
plt.title('Feature Importance from Random Forest')


#....feature importance
from matplotlib.pyplot import cm
feature_importance = list(zip(X.columns, gbr.feature_importances_))
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_importance = np.array(feature_importance, dtype=dtype)
feature_sort = np.sort(feature_importance, order='importance')[::-1]
name, score = zip(*list(feature_sort))
colors = cm.hsv(y / float(max(y)))
pd.DataFrame({'name':name,'score':score})[:15].plot.bar(x='name', y='score', color=colors)
plt.title('Feature Importance from Gradient Boosting Regressor')





