import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import *
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import codecs
import string

from sklearn.model_selection import GridSearchCV

# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

n_splits = 5

with codecs.open("./input/salesforse_12132.csv", "r", "Shift-JIS", "ignore") as file:
    train = pd.read_table(file, delimiter=",")

feature_to_use = [
"Additional_Collateral__c",
"Area_Manager__c",
"Average_Purchase_Price__c",
"Business_Type__c",
"Business_declared_BK__c",
"Cars_On_Lot__c",
"Dealer_License__c",
"Dealer_Vehicles_sold_per_month__c",
"Dealership_Type__c",
"Dealership_Years_in_Business__c",
"Dealership__c",
"Equity_Cur_Month__c",
"Equity_YTD__c",
"Gross_Sales_Cur_Month__c",
"Gross_Sales_YTD__c",
"Inventory_Asset_Cur_Month__c",
"Inventory_Asset_YTD__c",
"Inventory_Liability_Cur_Month__c",
"Inventory_Liability_YTD__c",
"Lawsuits_Liens__c",
"Lot_Size__c",
"Net_Profit_YTD__c",
"Net_Worth_Cur_Month__c",
"Net_Worth_YTD__c",
"Number_of_Principals__c",
"Photo_score__c",
"Tax_Lien_Filings__c",
"Total_Collections__c",
"Zip1__c",
"UCC_Filings_Manual_Search__c",
"Zillow_Value__c",
"of_vehicles_floored__c",
"Average_Days_to_Sale__c",
"Dealer_All_Equity__c",
"Min_FICO",
"Max_FICO",
"Home_Owner_Flag",
"Average_Bank_Balance",
"Average_Deposit",
"UW_Status__c",
"Credit_Line_Size__c",
]

train = train[feature_to_use]

### Category data handler
feature_cat_normal = [
"Area_Manager__c",
"Business_Type__c",
"Business_declared_BK__c",
"Dealership_Type__c",
"Dealership__c",
"Lawsuits_Liens__c",
]
train_cat_normal = train[feature_cat_normal]

for n_c, name1 in enumerate(feature_cat_normal):
    # Label Encode
    print ("Go to transform ", name1)
    lbl = LabelEncoder()
    train[name1] = lbl.fit_transform(train[name1].astype(str))

feature_name = ["Dealer_License__c"]
train[feature_name] = train[feature_name].applymap(lambda x: 0 if x == np.nan else 1)

feature_name = ["Zillow_Value__c"]
train[feature_name] = train[feature_name].applymap(lambda x: str(x).split('$')[1] if str(x).startswith('$') else x)
train[feature_name] = train[feature_name].applymap(lambda x: str(x).split('K')[0] if str(x).endswith('K') else x)
train[feature_name] = train[feature_name].applymap(lambda x: x if str(x).isnumeric() else 0)


feature_name = ["Dealership_Years_in_Business__c"]
ALPHA = string.ascii_letters
train[feature_name] = train[feature_name].applymap(lambda x: -1 if x == np.nan else x)
train[feature_name] = train[feature_name].applymap(lambda x: 0 if str(x).endswith(tuple(ALPHA)) else x)

feature_name = ["Zip1__c"]
NUMBER = string.digits
train[feature_name] = train[feature_name].applymap(lambda x: str(x)[0:2] if str(x).startswith(tuple(NUMBER)) else 0)
# print (train[feature_name])
#print (train[feature_name].describe())

feature_name = ["Average_Purchase_Price__c"]
train[feature_name] = train[feature_name].applymap(lambda x: str(x).split('$')[1] if str(x).startswith('$') else str(x))
train[feature_name] = train[feature_name].applymap(lambda x: str(x).split('-')[0] if str(x).find('-') else str(x))
train[feature_name] = train[feature_name].applymap(lambda x: str(x).split('.')[0] if str(x).find('.') else str(x))
train[feature_name] = train[feature_name].applymap(lambda x: str(x) if str(x).startswith(tuple(NUMBER)) else 0)
train[feature_name] = train[feature_name].applymap(lambda x: float(str(x).split(',')[0])*1000 if str(x).find(',') > 0 else str(x))

train.to_csv("salceforce_freature.csv", index=False)
train = pd.read_csv("salceforce_freature.csv")

print(train.shape)

### target encode
feature_name = ["UW_Status__c"]
trans_mapping = {
    'Activated': 1,
    'Ready for Activation':1,
    'Approved': 1,
    'Contract Sent' : 1,
    'In Contracting': 1,
    'Awaiting Decision':-1,
    'Contract Review':1,
    'Additional Stips Required for Contract':1,
    'Additional Stips required':-1,
    'Ready for Underwriting':-1,
    'Declined':0,
    'Withdrawn':-1,
    'Dead':-1 }
train[feature_name] = train[feature_name].applymap(lambda x: trans_mapping[x])
train = train[train["UW_Status__c"] > -1]

# col = [c for c in train.columns if c not in ['UW_Status__c','Credit_Line_Size__c']]
y_train_class = train['UW_Status__c']
y_train_regre = train['Credit_Line_Size__c']
train = train.drop(['UW_Status__c', 'Credit_Line_Size__c'], axis=1)

x1, x2, y1, y2 = model_selection.train_test_split(train, y_train_class, test_size=0.15, random_state=99)
for f in train.columns:
    if train[f].dtype=='object':
        print("here is object", f)

X_train = x1
y_train = y1
X_test = x2
y_test = y2

params = {'eta': 0.1,
          'max_depth': 5,
          'min_child_weight': 7,
          'gamma':0,
          'subsample': 0.9,
          'colsample_bytree': 0.9,
          'objective': 'binary:logistic',
          # 'reg_alpha': 0.01,
          #'objective': 'multi:softmax',
          #'num_class': 5,
          'eval_metric': 'auc',
          # 'learning_rate':0.1,
          # 'n_estimators': 1000,
          'early_stopping_rounds':50,
          'seed': 99,
          'gpu_id' : 0,
          'max_bin' : 16,
          'tree_method' : 'gpu_hist',
          'silent': True}

xgb_model = XGBClassifier(**params)

### parameter tuning

# # res = xgb_model.fit(X_train, y_train, eval_metric=['auc'], early_stopping_rounds="50", verbose=True)
# res = xgb_model.fit(X_train, y_train, eval_metric=['auc'], verbose=True)
#
# # Predict training set:
# dtrain_predictions = xgb_model.predict(X_test)
# dtrain_predprob = xgb_model.predict_proba(X_test)[:, 1]
#
# # Print model report:
# print ("\nModel Report")
# print ("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, dtrain_predictions) )
# print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, dtrain_predprob) )

# Predict on testing data:

### max_depth and min_child_weight
# param_test1 = {
#     # "'max_depth':range(3,10,2),
#     # 'min_child_weight':range(1,6,2)
#     'max_depth':[3,4,5,6,7,8,9,10],
#     'min_child_weight':[1,2,3,4,5,6,7]
# }
# gsearch1 = GridSearchCV(estimator = xgb_model,
#                        param_grid = param_test1, scoring='roc_auc',iid=False, cv=5)
# gsearch1.fit(X_train, y_train)
#
#
# ### gamma
# param_test2 = {
#     'gamma':[i/10.0 for i in range(0,5)]
# }
#
# # gsearch2 = GridSearchCV(estimator = xgb_model,
# #                         param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
# # gsearch2.fit(X_train, y_train)
#
# ### subsample and colsample_bytree
# param_test3 = {
#     'subsample':[i/100.0 for i in range(75,100, 5)],
#     'colsample_bytree':[i/100.0 for i in range(75,100, 5)]
# }
#
# # gsearch3 = GridSearchCV(estimator = xgb_model,
# #                         param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
# # gsearch3.fit(X_train, y_train)
#
# ### reg_alpha
# param_test4 = {
#     'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
# }
# gsearch4 = GridSearchCV(estimator = xgb_model,
#                         param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(X_train, y_train)

### eta
# param_test5 = {
#     'eta':[0.001, 0.002, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
# }
# gsearch5 = GridSearchCV(estimator = xgb_model,
#                         param_grid = param_test5, scoring='roc_auc', iid=False, cv=5)
# gsearch5.fit(X_train, y_train)


# print(gsearch1.grid_scores_)
# for params, mean_score, scores in gsearch1.grid_scores_:
#     print("%0.4f (+/-%0.03f) for %r "
#           % (mean_score, scores.std()*2, params))
# print("gogog")
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)


##### cross fit
kf = StratifiedKFold(n_splits=n_splits, random_state=22)
kf.get_n_splits(X_train, y_train)
y_preds_train = []
y_preds_test = []
for k, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):

    xgb_model.fit(X_train.iloc[train_idx],
                  y_train.iloc[train_idx],
               eval_set=[(X_train.iloc[valid_idx], y_train.iloc[valid_idx])],
               eval_metric='auc',
               verbose=False,
               early_stopping_rounds=50)

    y_pred_train = xgb_model.predict_proba(X_train.iloc[valid_idx])[:,1]
    y_pred_test = xgb_model.predict(X_test)
    # y_pred_k = np.argmax(y_pred_test, axis=1)
    accu = accuracy_score(y_test, y_pred_test)
    print('fold[{:>3d}]: accuracy = {:>.4f}'.format(k, accu))
    y_preds_train.append(y_pred_train)
    y_preds_test.append(y_pred_test)


col = [c for c in train.columns]
train["my_predict"] = xgb_model.predict(train[col])
train["my_predict_proba"] = xgb_model.predict_proba(train[col])[:,1]
cross_score = cross_val_score(xgb_model, x2, y2, cv=n_splits, scoring='accuracy')
print("    cross_score: %.5f" % (cross_score.mean()))

train_pred = xgb_model.predict(X_train[col])
accu = accuracy_score(train_pred, y_train)
print('Train : accuracy = {:>.4f}'.format(accu))

accu = accuracy_score(train["my_predict"], y_train_class)
print('Total : accuracy = {:>.4f}'.format(accu))
#
# importance = xgb_model.feature_importances_
# features = train.columns
# print(len(features))
#
# # for i, imp in enumerate(importance):
#     print (features[i],imp)
#     print (i)

# train = pd.concat([train, y_train_class, y_train_regre], axis=1)
# train.to_csv("my_res.csv", index=False)

# xgb.plot_importance(xgb_model)
# plt.show()

