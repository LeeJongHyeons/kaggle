import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold


train = pd.read_csv('D:/study/data/train.csv')
test = pd.read_csv('D:/study/data/test.csv')

print(train.shape) # (20000, 25)
print(test.shape) # (200000, 24)

print(train.head())
print(test.head())

target = train['target']
train_id = train['id']
test_id = test['id']
train.drop(['target', 'id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

print(train.shape) #(30000, 23)
print(test.shape)  #(20000, 23)

'''
cats_all = [c for c in train.columns if c not in ['day', 'month', 'target', 'id']]
cats_obj = ['bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7',
'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

for col in cats_obj:
    le = LabelEncoder()
    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
    train[col] = le.transform(list(train[col].astype(str).values))
    test[col] = le.transform(list(test[col].astype(str).values))

lgb_params = {'num_leaves': 23,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'min_data_in_leaf': 50,
        'objective': 'binary',
        'max_depth': -1,
        'learning_rate': 0.008,
        'boosting_type': "gbdt",
        'bagging_seed': 11,
        "metric": 'auc',
        "verbosity": -1,
        'reg_alpha': 0.3899927210061127,
        'reg_lambda': 0.6485237330340494,
        'random_state': 47,
        
        }

folds = KFold(n_splits=3, shuffle=True, random_state=42)
print(folds.n_splits)
aucs = list()
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importances = pd.DataFrame()
feature_importances['feature'] = train.colums

#training_start_time = time()
for fold, (trn_idx, test_idx) in enumerate(folds.split(train, target)):
    #start_time = time()
    print('Training on fold {}'.format(fold + 1))
    
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[test_idx], label=target.iloc[test_idx])
    clf = lgb.train(lgb_params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds=300)
    oof[test_idx] = clf.predict(train.iloc[test_idx], num_iteration=clf.best_iteration)
    
    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    aucs.append(clf.best_score['valid_1']['auc'])
    
    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
    
    #print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 50)
print('Training has finished.')

print('Mean auc:', np.mean(aucs))
print('-' * 50)
'''