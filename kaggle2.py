import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score

from IPython.display import display, HTML

def show_dataframe(X, rows=2):
    display(HTML(X.to_html(max_rows=rows)))

train = pd.read_csv('D:/study/data/train.csv')
test = pd.read_csv('D:/study/data/test.csv')

print(train.shape) # (300000, 25)
print(test.shape) # (200000, 24)

target = train['target']
train_id = train['id']
test_id = test['id']
train.drop(['target', 'id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

print(train.head())
print(test.head())

print(train.shape) # (300000, 23)
print(test.shape)  # (200000, 23)
 
cats_cols = []
for c in train.columns:
    if train[c].dtype == 'object':
        cats_cols.append(c)
print('Categorical columns :', cats_cols)
#Categorical columns : ['bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 
# 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

categorical_features_indices = np.where(test.dtypes != np.float)[0]
print('Categorical Feature Indices:', categorical_features_indices)
# Categorical Feature Indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
'''
d = {}; done = []
cols = train.columns.values
for c in cols: d[c] = []
for i in range(len(cols)):
    if i not in done:
        for j in range(i+1, len(cols)):
            if all(train[cols[i]] == train[cols[j]]):
                done.append(j)
                d[cols[i]].append(cols[j])
dub_cols = []
for k in d.keys():
    if len(d[k]) > 0:
        dub_cols += d[k]
print('Dublicates:', dub_cols)

const_cols = []
for k in d.keys():
    if len(d[k]) > 0:
        dub_cols += d[k]
print('Dublicates:', dub_cols)

cat_cols = []
for c in cols:
    if len(train[c].unique()) == 1:
        const_cols.append(c)
print('Constant cols:', const_cols)

plt.figure(figsize=(20,32))
for i in range(len(cat_cols)):
    c = cat_cols[i]
    
    means = train.groupby(c).y.mean()
    stds = train.groupby(c).y.std().fillna(0)
    maxs = train.groupby(c).y.max()
    mins = train.groupby(c).y.min()
    
    ddd = pd.concat([means, stds, maxs, mins], axis=1); 
    ddd.columns = ['means', 'stds', 'maxs', 'mins']
    ddd.sort_values('means', inplace=True)
    
    plt.subplot(8,2,2*i+1)
    ax = sns.countplot(train[c], order=ddd.index.values)
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.0f}'.format(y), (x.mean(), y), ha='center', va='bottom')
    
    plt.subplot(8,2,2*i+2)
    plt.fill_between(range(len(train[c].unique())), 
                     ddd.means.values - ddd.stds.values,
                     ddd.means.values + ddd.stds.values,
                     alpha=0.3
                    )
    plt.xticks(range(len(train[c].unique())), ddd.index.values)
    plt.plot(ddd.means.values, color='b', marker='.', linestyle='dashed', linewidth=0.7)
    plt.plot(ddd.maxs.values, color='g', linestyle='dashed', linewidth=0.7)
    plt.plot(ddd.mins.values, color='r', linestyle='dashed', linewidth=0.7)
    plt.xlabel(c + ': Maxs, Means, Mins and +- STDs')
    plt.ylim(55, 270)

train['eval_set'] = 0; test['eval_set'] = 1
df = pd.concat([train, test], axis=0, copy=True)
df.reset_index(drop=True, inplace=True)

def add_new_cols(x):
    if x not in new_col.keys():
        return int(len(new_cols.keys())/2)
    return new_col[x]

for c in cats_cols:
    new_col = train.groupby(c).y.mean().sort_values().reset_index()
    new_col = new_col.reset_index().set_index(c).drop('y', axis=1)['index'].to_dict()
    df[c + '_new'] = df[c].apply(add_new_col)

df_new = df.drop(cats_cols, axis=1)

show_dataframe(df_new, 5)

x = df.drop(list((set(const_cols) | set(dub_cols) | set(cat_cols))), axis=1)

x_train = x[x.eval_set == 0]
y_train = x_train.pop('y');
x_train = x_train.drop(['eval_set', 'ID'], axis=1)

x_test = x[x.eval_set == 1]
x_test = x_test.drop(['y', 'eval_set', 'ID'], axis=1)

y_mean = y_train.mean()

print('shape x_train: {}\nshape x_test: {}'.format(x_train.shape, x_test.shape))
'''

xgb_params = {
     'n_trees': 100,
     'eta': 0.005,
     'max_depth': 3,
     'subsample': 0.95,
     'colsamle_bytree': 0.6,
     'objective': 'rmse',
     'base_score':np.log(y_mean),
     'silent':1
}

dtrain = xgb.DMatrix(x_train, np.log(y_train))
dtest = xgb.DMatrix(x_test)

def the_metric(y_pred, y):
        y_true = y.get_label()
        return 'r2', r2_score(y_true, y_pred)
cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=2000, nfold=3,
early_stopping_rounds=50, feval=the_metric, verbose_eval=100, show_stdv=False)

num_boost_rounds = len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))

model = xgb.train(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_train_pred = np.exp(model, predict(dtrain))
y_pred = np.exp(model.predict(dtest))

print('First 5 predicted test values:', y_pred[:5])
# First 5 predicted test values: [  79.28653717   94.7334671    79.35385895   78.36151123  110.52820587]

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
train_scores = cv_result['train-r2-mean']
train_stds = cv_result['train-r2-std']
plt.plot(train_scores, color='green')
plt.fill_between(range(len(cv_result)), train_scores - train_stds, 
                 train_scores + train_stds, alpha=0.1, color='green')
test_scores = cv_result['test-r2-mean']
test_stds = cv_result['test-r2-std']
plt.plot(test_scores, color='red')
plt.fill_between(range(len(cv_result)), test_scores - test_stds,
                 test_scores + test_stds, alpha=0.1, color='red')
plt.title('Train and test cv scores (R2)')

plt.subplot(1, 4, 2)
plt.title('True vs. Pred. train')
plt.plot([80, 265], [80, 265], color='g', alpha=0.3)
plt.scatter(x=[np.mean(y_train)], y=[np.mean(y_train_pred)], marker='o', color='red')
plt.xlabel('Real train'); plt.ylabel('Pred. train')

plt.subplot(1, 4, 3)
sns.distplot(y_train, kde=False, color='g')
sns.distplot(y_train_pred, kde=False, color='b')
plt.title('Distr. of train and pred. test')

plt.figure(figsize=(18, 1))
plt.plot(y_train_pred[:200], color='r', linewidth=0.7)
plt.plot(y_train[:200], color='g', linewidth=0.7)
plt.title('First 200 true and pred. trains')

print('Mean error =', np.mean(y_train - y_train_pred))
print('Train r2 =', r2_score(y_train, y_train_pred))
# Mean error = 0.30814267228068215
# Train r2 = 0.6008123861