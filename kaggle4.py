import pandas as pd
import numpy as np                      
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import base

df_train = pd.read_csv("D:/study/data/train.csv")
df_test = pd.read_csv("D:/study/data/test.csv")

print('Train data set has got {} rows and {} columns'. format(df_train.shape[0], df_train.shape[1]))
print('Test data set has got {} rows and {} columns'. format(df_test.shape[0], df_train.shape[1]))
                                  
#Train data set has got 300000 rows and 25 columns
#Test data set has got 200000 rows and 25 columns          

#print(df_train.head())                      
'''
   id  bin_0  bin_1  bin_2 bin_3 bin_4  nom_0      nom_1    nom_2    nom_3     nom_4      nom_5      nom_6      nom_7      nom_8      nom_9  ord_0        ord_1        ord_2 ord_3 ord_4 ord_5  day  month  target
0   0      0      0      0     T     Y  Green   Triangle    Snake  Finland   Bassoon  50f116bcf  3ac1b8814  68f6ad3e9  c389000ab  2f4cb3d51      2  Grandmaster         Cold     h     D    kr    2      2       0
1   1      0      1      0     T     Y  Green  Trapezoid  Hamster   Russia     Piano  b3b4d25d0  fbcb50fc1  3b6dd5612  4cd920251  f83c56c21      1  Grandmaster          Hot     a     A    bF    7      8       0
2   2      0      0      0     F     Y   Blue  Trapezoid     Lion   Russia  Theremin  3263bdce5  0922e3cb8  a6a36f527  de9c9f684  ae6800dd0      1       Expert     Lava Hot     h     R    Jc    7      2       0
3   3      0      1      0     F     Y    Red  Trapezoid    Snake   Canada      Oboe  f12246592  50d7ad46a  ec69236eb  4ade6ab69  8270f0d71      1  Grandmaster  Boiling Hot     i     D    kW    2      1       1
4   4      0      0      0     F     N    Red  Trapezoid     Lion   Canada      Oboe  5b0f5acd5  1fe17a1fd  04ddac2be  cb43ab175  b164b72a7      1  Grandmaster     Freezing     a     R    qP    7      8       0
'''

X = df_train.drop(['target'], axis=1)
y = df_train['target']

x = y.value_counts()
plt.bar(x.index, x)
plt.gca().set_xticks([0, 1])
plt.title('distribution of target variable')
plt.show()

from sklearn.preprocessing import LabelEncoder
train = pd.DataFrame()
label = LabelEncoder()
for c in  X.columns:
    if(X[c].dtype=='object'):
        train[c] = label.fit_transform(X[c])
    else:
        train[c] = X[c]
train.head(3)

print('Train data set has got {} rows and {} columns'. format(train.shape[0], train.shape[1]))
# Train data set has got 300000 rows and 24 columns

def logistic(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pre = lr.predict(X_test)
    print('Accuracy :', accuracy_score(y_test, y_pre))
    #Accuracy : 0.6901333333333334

logistic(train, y)        

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()

one.fit(X)
train = one.transform(X)

print('Train Data Set Has Got {} Rows and {} Columns'.format(train.shape[0], train.shape[1]))
# Train Data Set Has Got 300000 Rows and 316461 Columns

from sklearn.feature_extraction import FeatureHasher
X_train_hash = X.copy()
for c in X.columns:
    X_train_hash[c]=X[c].astype('str')
hashing = FeatureHasher(input_type='string')
train = hashing.transform(X_train_hash.values)
print('Train Data Set Has Got {} Rows and {} Columns'.format(train.shape[0], train.shape[1]))

X_train_stat = X.copy()
for c in X_train_stat.columns:
    if(X_train_stat[c].dtype=='object'):
        X_train_stat[c] = X_train_stat[c].astype('category')
        counts = X_train_stat[c].value_counts()
        counts = counts.sort_index()
        counts = counts.fillna(0)
        counts += np.random.rand(len(counts)) / 1000
        X_train_stat[c].cat.categories = counts
print(X_train_stat.head(3))

'''
id  bin_0  bin_1  bin_2          bin_3          bin_4          nom_0          nom_1          nom_2          nom_3  ...       nom_8      nom_9 ord_0         ord_1         ord_2         ord_3         ord_4        ord_5 day month
0      0      0      0  153535.000386  191633.000784  127341.000810   29855.000645   45979.000771   36942.000618  ...  271.000342  19.000250     2  77428.000848  33768.000292  24740.000673   3974.000179   506.000957   2     2

1      0      1      0  153535.000386  191633.000784  127341.000810  101181.000879   29487.000077  101123.000031  ...  111.000230  13.000718     1  77428.000848  22227.000283  35276.000690  18258.000390  2603.000769   7     8

2      0      0      0  146465.000074  191633.000784   96166.000994  101181.000879  101295.000312  101123.000031  ...  278.000692  29.000859     1  25065.000826  63908.000343  24740.000673  16927.000419  2572.000486   7     2
'''

print('Train Data Set Has Got {} Rows and {} Columns'.format(X_train_stat.shape[0], X_train_stat.shape[1]))
logistic(X_train_stat, y)

X_train_cyclic = X.copy()
columns=['day', 'month']
for col in columns:
    X_train_cyclic[col+'_sin'] = np.sin((2 * np.pi*X_train_cyclic[col]) / max(X_train_cyclic[col]))
    X_train_cyclic[col+'_cos'] = np.cos((2 * np.pi*X_train_cyclic[col]) / max(X_train_cyclic[col]))
X_train_cyclic = X_train_cyclic.drop(columns, axis=1)

X_train_cyclic[['day_sin', 'day_cos']].head(3)

one = OneHotEncoder()

one.fit(X_train_cyclic)
train = one.transform(X_train_cyclic)

print('Train Data Set Has Got {} Rows and {} Columns'.format(train.shape[0], train.shape[1]))

X_target = df_train.copy()                                                                                                          
X_target['day'] = X_target['day'].astype('object')
X_target['month'] = X_target['month'].astype('object')
for col in X_target.columns:
    if (X_target[col].dtype=='object'):
        target= dict ( X_target.groupby(col)['target'].agg('sum')/X_target.groupby(col)['target'].agg('count'))
        X_target[col]= X_target[col] = X_target[col].replace(target).values

print(X_target.head(4))

logistic(X_target.drop('target', axis=1), y)

X['target'] = y
cols=X.drop(['target', 'id'], axis=1).columns

X_fold = X.copy()
X_fold[['ord_0', 'day', 'month']] = X_fold[['ord_0', 'day', 'month']].astype('object')
X_fold[['bin_3','bin_4']]=X_fold[['bin_3','bin_4']].replace({'Y':1,'N':0,'T':1,"F":0})
kf = KFold(n_splits=5, shuffle=False, random_state=2019)
for train_ind, val_ind in kf.split(X):
    for col in cols:
            if(X_fold[col].dtype=='object'):
                    replaced = dict(X.iloc[train_ind][[col, 'target']].groupby(col)['target'].mean())
                    X_fold.loc[val_ind, col] = X_fold.iloc[val_ind][col].replace(replaced).values

print(X_fold.head())

'''
  id  bin_0  bin_1  bin_2  bin_3  bin_4     nom_0     nom_1     nom_2     nom_3     nom_4     nom_5  ...     nom_7     nom_8     nom_9     ord_0     ord_1     ord_2     ord_3     ord_4     ord_5       day     month target      
0   0      0      0      0      1      1  0.327356  0.360281  0.305929   0.24171  0.237044  0.350506  ...  0.225806   0.38009       0.5  0.334926  0.403542  0.259103  0.307031  0.211418  0.412888  0.323473  0.244538      0      
1   1      0      1      0      1      1  0.327356  0.290501  0.358107  0.289501  0.304566  0.386047  ...  0.300429  0.195402     0.125  0.278366  0.403542  0.327796  0.208194  0.185704  0.293144  0.341711  0.327219      0      
2   2      0      0      0      0      1  0.242135  0.290501  0.293881  0.289501  0.355844  0.275828  ...  0.193384  0.224771  0.166667  0.278366  0.316665  0.402135  0.307031  0.354919  0.208748  0.341711  0.244538      0      
3   3      0      1      0      0      1  0.350536  0.290501  0.305929  0.340791  0.328661   0.22403  ...  0.349432  0.269006  0.233333  0.278366  0.403542  0.361036  0.330519  0.211418  0.358066  0.323473  0.255791      1      
4   4      0      0      0      0      0  0.350536  0.290501  0.293881  0.340791  0.328661  0.310154  ...  0.288571   0.36478  0.137931  0.278366  0.403542  0.225265  0.208194  0.354919  0.410066  0.341711  0.327219      0      

'''


