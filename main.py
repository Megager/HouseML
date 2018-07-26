import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

houses_train = pd.read_csv('train.csv')
houses_train.reindex(np.random.permutation(houses_train.index))

#correlation
corrmat = houses_train.corr()
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(houses_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#missing data
total = houses_train.isnull().sum().sort_values(ascending=False)
percent = (houses_train.isnull().sum()/houses_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
houses_train.drop((missing_data[missing_data['Total'] > 1]).index,1,inplace=True)
houses_train.drop(houses_train.loc[houses_train['Electrical'].isnull()].index, inplace=True)

#outliars
print(houses_train.sort_values(by='GrLivArea', ascending=False)[:2])
houses_train.drop(houses_train[houses_train['Id'] == 1299].index,inplace=True)
houses_train.drop(houses_train[houses_train['Id'] == 524].index,inplace=True)

#normality
houses_train['SalePrice'] = np.log(houses_train['SalePrice'])
houses_train['GrLivArea'] = np.log(houses_train['GrLivArea'])

#dummies
houses_train = houses_train.drop("Id", 1)
dummies_words = ['Gd', 'Pave', 'Shed', 'GdPrv', 'NA']
for col in houses_train.columns:
    if isinstance(houses_train[col][1], str) | pd.Series(dummies_words).isin(houses_train[col]).any():
        houses_train[col] = pd.get_dummies(houses_train[col])

    houses_train[col] = houses_train[col].replace(r'\s+', np.nan, regex=True)
    houses_train[col] = houses_train[col].fillna(0)


print(houses_train.head().to_string())
train = houses_train.sample(frac=0.8)
cross_validation = houses_train.drop(train.index)
train_results = train["SalePrice"]
cross_validation_results = cross_validation["SalePrice"]
train.drop("SalePrice", axis=1, inplace=True)
cross_validation.drop("SalePrice", axis=1, inplace=True)
model = RandomForestRegressor(1000)
model.fit(train, train_results)
print(model.score(cross_validation, cross_validation_results))
