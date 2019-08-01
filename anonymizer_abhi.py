import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import warnings
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import brentq
import chaospy as cp
import statsmodels.api as sm

class KDE(cp.Dist):
    def __init__(self, data):
        self.data = data
        self.kdes = [sm.nonparametric.KDEMultivariate( data=data[:, :1], var_type="c", bw="cv_ml")] + [ sm.nonparametric.KDEMultivariateConditional(endog=[data[:, 1]], exog=list(data[:, :1].T), dep_type='c', indep_type='c'*1, bw='cv_ml')] + [ sm.nonparametric.KDEMultivariateConditional( endog=[data[:, 2]], exog=list(data[:, :2].T), dep_type='c', indep_type='c'*2, bw='cv_ml') ] + [ sm.nonparametric.KDEMultivariateConditional( endog=[data[:, 3]], exog=list(data[:, :3].T), dep_type='c', indep_type='c'*3, bw='cv_ml') ] + [ sm.nonparametric.KDEMultivariateConditional( endog=[data[:, 4]], exog=list(data[:, :4].T), dep_type='u', indep_type='c'*4, bw='cv_ml') ]
        cp.Dist.__init__(self)
    
    def _cdf(self, q):
        out = [self.kdes[0].cdf(q[0])]
        out += [
            self.kdes[idx].cdf(endog_predict=q[idx], exog_predict=q[:idx].T)
            for idx in range(1, len(self))
        ]
        out = np.array(out)
        out = out.reshape(q.shape)
        return out
    def _pdf(self, q):
        out = [self.kdes[0].pdf(q[0])]
        out += [
            self.kdes[idx].pdf(endog_predict=q[idx], exog_predict=q[:idx].T)
            for idx in range(1, len(self))
        ]
        out = np.array(out)
        out = out.reshape(q.shape)
        return out
    def _bnd(self, x):
        return np.min(self.data, 0), np.max(self.data, 0)
    def __len__(self):
        return self.data.shape[-1]


df = pd.read_csv('tweets_original.tsv',delimiter='\t',encoding='utf-8')
df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Compound", "Negative" ,"Neutral", "Positive"], how="all")
df.round(3)
le = preprocessing.LabelEncoder()
le.fit(df['Locations'])
#print(le.classes_)
df['Locations'] = le.transform(df['Locations'])
#dic = dict(zip(le.classes_, le.transform(le.classes_)))
dic = dict(zip(le.transform(le.classes_),le.classes_))
print(df.head())

comp = df['Compound']
pos = df['Positive']
neg = df['Negative']
neu = df['Neutral']
loc = df['Locations']

D = np.vstack((comp, pos, neg, neu, loc)).T
print(D)
cp_kde = KDE(D)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

samples = cp.Iid(cp.Uniform(), 5).sample(len(df))
R = cp.approximation.approximate_inverse(cp_kde, samples, iterations=100, tol=1e-5)

cols = ["Compound", "Negative" ,"Neutral", "Positive", "Locations"]
df1 = pd.DataFrame(R.T, columns=cols)
df1['Compound'] = df1['Compound'].round(3)
df1['Positive'] = df1['Positive'].round(3)
df1['Negative'] = df1['Negative'].round(3)
df1['Neutral'] = df1['Neutral'].round(3)

df1['Locations'] = df1['Locations'].round().astype(int)

df1['Locations']=df1['Locations'].map(dic).fillna('EARTH')

df2 = df.drop(['Compound', 'Negative', 'Neutral', 'Positive', 'Locations'],axis=1)
newdf = pd.concat([df1,df2], axis = 1)
newdf.to_csv('tweets_twins.csv',index=False)

