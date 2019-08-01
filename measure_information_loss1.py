import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame 
from utils import best_fit_distribution
from utils import plot_result
import sys
import scipy.stats as st
import statsmodels as sm
import warnings
import matplotlib.pyplot as plt

def measure_entropy(data1, data2):
    y1, x1 = np.histogram(data1, bins=10000, range=(-3.9999, 3.9999), density=True)
    y2, x2 = np.histogram(data2, bins=10000, range=(-3.9999, 3.9999), density=True)
    vc1 = np.where(y1==0, 0.0001, y1) 
    vc2 = np.where(y2==0, 0.0001, y2)
    entropy = st.entropy(vc1, vc2)
    return (entropy)


df1 = pd.read_csv('tweets_original.tsv',delimiter='\t',encoding='utf-8')
df2 = pd.read_csv('tweets_twins.tsv',delimiter='\t',encoding='utf-8')

df = pd.merge(df1, df2, how='inner', on=['Uuid'])
df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Compound_x", "Negative_x" ,"Neutral_x", "Positive_x","Compound_y", "Negative_y" ,"Neutral_y", "Positive_y"], how="all")
df.round(3)


for c in ['Compound','Neutral','Negative','Positive']:
    data1 = df[c+'_x']
    data2 = df[c+'_y']
    print("Information loss in column: ", c, measure_entropy(data1, data2))

'''
locations = df.Locations_x.unique()
for L in locations:
    print('--------------------------------')
    print("Location : ", L)
    print('--------------------------------')
    
    com_x = df[df['Locations_x'] == L]['Compound_x']
    pos_x = df[df['Locations_x'] == L]['Positive_x']
    neg_x = df[df['Locations_x'] == L]['Negative_x']
    neu_x = df[df['Locations_x'] == L]['Neutral_x']

    com_y = df[df['Locations_y'] == L]['Compound_y']
    pos_y = df[df['Locations_y'] == L]['Positive_y']
    neg_y = df[df['Locations_y'] == L]['Negative_y']
    neu_y = df[df['Locations_y'] == L]['Neutral_y']

    print("Information loss in column Compound in location: ", L, measure_entropy(com_x.values, com_y.values))
    print("Information loss in column Positive in location: ", L, measure_entropy(pos_x.values, pos_y.values))
    print("Information loss in column Negative in location: ", L, measure_entropy(neg_x.values, neg_y.values))
    print("Information loss in column Neutral in location: ", L, measure_entropy(neu_x.values, neu_y.values))
''' 
