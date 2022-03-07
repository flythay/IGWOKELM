import time
import numpy as np
from os import walk
import pandas as pd
from numpy import trapz
from numpy import linalg
from random import sample
from sklearn import metrics
from numpy.random import seed
from scipy.sparse import identity
from sklearn import preprocessing
from scipy.spatial import distance
from scipy.linalg.blas import sgemm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

from funcoes import DADOS, IGWO_KELM

def IGWO_KELM_FINAL(pasta,NMAX,BETAMAX,OMEGAMAX):
    DIC_DEV={}
    DIC_VAL={}
    [DIC_DEV,DIC_VAL]=DADOS(pasta)
    t=time.time() 
    TEMPO_TOTAL=[]
    column_names = ['C','GAMA','AUC','TEMPO','BASE']
    RESULTADO = pd.DataFrame(columns = column_names)
    col_names = ['BASE','TEMPO']
    TEMPO_TOTAL = pd.DataFrame(columns = col_names)
    for key in DIC_DEV:   
        CHAVE=str(key)
        print(CHAVE)
        [df,tempo]=IGWO_KELM(key,NMAX,BETAMAX,OMEGAMAX,DIC_DEV,DIC_VAL)
        TEMPO_TOTAL.append([CHAVE, tempo]) 
        RESULTADO = pd.concat([RESULTADO, df], ignore_index=True) 
    TEMPO_MAIN=time.time() - t
    return RESULTADO, TEMPO_TOTAL, TEMPO_MAIN

pasta=r'C:/Users/Thayse/Desktop/Dados/Imbalanced'
NMAX=30
BETAMAX=10
OMEGAMAX=10

[RESULTADO, TEMPO_TOTAL, TEMPO_MAIN] = IGWO_KELM_FINAL(pasta,NMAX,BETAMAX,OMEGAMAX)

print("TEMPO_MAIN: \n",TEMPO_MAIN)

writer = pd.ExcelWriter('RESULTADO.xlsx', engine='xlsxwriter')
RESULTADO.to_excel(writer, sheet_name='BASE')
writer.save()

writer = pd.ExcelWriter('TEMPO_TOTAL.xlsx', engine='xlsxwriter')
TEMPO_TOTAL.to_excel(writer, sheet_name='BASE')
writer.save()