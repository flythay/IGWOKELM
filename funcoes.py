import os
import time
import numpy as np
from os import walk
import pandas as pd
from random import sample
from sklearn import metrics
from numpy.random import seed
from scipy.sparse import identity
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import euclidean_distances

SEMENTE=0

def MUDA_SEMENTE():
    global SEMENTE
    SEMENTE += 1
    return SEMENTE


def DADOS(folder):
    nome=[]
    caminho=[]
    for root, dirs, files in os.walk(folder):
        for name in files:
            aux=name.split(".dat")[0]
            nome.append(aux)
            path=os.path.abspath(os.path.join(root, name))
            caminho.append(path)
    train=[s for s in nome if 'tra' in s]
    test=[s for s in nome if 'tst' in s]
    train_path=[s for s in caminho if 'tra' in s]
    test_path=[s for s in caminho if 'tst' in s]
    bases=pd.DataFrame({'NAME_TRAIN': train, 
                        'PATH_TRAIN': train_path, 
                        'NAME_TEST': test, 
                        'PATH_TEST': test_path})
    n=len(bases)
    i=0
    dev=[]
    val=[]
    DEV={}
    VAL={}
    while i<n:
        caminho_dev=bases.iloc[i,1]
        caminho_val=bases.iloc[i,3]
        chave=bases.iloc[i,0]
        CHAVE=chave[:-3]
        dev=pd.read_csv(caminho_dev, delimiter=",",header=None,comment="@")
        val=pd.read_csv(caminho_val, delimiter=",",header=None,comment="@")   
        DEV[CHAVE] = [dev]
        VAL[CHAVE] = [val]    
        i+=1
    return DEV,VAL


def TRATADADOS(base):
    df=pd.DataFrame(base[0])
    aux=df.apply(LabelEncoder().fit_transform)
    X=aux.iloc[:,:-1]
    Y=aux.iloc[:,-1]
    return X,Y


def CRIA_PARAMETROS(intervalo):
    POSITIVO=[]
    NEGATIVO=[]
    PARAMETROS=[]
    i=1
    while i<intervalo:
        NEGATIVO=pow(2,-i)
        POSITIVO=pow(2,i)
        PARAMETROS.append(NEGATIVO)
        PARAMETROS.append(POSITIVO)
        i=i+1
    PARAMETROS.sort()
    return PARAMETROS


def KELM(DEV,Y_DEV,VAL,Y_VAL,C,GAMA):
    t=time.time() 
    n=len(Y_DEV)
    K=rbf_kernel(DEV,DEV,GAMA)
    IC=(1/C)*np.identity(n)
    INV=np.linalg.inv(K+IC)
    COEF=np.dot(INV,Y_DEV)
    KVAL=rbf_kernel(DEV,VAL,GAMA)
    PRED=np.matmul(KVAL.T,COEF)
    AUC=metrics.roc_auc_score(Y_VAL, PRED)
    TEMPO=time.time() - t
    return AUC, TEMPO


def CRIALOBOS(DEV,Y_DEV,VAL,Y_VAL,C,GAMA):
    i=0
    j=0
    n=len(C)
    m=len(GAMA)
    PACK=[]
    LOBOS=[]
    while i<n:
        c=C[i]
        while j<m:
            g=GAMA[j]
            [AUC,TEMPO]=KELM(DEV,Y_DEV,VAL,Y_VAL,c,g)
            PACK.append([c,g,AUC,TEMPO])
            j+= 1
        j=0
        i+= 1
        LOBOS=pd.DataFrame(PACK,columns=['C','GAMA','AUC','TEMPO'])
        LOBOS.sort_values(by=['AUC'],inplace=True,ascending=[False])   
    return LOBOS


def LOBOSBETA(DEV,Y_DEV,VAL,Y_VAL,BETA,B): 
    i=0
    D=distance.euclidean(BETA.iloc[0,:2],BETA.iloc[1,:2])
    AUC_alpha=BETA.iloc[0,2]
    BETAS=BETA
    beta=[]
    data=[]
    ALPHAB=BETA.iloc[0,:2]
    while i<B:
        SEED=MUDA_SEMENTE()
        np.random.seed(SEED)
        r=np.random.uniform(0,1,size=2)
        beta=abs(ALPHAB+2*D*r-D)
        [tempo,auc]=KELM(DEV,Y_DEV,VAL,Y_VAL,beta[0],beta[1])
        data = {'C': [beta[0]],
	            'GAMA': [beta[1]],
	            'AUC': [auc],
                'TEMPO': [tempo]}
        BETAS.append(data,ignore_index=True)
        BETAS.sort_values(by=['AUC'],ascending=False) 
        if (auc>AUC_alpha):
            print("Entrou")
            ALPHA=BETAS.iloc[0,:2]
            D=distance.euclidean(ALPHA,BETAS.iloc[0,:2])
        i+=1
    return BETAS


def LOBOSDELTA(DEV,Y_DEV,VAL,Y_VAL,LOBOS,NOMEGA):
    i=0
    delta=[]
    DELTAS=[]
    deltas=[]
    alpha=LOBOS.iloc[0,:2]
    n=len(LOBOS)
    N=n-NOMEGA
    DELTAS=LOBOS.iloc[1:N,:2]
    SEED=MUDA_SEMENTE()
    np.random.seed(SEED)
    r=np.random.uniform(0,1,size=2)
    while i<(N-1):
        tal=2*(1-i/N)         
        A=2*tal*r[0]-tal
        C=2*r[1]
        L=abs(C*alpha-DELTAS.iloc[i,:2])
        delta=abs(alpha-A*L)
        [tempo,auc]=KELM(DEV,Y_DEV,VAL,Y_VAL,delta[0],delta[1])
        deltas.append([delta[0],delta[1],auc,tempo])
        i+=1
    DELTAS=pd.DataFrame(deltas,columns=['C','GAMA','AUC','TEMPO'])  
    return DELTAS


def LOBOSOMEGA(DEV,Y_DEV,VAL,Y_VAL,n):
    li=pow(2,-10)
    ls=pow(2,10)
    i=0
    omegas=[]
    while i<n:
        SEED=MUDA_SEMENTE()
        np.random.seed(SEED)
        D=np.random.uniform(0,1,size=2)
        omegaC=(ls-li)*D[0]+ls
        omegaGAMA=(ls-li)*D[1]+ls
        [auc,tempo]=KELM(DEV,Y_DEV,VAL,Y_VAL,omegaC,omegaGAMA)
        omegas.append([omegaC,omegaGAMA,auc,tempo])
        i+=1
    OMEGAS=pd.DataFrame(omegas,columns=['C','GAMA','AUC','TEMPO'])
    return OMEGAS



def IGWO_KELM(key,NMAX,BETAMAX,OMEGAMAX,DIC_DEV,DIC_VAL):
    C=CRIA_PARAMETROS(10)
    GAMA=C
    t=time.time() 
    i=0
    train=DIC_DEV[key]
    test=DIC_VAL[key]
    [DEV,Y_DEV]=TRATADADOS(train)
    [VAL,Y_VAL]=TRATADADOS(test)
    LOBOS=CRIALOBOS(DEV,Y_DEV,VAL,Y_VAL,C,GAMA)
    DIST=LOBOS.iloc[:2,:]
    FINAL=LOBOS
    FINAL['BASE'] = str(key)
    while i<NMAX:
        print(i)
        BETAS=LOBOSBETA(DEV,Y_DEV,VAL,Y_VAL,DIST,BETAMAX)
        FINAL.append([BETAS])
        FINAL.sort_values(by=['AUC'],inplace=True,ascending=False)
        DELTAS=LOBOSDELTA(DEV,Y_DEV,VAL,Y_VAL,FINAL,OMEGAMAX)
        FINAL.append([DELTAS])
        FINAL.sort_values(by=['AUC'],inplace=True,ascending=False)
        OMEGAS=LOBOSOMEGA(DEV,Y_DEV,VAL,Y_VAL,OMEGAMAX)
        FINAL.append([OMEGAS])
        FINAL.sort_values(by=['AUC'],inplace=True,ascending=False)
        ALPHA=FINAL.iloc[0,:2]
        MELHORDELTA=FINAL.iloc[1,:2]
        CHECK=FINAL.iloc[0,2]
        if CHECK == 1:
            break
        i=i+1
    TEMPO_TOTAL=time.time() - t
    return FINAL,TEMPO_TOTAL

# cm = confusion_matrix(y_test, predicted)

