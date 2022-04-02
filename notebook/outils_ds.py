import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(fname, separator):
    df = pd.read_csv(fname, sep=separator)
    #print(df.head())
    return df

def info_data(data):
    print(data.shape)
    print(data.info())
    print(data.columns.tolist())
    
def plotNanDistribution(data, threshold, text_pos=-4.5):
    plt.figure(figsize=(30, 10))
    data.isnull().mean().sort_values(ascending=False).plot.bar(color='black', alpha=0.5)
    plt.axhline(y=threshold, color='b', linestyle='-')
    plt.text(text_pos, threshold, '< '+str(threshold), color='b')
    plt.ylabel('Pourcentage', fontdict={'fontsize' : 18})
    plt.title('Moyenne des valeurs manquantes par colonne', fontsize=25, weight='bold' )
    plt.show()
    
def printFullColumns(data, threshold, keep_cols=[]):
    data_cols=data.columns.tolist()
    for col in data_cols:
        perc_data_NaN=data[col].isnull().sum()/len(data.index)*100
        if perc_data_NaN < threshold*100:
            keep_cols.append(col)
    print(keep_cols)
    print('il nous reste :', len(keep_cols), 'colonnes.')
    return keep_cols

def plotCleaning(data, data_clean):  
    data_col=[data_clean.shape[1], data.shape[1]-data_clean.shape[1]]
    data_row=[data_clean.shape[0], data.shape[0]-data_clean.shape[0]]
    fig,ax=plt.subplots(1,2) 
    fig.subplots_adjust(hspace=0.4,wspace=1) 
    _=ax[0].pie(data_col, labels=['Colonnes gardées', 'Colonnes supprimées'], explode=(0,0.1), autopct='%1.1f%%')
    _=ax[1].pie(data_row, labels=['Lignes gardées', 'lignes supprimées'], explode=(0,0.1), autopct='%1.1f%%')
    _=fig.suptitle('Ratio', fontsize=30)
    _=ax[0].set(title='Distribution des colonnes')
    _=ax[1].set(title='Distribution des lignes')
    plt.savefig('postnet.png')
    
def cleanData(data, threshold, mask):
    print("Taille du tableau avant le nettoyage", data.shape)
    if mask==None:
        data_tmp=data
    else: 
        data_tmp=data[mask]
    data_tmp=data_tmp.drop_duplicates()
    print("Taille du tableau après le drop", data_tmp.shape)
    threshold_valid=(1-threshold)*len(data_tmp.index)
    # apply dropna on all columns except excluded_cols
    cols = data_tmp.columns.tolist()
    res = data_tmp[cols].dropna(thresh=threshold_valid, axis=1)
    print("Taille du tableau après le nettoyage", res.shape)
    print(" {} colonnes supprimées et {} lignes supprimées".format(data.shape[1] - res.shape[1], data.shape[0] - res.shape[0]))
    plotCleaning(data, res)
    return res
