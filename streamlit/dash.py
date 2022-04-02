#import outils_ds as ods
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import numpy as np
import scipy.stats as st
#import yellowbrick
#import time
#import xgboost as xgb
#import lightgbm as lgb
#import re
import shap
import streamlit as st
#import sklearn
import pickle
#import xgboost
from xgboost import plot_importance
import requests
#import plotly
#import plotly.express as px
#import json

st.set_page_config(layout="wide")

application_train_clean_full = pd.read_csv('dataframe_st.csv')
del application_train_clean_full['Unnamed: 0']
#application_train_clean_full = application_train_clean_full.sample(n = 100, random_state = 42)
application_train_clean_full_nerf = application_train_clean_full.sample(n = 10, random_state = 42)
xgb_with_h = pickle.load(open('finalized_model.sav', 'rb'))    

label = ['20-30', '30-40', '40-50', '50-60', '60-70']

application_train_clean_full_api = application_train_clean_full_nerf.copy()
del application_train_clean_full_api['TARGET']
del application_train_clean_full_api['AGE']
del application_train_clean_full_api['LABEL_AGE']

application_train_clean_full_shap_global = application_train_clean_full.copy()
del application_train_clean_full_shap_global['TARGET']
del application_train_clean_full_shap_global['AGE']
del application_train_clean_full_shap_global['LABEL_AGE']

application_train_clean_full_shap_client = application_train_clean_full_nerf.copy()
application_train_clean_full_shap_client = application_train_clean_full_shap_client.assign(number=range(0,10))
del application_train_clean_full_shap_client['TARGET']
del application_train_clean_full_shap_client['AGE']
del application_train_clean_full_shap_client['LABEL_AGE']


#Menu déroulant
selectbox = st.sidebar.selectbox("Le point de vue à afficher : ",("Global", "Client", "Nouveau_client"))

if selectbox == "Global":
    
    st.title("Visualisation globale")
    
    st.subheader("Tableau")
    st.dataframe(application_train_clean_full_nerf)
    
    col1, col2= st.columns([7,1])
    with col1:
        st.subheader("Montant des crédits")
        fig2 = plt.figure(figsize = (20,2.07))
        sns.boxplot(application_train_clean_full['AMT_CREDIT'], color = 'red', showfliers=False, showmeans=True, meanprops={"marker": "+","markeredgecolor": "black","markersize": "10"})
        st.pyplot(fig2)
    with col2:
        st.subheader("Description")
        st.dataframe(application_train_clean_full['AMT_CREDIT'].describe())
   
    col3, col4 = st.columns([7,1])
    with col3:
        st.subheader("Remboursements annuels")
        fig3 = plt.figure(figsize = (20,2.19))
        sns.histplot(application_train_clean_full['AMT_ANNUITY'], color = 'green', kde='True')
        st.pyplot(fig3)
    with col4:
        st.subheader("Description") 
        st.dataframe(application_train_clean_full['AMT_ANNUITY'].describe())
    
    col5, col6 = st.columns([7,1])
    with col5:
        st.subheader("Prix des biens")
        fig4 = plt.figure(figsize = (20,1.95))
        sns.distplot(application_train_clean_full['AMT_GOODS_PRICE'], color = 'orange')
        st.pyplot(fig4)
    with col6:
        st.subheader("Description") 
        st.dataframe(application_train_clean_full['AMT_GOODS_PRICE'].describe())
        
    col7, col8 = st.columns([7,1])
    with col7:
        st.subheader("Revenu des clients")
        fig1 = plt.figure(figsize = (20,1.94))
        sns.distplot(application_train_clean_full['AMT_INCOME_TOTAL'], color = 'blue')
        plt.xticks([0, 2500000, 5000000, 7500000, 10000000, 12500000, 15000000, 17500000, 20000000])
        st.pyplot(fig1)
    with col8 : 
        st.subheader("Description")
        st.dataframe(application_train_clean_full['AMT_INCOME_TOTAL'].describe())    
    
    col9, col10 = st.columns([7,1])
    with col9:
        st.subheader("Age des clients")
        fig5 = plt.figure(figsize = (20,2.07))
        label_test = application_train_clean_full.groupby('LABEL_AGE').count().reset_index()
        plt.pie(label_test['AGE'], labels = label, autopct='%1.1f%%', textprops={'fontsize': 5})
        #fig5 = plt.gcf()
        #fig5.set_size_inches(4,4)
        #fig5 = px.pie(label_test, values = label, names = 'LABEL_AGE')
        #st.plotly_chart(fig5)
        st.pyplot(fig5) 
    with col10:
        st.subheader("Description")
        st.dataframe(application_train_clean_full['AGE'].describe())
    
    
    st.subheader("Données les plus importantes")
    #fig = plt.figure(figsize = (10,4))
    fig6, ax = plt.subplots(figsize=(20,5))
    #plt.suptitle('Feature importance', fontsize=30)
    #plt.xlabel('Features', fontsize=30)
    #plt.ylabel('F score', fontsize=30)
    #plt.tick_params(labelsize = 30)
    #plt.rcParams["font.size"] = 30   
    importance_type = st.selectbox('Select the desired importance type', ('weight','gain','cover'),index=0)
    plot_importance(xgb_with_h, grid=True, importance_type = importance_type, max_num_features=10, ax=ax)
    st.pyplot(fig6)
    
    explainer = shap.Explainer(xgb_with_h, application_train_clean_full_shap_global)
    shap_values = explainer(application_train_clean_full_shap_global) 
    fig11 = plt.figure(figsize = (20,2))
    shap.plots.beeswarm(shap_values)
    st.pyplot(fig11)
    

if selectbox == "Client":
    
    st.title("Visualisation client")
    
    client = st.selectbox("Le client à afficher :", application_train_clean_full_nerf['SK_ID_CURR'])
    ligne_client = application_train_clean_full_nerf.loc[application_train_clean_full_nerf['SK_ID_CURR']==client]
    st.subheader("Données client")
    st.dataframe(ligne_client)
    
    #st.subheader("Montant du crédit client")
    #fig7 = plt.figure(figsize = (20,2))
    #plt.axvline(x = ligne_client['AMT_CREDIT'].item(), color ="black", linestyle ="--") 
    #sns.boxplot(application_train_clean_full['AMT_CREDIT'], color = 'red', showfliers=False, showmeans=True, meanprops={"marker": "+","markeredgecolor": "black","markersize": "10"})
    #st.pyplot(fig7)

    #st.subheader("Remboursements annuels du client")
    #fig8 = plt.figure(figsize = (20,2))
    #plt.axvline(x = ligne_client['AMT_ANNUITY'].item(), color ="black", linestyle ="--") 
    #sns.histplot(application_train_clean_full['AMT_ANNUITY'], color = 'green', kde='True')
    #st.pyplot(fig8)
    
   # st.subheader("Prix du bien du client")
   # fig9 = plt.figure(figsize = (20,2))
    #plt.axvline(x = ligne_client['AMT_GOODS_PRICE'].item(), color ="black", linestyle ="--") 
    #sns.distplot(application_train_clean_full['AMT_GOODS_PRICE'], color = 'orange')
    #st.pyplot(fig9)
    
   # st.subheader("Age du client")
    #fig10 = plt.figure(figsize = (20,2))
   # label_test = application_train_clean_full.groupby('LABEL_AGE').count().reset_index()
    
   # if ligne_client['LABEL_AGE'].item() == '20-30':
    #    myexplode = [0.2,0,0,0,0]
   # if ligne_client['LABEL_AGE'].item() == '30-40':
   #     myexplode = [0,0.2,0,0,0]
   # if ligne_client['LABEL_AGE'].item() == '40-50':
   #     myexplode = [0,0,0.2,0,0]
   # if ligne_client['LABEL_AGE'].item() == '50-60':
   #     myexplode = [0,0,0,0.2,0]
   # if ligne_client['LABEL_AGE'].item() == '60-70':
   #     myexplode = [0,0,0,0,0.2]
    
   # plt.pie(label_test['AGE'], labels = label, autopct='%1.1f%%', explode = myexplode, textprops={'fontsize': 5})
    #st.pyplot(fig10)

    st.subheader("Acceptation du crédit")
    
    #appel à l'API
    
    ligne_client_api = application_train_clean_full_api.loc[application_train_clean_full_api['SK_ID_CURR']==client]
    url = 'http://localhost:5000/predict'
    r = requests.post(url,json=ligne_client_api.drop('SK_ID_CURR', axis=1).squeeze().to_json())
    res = r.json()
    res = res[0]
    if res[0]>0.5:
        #new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Prêt accordé</p>'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.markdown('**Prêt accordé**')
        st.write('Le taux d\'acceptation est de :', round(res[0]*100,3), '%') 
        
        if(st.button("Cliquez pour connaître les raisons de l'acceptation")):
            ligne_client_shap = application_train_clean_full_shap_client.loc[application_train_clean_full_shap_client['SK_ID_CURR']==client]
            num = ligne_client_shap['number']
            num = num.iloc[0]
            explainer = shap.Explainer(xgb_with_h, application_train_clean_full_shap_client)
            shap_values = explainer(application_train_clean_full_shap_client)
            fig12 = plt.figure(figsize = (10,2))
            shap.plots.waterfall(shap_values[num])
            st.pyplot(fig12)
        
    else:
        st.markdown('**Prêt refusé**')
        st.write('Le taux de refus est de :', round(res[1]*100,3), '%')
        
        #Shap
        
        if(st.button("Cliquez pour connaître les raisons du refus")):
            ligne_client_shap = application_train_clean_full_shap_client.loc[application_train_clean_full_shap_client['SK_ID_CURR']==client]
            num = ligne_client_shap['number']
            num = num.iloc[0]
            explainer = shap.Explainer(xgb_with_h, application_train_clean_full_shap_client)
            shap_values = explainer(application_train_clean_full_shap_client)
            fig12 = plt.figure(figsize = (10,2))
            shap.plots.waterfall(shap_values[num])
            st.pyplot(fig12)
   
    
    
if selectbox == "Nouveau_client":
 
    st.title("Outil de simulation rapide")
    new = []
    new = application_train_clean_full_api.median()
    
    age = st.number_input('Age du client')
    age = age*(-365.25)
    new['DAYS_BIRTH']=age
    
    cred = st.number_input('Montant du crédit')
    new['AMT_CREDIT']=cred
    
    prix = st.number_input('Prix du bien')
    new['AMT_GOODS_PRICE']=prix
    
    remb = st.number_input('Remboursement du crédit (par an)')
    new['AMT_ANNUITY']=remb
    
    rev = st.number_input('Revenus')
    rev = rev
    new['AMT_INCOME_TOTAL']=rev
    
    emploi = st.number_input('Depuis quand le client travail ? (en jours)')
    emploi = -emploi
    new['DAYS_EMPLOYED']=emploi
    
    ville = st.number_input('Le client habite-t-il dans la même ville qu\'il travaille ? (oui = 0, non = 1)')
    ville = ville
    new['LIVE_CITY_NOT_WORK_CITY']=ville
    
    #appel à l'API   
    
    url = 'http://localhost:5000/predict'
    r = requests.post(url,json=new.drop('SK_ID_CURR', axis=0).to_json())
    res = r.json()
    res = res[0]
    if res[0]>0.5:
        #new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Prêt accordé</p>'
        #st.markdown(new_title, unsafe_allow_html=True)
        st.markdown('**Prêt potentiellement accordé**')
        st.write('Le taux d\'acceptation est de :', round(res[0]*100,3), '%') 
    else:
        st.markdown('**Prêt potentiellement refusé**')
        st.write('Le taux de refus est de :', round(res[1]*100,3), '%')
    
#live city not work city/amt income total
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    