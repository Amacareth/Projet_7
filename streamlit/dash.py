import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st
import pickle
import requests
import streamlit.components.v1 as components
    

st.set_page_config(page_title='Prêt à dépenser', page_icon='💻',layout="wide")

application_train_clean_full = pd.read_csv('dataframe_st.csv')
del application_train_clean_full['Unnamed: 0']
application_train_clean_full_nerf = application_train_clean_full.sample(n = 20, random_state = 42)
xgb_with_h = pickle.load(open('finalized_model.sav', 'rb'))    

label = ['20-30', '30-40', '40-50', '50-60', '60-70']

application_train_clean_full_api = application_train_clean_full_nerf.copy()
del application_train_clean_full_api['TARGET']
del application_train_clean_full_api['AGE']
del application_train_clean_full_api['LABEL_AGE']

application_train_clean_full_shap_global = application_train_clean_full.copy()
application_train_clean_full_shap_global['DAYS_EMPLOYED'] = -application_train_clean_full_shap_global['DAYS_EMPLOYED']
del application_train_clean_full_shap_global['TARGET']
del application_train_clean_full_shap_global['AGE']
del application_train_clean_full_shap_global['LABEL_AGE']

application_train_clean_full_shap_client = application_train_clean_full_nerf.copy()
application_train_clean_full_shap_client = application_train_clean_full_shap_client.assign(number=range(0,20))
del application_train_clean_full_shap_client['TARGET']
del application_train_clean_full_shap_client['AGE']
del application_train_clean_full_shap_client['LABEL_AGE']



shap.initjs()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)







#Menu déroulant
selectbox = st.sidebar.selectbox("Quelle partie choisissez-vous ? ",("Visualisation", "Crédit", "Simulation"))





if selectbox == "Visualisation":

    
    st.title('Visualisation globale et client')
    choice = st.radio('Afficher positionnement client ?', ('Non', 'Oui'))
    if choice == 'Non':
        st.subheader("Tableau global")
        st.dataframe(application_train_clean_full_nerf)
    else :
        client = st.selectbox("Le client à afficher :", application_train_clean_full_nerf['SK_ID_CURR'])
        ligne_client = application_train_clean_full_nerf.loc[application_train_clean_full_nerf['SK_ID_CURR']==client]
        ligne_client['DAYS_EMPLOYED'] = -ligne_client['DAYS_EMPLOYED']
        st.subheader("Données client")
        st.dataframe(ligne_client)
    feature = st.multiselect("Quelles données affichées ?", ['AMT_CREDIT', 'AMT_ANNUITY','AMT_GOODS_PRICE','AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'LIVE_CITY_NOT_WORK_CITY', 'CODE_GENDER_F', 'AGE'], help='emploi : en mois, city : 0 = adresse identique ; genre : 0 = homme')
    if len(feature)==0:
        st.subheader('Veuillez choisir une donnée à afficher')
    if len(feature)==1:   
        if choice == 'Non':
           for i in feature :
               if i == 'AGE':
                   fig = plt.figure(figsize = (20,2.19))
                   label_test = application_train_clean_full.groupby('LABEL_AGE').count().reset_index()
                   plt.pie(label_test[i], labels = label, autopct='%1.1f%%', textprops={'fontsize': 5})
                   st.pyplot(fig)
               elif i == 'LIVE_CITY_NOT_WORK_CITY':
                   fig = plt.figure(figsize = (10,1.5))
                   sns.countplot(application_train_clean_full_shap_global[i], color = 'navy')
                   st.pyplot(fig)
               elif i == 'CODE_GENDER_F':
                   fig = plt.figure(figsize = (10,1.5))
                   sns.countplot(application_train_clean_full_shap_global[i], color = 'maroon')
                   st.pyplot(fig)
               else :
                   graph_option = st.selectbox("Quel type de graphique ?", ['','histplot', 'boxplot', 'stripplot', 'violinplot', 'kdeplot'])
                   if graph_option == 'histplot' :
                       fig = plt.figure(figsize = (10,1.5))
                       sns.histplot(application_train_clean_full_shap_global[i], color = 'limegreen')
                       st.pyplot(fig)
                   if graph_option == 'boxplot' :
                       fig = plt.figure(figsize = (10,1.5))
                       sns.boxplot(application_train_clean_full_shap_global[i], color = 'gold', showfliers=False, showmeans=True, meanprops={"marker": "o", "markerfacecolor" : "white","markeredgecolor": "black","markersize": "15"})
                       st.pyplot(fig)
                   if graph_option == 'stripplot' :
                       fig = plt.figure(figsize = (10,1.5))
                       sns.stripplot(application_train_clean_full_shap_global[i], color = 'crimson', linewidth = 0.5)
                       st.pyplot(fig)
                   if graph_option == 'violinplot' :
                       fig = plt.figure(figsize = (10,1.5))
                       sns.violinplot(application_train_clean_full_shap_global[i], color = 'cyan')
                       st.pyplot(fig)
                   if graph_option == 'kdeplot' :
                       fig = plt.figure(figsize = (10,1.5))
                       sns.kdeplot(application_train_clean_full_shap_global[i], color = 'magenta', bw_adjust=0.5, fill=True)
                       st.pyplot(fig)
           feat_imp = st.checkbox('Afficher les données importantes') 
           if feat_imp: 
               explainer = shap.Explainer(xgb_with_h, application_train_clean_full_shap_global)
               shap_values = explainer(application_train_clean_full_shap_global) 
               fig = plt.figure(figsize = (20,2))
               shap.plots.beeswarm(shap_values)
               st.pyplot(fig)
        if choice == 'Oui':
           for i in feature :
               if i == 'AGE':
                   fig = plt.figure(figsize = (20,2.19))
                   label_test = application_train_clean_full.groupby('LABEL_AGE').count().reset_index()
                   if ligne_client['LABEL_AGE'].item() == '20-30':
                       myexplode = [0.2,0,0,0,0]
                   if ligne_client['LABEL_AGE'].item() == '30-40':
                       myexplode = [0,0.2,0,0,0]
                   if ligne_client['LABEL_AGE'].item() == '40-50':
                       myexplode = [0,0,0.2,0,0]
                   if ligne_client['LABEL_AGE'].item() == '50-60':
                       myexplode = [0,0,0,0.2,0]
                   if ligne_client['LABEL_AGE'].item() == '60-70':
                       myexplode = [0,0,0,0,0.2]
                   plt.pie(label_test[i], labels = label, autopct='%1.1f%%', explode = myexplode, textprops={'fontsize': 5})
                   st.pyplot(fig)
               elif i == 'LIVE_CITY_NOT_WORK_CITY':
                   fig = plt.figure(figsize = (10,1.5))
                   plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                   sns.countplot(application_train_clean_full_shap_global[i], color = 'navy')
                   st.pyplot(fig)
               elif i == 'CODE_GENDER_F':
                   fig = plt.figure(figsize = (10,1.5))
                   plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                   sns.countplot(application_train_clean_full_shap_global[i], color = 'maroon')
                   st.pyplot(fig)
               else :
                   graph_option = st.selectbox("Quel type de graphique ?", ['','histplot', 'boxplot', 'stripplot', 'violinplot', 'kdeplot'])
                   if graph_option == 'histplot' :
                       fig = plt.figure(figsize = (10,1.5))
                       plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                       sns.histplot(application_train_clean_full_shap_global[i], color = 'limegreen')
                       st.pyplot(fig)
                   if graph_option == 'boxplot' :
                       fig = plt.figure(figsize = (10,1.5))
                       plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                       sns.boxplot(application_train_clean_full_shap_global[i], color = 'gold', showfliers=False, showmeans=True, meanprops={"marker": "o", "markerfacecolor" : "white","markeredgecolor": "black","markersize": "15"})
                       st.pyplot(fig)
                   if graph_option == 'stripplot' :
                       fig = plt.figure(figsize = (10,1.5))
                       plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                       sns.stripplot(application_train_clean_full_shap_global[i], color = 'crimson', linewidth = 0.5)
                       st.pyplot(fig)
                   if graph_option == 'violinplot' :
                       fig = plt.figure(figsize = (10,1.5))
                       plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                       sns.violinplot(application_train_clean_full_shap_global[i], color = 'cyan')
                       st.pyplot(fig)
                   if graph_option == 'kdeplot' :
                       fig = plt.figure(figsize = (10,1.5))
                       plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                       sns.kdeplot(application_train_clean_full_shap_global[i], color = 'magenta', bw_adjust=0.5, fill=True)
                       st.pyplot(fig)
    if len(feature)>=2: 
        feature0 = feature[0]
        feature1 = feature[1]
        graph_option = st.selectbox("Quel type de graphique ?", ['','histplot', 'boxplot', 'stripplot', 'violinplot', 'kdeplot'])
        if choice == 'Non':
            for i in feature :
                if i == 'AGE':
                    fig = plt.figure(figsize = (20,2.19))
                    label_test = application_train_clean_full.groupby('LABEL_AGE').count().reset_index()
                    plt.pie(label_test[i], labels = label, autopct='%1.1f%%', textprops={'fontsize': 5})
                    st.pyplot(fig)
                elif i == 'LIVE_CITY_NOT_WORK_CITY':
                    fig = plt.figure(figsize = (10,1.5))
                    sns.countplot(application_train_clean_full_shap_global[i], color = 'navy')
                    st.pyplot(fig)
                elif i == 'CODE_GENDER_F':
                    fig = plt.figure(figsize = (10,1.5))
                    sns.countplot(application_train_clean_full_shap_global[i], color = 'maroon')
                    st.pyplot(fig)
                else :
                    if graph_option == 'histplot' :
                        fig = plt.figure(figsize = (10,1.5))
                        sns.histplot(application_train_clean_full_shap_global[i], color = 'limegreen')
                        st.pyplot(fig)
                    if graph_option == 'boxplot' :
                        fig = plt.figure(figsize = (10,1.5))
                        sns.boxplot(application_train_clean_full_shap_global[i], color = 'gold', showfliers=False, showmeans=True, meanprops={"marker": "o", "markerfacecolor" : "white","markeredgecolor": "black","markersize": "15"})
                        st.pyplot(fig)
                    if graph_option == 'stripplot' :
                        fig = plt.figure(figsize = (10,1.5))
                        sns.stripplot(application_train_clean_full_shap_global[i], color = 'crimson', linewidth = 0.5)
                        st.pyplot(fig)
                    if graph_option == 'violinplot' :
                        fig = plt.figure(figsize = (10,1.5))
                        sns.violinplot(application_train_clean_full_shap_global[i], color = 'cyan')
                        st.pyplot(fig)
                    if graph_option == 'kdeplot' :
                        fig = plt.figure(figsize = (10,1.5))
                        sns.kdeplot(application_train_clean_full_shap_global[i], color = 'magenta', bw_adjust=0.5, fill=True)
                        st.pyplot(fig)
            biv_analiz=st.checkbox('Afficher l\'analyse bivariée entre les données', help='Seulement les deux premières données sélectionnées seront analysées de façon bivariée')
            if biv_analiz:
                fig30 = sns.jointplot(x=feature0, y = feature1, data = application_train_clean_full_shap_global, color='darkturquoise')
                st.pyplot(fig30)
            feat_imp = st.checkbox('Afficher les données importantes') 
            if feat_imp: 
                explainer = shap.Explainer(xgb_with_h, application_train_clean_full_shap_global)
                shap_values = explainer(application_train_clean_full_shap_global) 
                fig = plt.figure(figsize = (20,2))
                shap.plots.beeswarm(shap_values)
                st.pyplot(fig)
        if choice == 'Oui':
            for i in feature :
                if i == 'AGE':
                    fig = plt.figure(figsize = (20,2.19))
                    label_test = application_train_clean_full.groupby('LABEL_AGE').count().reset_index()
                    if ligne_client['LABEL_AGE'].item() == '20-30':
                        myexplode = [0.2,0,0,0,0]
                    if ligne_client['LABEL_AGE'].item() == '30-40':
                        myexplode = [0,0.2,0,0,0]
                    if ligne_client['LABEL_AGE'].item() == '40-50':
                        myexplode = [0,0,0.2,0,0]
                    if ligne_client['LABEL_AGE'].item() == '50-60':
                        myexplode = [0,0,0,0.2,0]
                    if ligne_client['LABEL_AGE'].item() == '60-70':
                        myexplode = [0,0,0,0,0.2]
                    plt.pie(label_test[i], labels = label, autopct='%1.1f%%', explode = myexplode, textprops={'fontsize': 5})
                    st.pyplot(fig)
                elif i == 'LIVE_CITY_NOT_WORK_CITY':
                    fig = plt.figure(figsize = (10,1.5))
                    plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                    sns.countplot(application_train_clean_full_shap_global[i], color = 'navy')
                    st.pyplot(fig)
                elif i == 'CODE_GENDER_F':
                    fig = plt.figure(figsize = (10,1.5))
                    plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                    sns.countplot(application_train_clean_full_shap_global[i], color = 'maroon')
                    st.pyplot(fig)
                else :
                    if graph_option == 'histplot' :
                        fig = plt.figure(figsize = (10,1.5))
                        plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                        sns.histplot(application_train_clean_full_shap_global[i], color = 'limegreen')
                        st.pyplot(fig)
                    if graph_option == 'boxplot' :
                        fig = plt.figure(figsize = (10,1.5))
                        plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                        sns.boxplot(application_train_clean_full_shap_global[i], color = 'gold', showfliers=False, showmeans=True, meanprops={"marker": "o", "markerfacecolor" : "white","markeredgecolor": "black","markersize": "15"})
                        st.pyplot(fig)
                    if graph_option == 'stripplot' :
                        fig = plt.figure(figsize = (10,1.5))
                        plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                        sns.stripplot(application_train_clean_full_shap_global[i], color = 'crimson', linewidth = 0.5)
                        st.pyplot(fig)
                    if graph_option == 'violinplot' :
                        fig = plt.figure(figsize = (10,1.5))
                        plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                        sns.violinplot(application_train_clean_full_shap_global[i], color = 'cyan')
                        st.pyplot(fig)
                    if graph_option == 'kdeplot' :
                        fig = plt.figure(figsize = (10,1.5))
                        plt.axvline(x = ligne_client[i].item(), color ="black", linestyle ="--")
                        sns.kdeplot(application_train_clean_full_shap_global[i], color = 'magenta', bw_adjust=0.5, fill=True)
                        st.pyplot(fig)
            biv_analiz=st.checkbox('Voir l\'analyse bivariée entre les données', help='Seulement les deux premières données sélectionnées seront analysées de façon bivariée')
            if biv_analiz:
                fig30 = sns.jointplot(x=feature0, y = feature1, data = application_train_clean_full_shap_global, color='darkturquoise')
                fig30.ax_joint.axvline(x=ligne_client[feature0].item(),color ="red", linestyle ="--")
                fig30.ax_joint.axhline(y=ligne_client[feature1].item(),color ="red", linestyle ="--")
                st.pyplot(fig30)
            
            
    
    


if selectbox == "Crédit":
    
    st.title("Crédit client")
    
    client = st.selectbox("Le client à afficher :", application_train_clean_full_nerf['SK_ID_CURR'])
    ligne_client = application_train_clean_full_nerf.loc[application_train_clean_full_nerf['SK_ID_CURR']==client]
    st.subheader("Données client")
    st.dataframe(ligne_client)
    
    #appel à l'API
    
    ligne_client_api = application_train_clean_full_api.loc[application_train_clean_full_api['SK_ID_CURR']==client]
    url = 'http://localhost:5000/predict'
    r = requests.post(url,json=ligne_client_api.drop('SK_ID_CURR', axis=1).squeeze().to_json())
    res = r.json()
    res = res[0]
    if res[0]>0.5:
        new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Prêt accordé</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write('Le taux d\'acceptation est de :', round(res[0]*100,3), '%') 
        
        ligne_client_shap = application_train_clean_full_shap_client.loc[application_train_clean_full_shap_client['SK_ID_CURR']==client]
        num = ligne_client_shap['number']
        num = num.iloc[0]
        explainer = shap.Explainer(xgb_with_h, application_train_clean_full_shap_client)
        shap_values = explainer(application_train_clean_full_shap_client)
        
        if(st.button("Cliquez pour connaître les raisons de l'acceptation")):
            st_shap(shap.force_plot(explainer.expected_value, shap_values.values[num,:], application_train_clean_full_shap_client.iloc[num,:]))
            fig12 = plt.figure(figsize = (10,2))
            shap.plots.waterfall(shap_values[num])
            st.pyplot(fig12)
        
    else:
        new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Prêt refusé</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write('Le taux de refus est de :', round(res[1]*100,3), '%')
        
        ligne_client_shap = application_train_clean_full_shap_client.loc[application_train_clean_full_shap_client['SK_ID_CURR']==client]
        num = ligne_client_shap['number']
        num = num.iloc[0]
        explainer = shap.Explainer(xgb_with_h, application_train_clean_full_shap_client)
        shap_values = explainer(application_train_clean_full_shap_client)
        
        #Shap
    
        if(st.button("Cliquez pour connaître les raisons du refus")):
            st_shap(shap.force_plot(explainer.expected_value, shap_values.values[num,:], application_train_clean_full_shap_client.iloc[num,:]))
            fig12 = plt.figure(figsize = (10,2))
            shap.plots.waterfall(shap_values[num])
            st.pyplot(fig12)
   
    
   
    
   
    
if selectbox == "Simulation":
 
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
    
    emploi = st.number_input('Depuis quand le client travail ? (en mois)')
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
        new_title = '<p style="font-family:sans-serif; color:Green; font-size: 32px;">Prêt potentiellement accordé</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write('Le taux d\'acceptation est de :', round(res[0]*100,3), '%') 
    else:
        new_title = '<p style="font-family:sans serif; color:Red; font-size: 32px;">Prêt potentiellement refusé</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write('Le taux de refus est de :', round(res[1]*100,3), '%')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    