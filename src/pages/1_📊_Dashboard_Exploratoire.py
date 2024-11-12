import streamlit as st
import pandas as pd
import os
import plotly.express as px
# Configuration de la page
st.set_page_config(page_title="Analyse du Churn Client", layout="wide")

# Fonction pour charger les données

def load_data():
    current_dir = os.getcwd()
    # Construire le chemin relatif vers le fichier
    file_path = os.path.join(current_dir, 'data/data.csv')
    data = pd.read_csv(file_path)
    return data

# Charger les données
data = load_data()


# Titre du Dashboard
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Dashboard d'Analyse du Churn Client </h1>", unsafe_allow_html=True)

# Calcul des métriques globales
total_clients = data.shape[0]
total_churn = data['Churn'].value_counts().get('Yes', 0)
churn_rate = (total_churn / total_clients) * 100
avg_tenure = data['tenure'].mean()
avg_monthly_charges = data['MonthlyCharges'].mean()
male_count = data[data['gender'] == 'Male'].shape[0]
female_count = data[data['gender'] == 'Female'].shape[0]

# Organisation des données en onglets
tab_readme, tab1, tab2, tab3 = st.tabs(["📘 Readme", "📈 Données Brutes et Statistiques", "📊 Visualisations Univariées", "📋 Analyses Complètes et Comparaison des Services"])

# Onglet Readme
with tab_readme:
    st.markdown("<h2 style='color: #007BFF;'>📚 Readme - Contexte, Objectif et Fonctionnement de l'Application</h2>", unsafe_allow_html=True)
    st.markdown("""
    ## 🔍 **Contexte**
    
    Le churn client (ou l'attrition client) est un enjeu critique pour de nombreuses entreprises, car il représente la perte de clients. Ce tableau de bord interactif permet d'analyser les raisons potentielles de cette perte en se concentrant sur les données d'une entreprise de télécommunications.

    ## 🎯 **Objectif et Utilité de l'Application**
    
    L'objectif de cette application est de fournir une analyse approfondie du churn client afin de mieux comprendre pourquoi les clients quittent l'entreprise. **En comprenant ces facteurs, les entreprises peuvent mettre en place des stratégies adaptées pour améliorer la satisfaction client et réduire le churn.**
    
    Ce tableau de bord interactif permet :
    - 📊 Une **visualisation des données** sous forme de graphiques interactifs.
    - 🔎 **Explorer les tendances** liées aux services, aux contrats, et aux données démographiques.
    - 📈 **Analyser les relations** entre différents services et le churn pour identifier les points d'amélioration.

    ## ⚙️ **Fonctionnement de l'Application**
    
    L'application est divisée en plusieurs onglets pour faciliter la navigation :
    
    - **📈 Données Brutes et Statistiques** : Permet de visualiser les données brutes et de voir les statistiques descriptives de base.
    - **📊 Visualisations Univariées** : Présente des analyses de distribution de différentes colonnes et leur relation avec le churn.
    - **📋 Analyses Complètes et Comparaison des Services** : Propose des analyses interactives comparant différents services et caractéristiques du client en fonction du churn.

    ## 🗂️ **Description des Colonnes du Dataset**
    
    - **customerID** : Identifiant unique du client.
    - **gender** : Genre du client (Male, Female).
    - **SeniorCitizen** : Indique si le client est un senior (1) ou non (0).
    - **Partner** : Indique si le client a un partenaire ou non (Yes, No).
    - **Dependents** : Indique si le client a des personnes à charge (Yes, No).
    - **tenure** : Nombre de mois pendant lesquels le client est resté avec l'entreprise.
    - **PhoneService** : Indique si le client a souscrit au service de téléphonie.
    - **MultipleLines** : Indique si le client a plusieurs lignes de téléphonie.
    - **InternetService** : Type de service internet (DSL, Fibre optique, Aucun).
    - **OnlineSecurity** : Indique si le client a souscrit à la sécurité en ligne (Yes, No, No internet service).
    - **OnlineBackup** : Indique si le client a souscrit à la sauvegarde en ligne (Yes, No, No internet service).
    - **DeviceProtection** : Indique si le client a une protection d'appareil (Yes, No, No internet service).
    - **TechSupport** : Indique si le client bénéficie d'une assistance technique (Yes, No, No internet service).
    - **StreamingTV** : Indique si le client a un service de streaming TV (Yes, No, No internet service).
    - **StreamingMovies** : Indique si le client a un service de streaming de films (Yes, No, No internet service).
    - **Contract** : Type de contrat du client (Mensuel, Un an, Deux ans).
    - **PaperlessBilling** : Indique si le client utilise la facturation sans papier (Yes, No).
    - **PaymentMethod** : Méthode de paiement utilisée par le client (Prélèvement automatique, Carte de crédit, etc.).
    - **MonthlyCharges** : Frais mensuels facturés au client.
    - **TotalCharges** : Frais totaux facturés au client.
    - **Churn** : Indique si le client a quitté l'entreprise (Yes, No).

    ## 🔗 **Utilisation du Tableau de Bord**
    
    Ce tableau de bord est conçu pour permettre aux utilisateurs d'explorer les données de manière interactive. Utilisez les onglets pour naviguer à travers les différentes sections et les graphiques pour explorer les relations entre les variables et le churn client.
    """, unsafe_allow_html=True)

# Onglet 1 : Données brutes et statistiques descriptives
with tab1:
    st.markdown("<h2 style='color: #4CAF50;'>📈 Données Brutes et Statistiques</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    # Affichage des métriques principales avec HTML/CSS pour mise en forme
    with col1:
        st.markdown("""
        <div style='background-color: #D5E8D4; padding: 10px; border-radius: 5px;'>
            <h3 style='color: #4CAF50;'>👥 Nombre Total de Clients</h3>
            <p style='font-size: 24px; font-weight: bold;'>""" + f"{total_clients}" + """</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color: #F8CECC; padding: 10px; border-radius: 5px;'>
            <h3 style='color: #FF5733;'>📉 Taux de Churn (%)</h3>
            <p style='font-size: 24px; font-weight: bold;'>""" + f"{churn_rate:.2f}%" + """</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background-color: #D4E0F0; padding: 10px; border-radius: 5px;'>
            <h3 style='color: #2E86C1;'>🕒 Durée Moyenne d'Engagement</h3>
            <p style='font-size: 24px; font-weight: bold;'>""" + f"{avg_tenure:.2f} mois" + """</p>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("""
        <div style='background-color: #FCF3CF; padding: 10px; border-radius: 5px;'>
            <h3 style='color: #D4AC0D;'>💰 Frais Mensuels Moyens (€)</h3>
            <p style='font-size: 24px; font-weight: bold;'>""" + f"{avg_monthly_charges:.2f}" + """</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div style='background-color: #D5E8D4; padding: 10px; border-radius: 5px;'>
            <h3 style='color: #16A085;'>♂️ Nombre d'Hommes</h3>
            <p style='font-size: 24px; font-weight: bold;'>""" + f"{male_count}" + """</p>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div style='background-color: #F5B7B1; padding: 10px; border-radius: 5px;'>
            <h3 style='color: #C0392B;'>♀️ Nombre de Femmes</h3>
            <p style='font-size: 24px; font-weight: bold;'>""" + f"{female_count}" + """</p>
        </div>
        """, unsafe_allow_html=True)

    # Affichage des données et statistiques descriptives
    col7, col8 = st.columns(2)
    with col7:
        if st.checkbox('Afficher les données brutes'):
            with st.expander("📊 Aperçu des Données Brutes", expanded=False):
                st.dataframe(data, use_container_width=True)

    with col8:
        if st.checkbox('Afficher les statistiques descriptives'):
            with st.expander("📈 Statistiques descriptives", expanded=False):
                st.dataframe(data.describe(), use_container_width=True)

# Onglet 2 : Visualisations Univariées
with tab2:
    st.markdown("<h2 style='color: #FF5733;'>📊 Visualisations Univariées et Comparaison des Services</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Sélection d'une colonne à visualiser
    with col1:
        with st.container(border=True):
            col = st.selectbox('Choisissez une colonne à visualiser', data.drop(columns="customerID").columns)
            if col:
                
                if data[col].dtype == 'object':
                    fig = px.histogram(data, x=col, color='Churn', barmode='group', title=f"Distribution de {col}")
                else:
                    fig = px.histogram(data, x=col, marginal="box", nbins=30, title=f"Distribution de {col}")
                st.plotly_chart(fig, use_container_width=True)

    # Analyse démographique
    with col2:
        st.subheader("📊 Analyse démographique")
        demographic_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
        with st.container(border=True):
            selected_demo = st.selectbox("Sélectionnez une variable démographique", demographic_cols)
            if selected_demo:
                fig_demo = px.histogram(data, x=selected_demo, color='Churn', barmode='group', title=f"Analyse de {selected_demo} par rapport au Churn")
                st.plotly_chart(fig_demo, use_container_width=True)

    # Comparaison des services
    st.subheader("📊 Comparaison des Services par rapport au Churn")
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    with st.container(border=True):
        selected_service = st.selectbox("Sélectionnez un service à analyser", service_cols)
        if selected_service:
            fig_service = px.histogram(data, x=selected_service, color='Churn', barmode='group', title=f"Analyse de {selected_service} par rapport au Churn")
            st.plotly_chart(fig_service, use_container_width=True)

# Onglet 3 : Analyses Complètes et Comparaison des Services
with tab3:
    st.markdown("<h2 style='color: #8E44AD;'>📋 Analyses Complètes et Comparaison des Services</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    
        # Analyse des contrats
    with col1:
        with st.container(border=True):
            st.subheader("📋 Analyse des Contrats par rapport au Churn")
            fig_contract = px.histogram(data, x='Contract', color='Churn', barmode='group', title="Répartition des Contrats par Churn")
            st.plotly_chart(fig_contract, use_container_width=True)
    
    # Analyse des méthodes de paiement
    with col2:
        with st.container(border=True):
            st.subheader("📋 Analyse des Méthodes de Paiement par rapport au Churn")
            fig_payment = px.histogram(data, x='PaymentMethod', color='Churn', barmode='group', title="Répartition des Méthodes de Paiement par Churn")
            st.plotly_chart(fig_payment, use_container_width=True)
    with st.container(border=True):
        # Comparaison des frais mensuels
        st.subheader("📋 Comparaison des Frais Mensuels par Churn")
        fig_monthly_charges = px.box(data, x='Churn', y='MonthlyCharges', color='Churn', title="Comparaison des Frais Mensuels par Churn")
        st.plotly_chart(fig_monthly_charges,use_container_width=True)
    with st.container(border=True):
        # Relation entre la durée d'engagement et le churn
        st.subheader("📋 Durée d'Engagement et Churn")
        fig_tenure_churn = px.histogram(data, x='tenure', color='Churn', nbins=30, title="Durée d'Engagement par Churn", marginal='box')
        st.plotly_chart(fig_tenure_churn, use_container_width=True)

# Filtrage interactif
st.sidebar.header("🔍 Filtres")
tenure = st.sidebar.slider("Durée d'engagement (mois)", min_value=int(data['tenure'].min()), max_value=int(data['tenure'].max()), value=(int(data['tenure'].min()), int(data['tenure'].max())))
filtered_data = data[(data['tenure'] >= tenure[0]) & (data['tenure'] <= tenure[1])]
st.sidebar.write("🔍 **Données filtrées selon la durée d'engagement :**")
with st.sidebar.expander("📊 Aperçu des Données Filtrées", False):
    st.write(filtered_data)

# Footer avec emojis et style
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        color: #4CAF50;
        font-size: 16px;
        margin-top: 20px;
    }
    .footer a {
        color: #4CAF50;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <p class='footer'>💡 Dashboard créé avec ❤️ par <a href='https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning/' target='_blank'>Abraham KOLOBOE</a></p>
    """,
    unsafe_allow_html=True
)