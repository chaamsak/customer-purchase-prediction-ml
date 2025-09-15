import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_input

st.title("🧠 Prédiction d'achat client (ML Express)")

# Chargement du modèle
model = joblib.load("modele_random_forest.joblib")

st.sidebar.header("Mode d'utilisation")
mode = st.sidebar.radio("Choisissez le mode :", ["Prédiction par fichier", "Saisie manuelle"])

if mode == "Prédiction par fichier":
    st.header("Prédiction à partir d'un fichier CSV")
    uploaded_file = st.file_uploader("Chargez votre fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu des données :", df.head())
        X = preprocess_input(df)
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        result = df.copy()
        result["Prediction"] = preds
        result["Probability"] = proba
        st.write("Résultats :", result)
        st.download_button("Télécharger les résultats", result.to_csv(index=False), file_name="predictions.csv")

else:
    st.header("Prédiction par saisie manuelle")
    # Champs de saisie pour chaque variable
    data = {}
    data["Administrative"] = st.number_input("Administrative (Nombre de pages consultées dans la section administrative du site)", min_value=0, value=2)
    data["Administrative_Duration"] = st.number_input("Administrative_Duration (Temps total (en secondes) passé sur les pages administratives)", min_value=0.0, value=60.0)
    data["Informational"] = st.number_input("Informational (Nombre de pages d'information (ex : FAQ, politique de retour) consultées)", min_value=0, value=1)
    data["Informational_Duration"] = st.number_input("Informational_Duration (Temps total passé sur les pages d'information)", min_value=0.0, value=30.0)
    data["ProductRelated"] = st.number_input("ProductRelated (Nombre de pages liées aux produits (détails, catalogues, etc.) consultées)", min_value=0, value=20)
    data["ProductRelated_Duration"] = st.number_input("ProductRelated_Duration (Temps total passé sur les pages produit)", min_value=0.0, value=300.0)
    data["BounceRates"] = st.number_input("BounceRates (Taux de rebond de la session (proportion de visites avec une seule interaction))", min_value=0.0, max_value=1.0, value=0.02)
    data["ExitRates"] = st.number_input("ExitRates (Taux de sortie de la page (proportion de sessions quittant le site à partir d'une page))", min_value=0.0, max_value=1.0, value=0.1)
    data["PageValues"] = st.number_input("PageValues (Valeur monétaire estimée des pages précédant une conversion)", min_value=0.0, value=5.0)
    data["SpecialDay"] = st.number_input("SpecialDay (Indice (0 à 1) indiquant la proximité d'un jour spécial (ex : Saint-Valentin))", min_value=0.0, max_value=1.0, value=0.4)
    data["Month"] = st.selectbox("Month (Mois de la session, ex : Jan, Feb, etc.)", ["Nov", "May", "Feb", "Mar", "June", "Jul", "Aug", "Sep", "Oct", "Dec"], index=0)
    data["OperatingSystems"] = st.number_input("OperatingSystems (Code représentant le système d'exploitation de l'utilisateur)", min_value=1, value=2)
    data["Browser"] = st.selectbox("Browser (Code représentant le navigateur utilisé)", [1, 2, 5, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13], index=0)
    data["Region"] = st.number_input("Region (Code géographique de la région d'où provient l'utilisateur)", min_value=1, value=1, max_value=9)
    data["TrafficType"] = st.selectbox("TrafficType (Source de trafic, ex : moteur de recherche, direct, publicité)", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], index=0)
    data["VisitorType"] = st.selectbox("VisitorType (Type de visiteur : 'Returning_Visitor', 'New_Visitor' ou 'Other')", ["Returning_Visitor", "New_Visitor", "Other"], index=0)
    data["Weekend"] = st.checkbox("Weekend (Booléen indiquant si la visite a eu lieu pendant un week-end (True ou False))", value=True)

    if st.button("Prédire"):
        df = pd.DataFrame([data])
        X = preprocess_input(df)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0, 1]
        label = "Achat" if pred else "Pas d'achat"
        st.success(f"Prédiction : {label} (probabilité : {proba:.2f})") 