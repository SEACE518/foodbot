import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import re
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Chatbot Restaurant Africain 🍲",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style personnalisé
st.markdown("""
    <style>
    body { background-color: #FFF8F0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stButton>button { background-color: #D2691E; color: white; border-radius: 12px; padding: 12px 24px; font-size: 18px; transition: background-color 0.3s ease; }
    .stButton>button:hover { background-color: #A0522D; }
    .stTextInput>div>div>input { border-radius: 12px; padding: 14px; font-size: 16px; border: 2px solid #D2691E; }
    .chat-bubble { background-color: #fff3e0; padding: 18px 24px; border-radius: 20px; margin-bottom: 15px; box-shadow: 0px 6px 12px rgba(210,105,30,0.3); max-width: 70%; }
    .user-bubble { background-color: #ffe0b2; text-align: right; margin-left: auto; }
    .suggestion-box { background-color: #fff3e0; border-left: 5px solid #D2691E; padding: 10px 15px; margin-bottom: 10px; border-radius: 8px; cursor: pointer; }
    .suggestion-box:hover { background-color: #f5deb3; }
    .sidebar .sidebar-content { background-color: #fbe9e7; }
    </style>
""", unsafe_allow_html=True)

# Liste des mots vides français adaptés au domaine restaurant africain
FRENCH_STOPWORDS = {
    'le', 'la', 'les', 'de', 'des', 'du', 'un', 'une', 'et', 'à', 'en', 'pour', 'avec', 'sur', 'au', 'aux',
    'ce', 'ces', 'dans', 'par', 'plus', 'ne', 'pas', 'que', 'qui', 'se', 'sa', 'son', 'ses', 'nous', 'vous',
    'ils', 'elles', 'être', 'avoir', 'faire', 'comme', 'mais', 'ou', 'donc', 'or', 'ni', 'car',
    # Mots spécifiques restauration africaine (exemples)
    'plat', 'plats', 'ingrédient', 'ingrédients', 'cuisson', 'réservation', 'menu',
    'commande', 'serveur', 'serveuse', 'client', 'clients', 'restaurant', 'table',
    'boisson', 'boissons', 'heure', 'heures', 'jour', 'jours', 'prix', 'tarif',
    'spécialité', 'spécialités', 'africain', 'africaine', 'afrique', 'nourriture',
    'repas', 'déjeuner', 'dîner', 'goûter', 'service', 'commande', 'livraison',
    'allergie', 'allergies', 'intolérance', 'intolérances', 'végétarien', 'végétarienne',
    'vegan', 'sans', 'gluten', 'piment', 'épicé', 'douceur',
    # Plats africains
    'foutou', 'fufu', 'attiéké', 'alloco', 'ceebu', 'jën', 'thiebou', 'dieune',
    'tô', 'soupe', 'egusi', 'kedjenou', 'ayimolou', 'poulet', 'yassa', 'mafé',
    'arachide', 'couscous', 'thiéboudiène', 'moambe', 'jollof', 'rice', 'bananes', 'plantains'
}

# Dictionnaire des descriptions des plats avec l'icône de chef cuistot
dish_descriptions = {
    'foutou': "👨‍🍳 Pâte de banane plantain, manioc ou igname, accompagnant les sauces.",
    'attiéké': "👨‍🍳 Semoule de manioc fermentée, servie avec poisson grillé.",
    'alloco': "👨‍🍳 Bananes plantains frites, croustillantes et moelleuses.",
    'ceebu jën': "👨‍🍳 Riz au poisson sénégalais, avec légumes et sauce tomate.",
    'kedjenou': "👨‍🍳 Ragoût de poulet ivoirien cuit à l'étouffée avec légumes et épices."
    # Ajoutez d'autres plats ici avec leurs descriptions
}

# Texte des connaissances intégré directement dans le code
knowledge_text = """
Foutou est une pâte traditionnelle africaine faite à base de manioc, d'igname ou de banane plantain. Elle accompagne souvent les sauces épicées.
Attiéké est un plat ivoirien préparé à partir de semoule de manioc fermentée. Il est souvent servi avec du poisson grillé et une salade fraîche.
Alloco désigne des bananes plantains frites, croustillantes à l'extérieur et moelleuses à l'intérieur. C'est un accompagnement populaire en Côte d'Ivoire.
Ceebu Jën, aussi appelé Thiéboudiène, est le plat national sénégalais composé de riz, poisson, légumes et une sauce tomate épicée.
Le Tô est une bouillie épaisse à base de farine de mil ou de maïs. Il est consommé avec des sauces variées en Afrique de l'Ouest.
La soupe Egusi est une soupe nigériane riche à base de graines de melon (egusi), souvent préparée avec des légumes et de la viande.
Le Kedjenou est un ragoût de poulet ivoirien cuit lentement avec des légumes et des épices, très parfumé.
L'Ayimolou est un plat togolais composé de riz et haricots, souvent accompagné de sauces locales.
Le Poulet Yassa est un plat sénégalais de poulet mariné au citron, oignons et moutarde, puis mijoté.
Le Mafé est un ragoût ou sauce à base de pâte d'arachide, populaire en Afrique de l'Ouest, souvent servi avec du riz.
Le Couscous africain est un plat à base de semoule de mil ou de maïs, accompagné de légumes et de viande.
Le Poulet Moambe est une spécialité congolaise où le poulet est cuisiné dans une sauce à base de pâte d'arachide et de moambe.
Le Jollof Rice est un plat ouest-africain de riz épicé cuit avec des tomates, oignons et piments.
Les bananes plantains frites (Alloco) sont souvent servies en accompagnement ou en snack.
Nos plats peuvent être adaptés pour les régimes végétariens ou sans gluten sur demande.
Nous proposons également un service de réservation de table et de livraison à domicile.
Pour toute allergie ou intolérance, merci de nous en informer avant la commande.
Nos horaires d'ouverture sont de 11h00 à 22h00 tous les jours.
Nous acceptons les paiements en espèces et par carte bancaire.
N'hésitez pas à demander des recommandations à notre personnel pour découvrir nos spécialités du jour.
"""

# Nettoyage du texte
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = [word for word in text.split() if word not in FRENCH_STOPWORDS]
    return ' '.join(words)

# Construction du corpus
@st.cache_data(show_spinner=False)
def build_corpus(text: str):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    processed = [preprocess(s) for s in sentences]
    vec = TfidfVectorizer()
    matrix = vec.fit_transform(processed)
    return sentences, matrix, vec

# Recherche de réponse
def get_response(sentences: list[str], matrix: np.ndarray, vec: TfidfVectorizer, query: str, threshold: float = 0.1) -> tuple[str, float]:
    q_processed = preprocess(query)
    q_vec = vec.transform([q_processed])
    sims = cosine_similarity(q_vec, matrix).flatten()
    idx = int(np.argmax(sims))
    score = sims[idx]
    if score < threshold:
        return ("Désolé, je n'ai pas compris votre demande. Pouvez-vous reformuler ou poser une autre question sur notre restaurant ou nos plats ?", score)
    return (sentences[idx], score)

# Suggestions de recherche en temps réel
def get_search_suggestions(query: str, sentences: list[str]) -> list[str]:
    suggestions = []
    for sentence in sentences:
        if query.lower() in sentence.lower():
            suggestions.append(sentence)
    return suggestions[:5]  # Limiter à 5 suggestions

# Fonction principale
def main():
    # Initialiser l'historique du feedback si ce n'est pas déjà fait
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1046/1046784.png", width=120)  # Image cuisine africaine
        st.title("Chatbot Restaurant Africain")
        st.write("---")
        st.write("### 📚 À propos")
        st.info("Cet assistant vous aide à découvrir notre menu, réserver une table, et répondre à vos questions sur nos spécialités africaines.")
        st.write("---")
        st.write(f"🗓️ **Date :** {datetime.now().strftime('%d/%m/%Y')}")

    st.title("🤖 Votre assistant culinaire africain")
    st.write("Posez vos questions sur nos plats, horaires, réservations, et plus encore !")

    # Avertissement
    st.warning("⚠️ Ce chatbot fournit des informations générales. Pour toute demande spécifique, contactez-nous directement.")

    # Construction du corpus à partir du texte intégré
    sentences, matrix, vec = build_corpus(knowledge_text)

    # Zone de saisie
    with st.form(key='question_form', clear_on_submit=True):
        question = st.text_input("💬 Posez votre question :", placeholder="Ex: Quels sont les ingrédients de l'attiéké ?")
        
        # Suggestions en temps réel
        if question:
            suggestions = get_search_suggestions(question, sentences)
            if suggestions:
                st.write("### Suggestions :")
                for suggestion in suggestions:
                    st.markdown(f"<div class='suggestion-box'>{suggestion}</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Envoyer 🍽️")
    
    if submitted and question:
        # Animation de chargement
        with st.spinner('Recherche de la meilleure réponse...'):
            time.sleep(1)  # pause naturelle
            response, score = get_response(sentences, matrix, vec, question)

        # Affichage stylé
        st.markdown(f"<div class='chat-bubble user-bubble'>{question}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'>{response}</div>", unsafe_allow_html=True)
        st.markdown(f"🔍 *Score de pertinence :* `{round(score, 2)}`")
        st.info("ℹ️ **Note** : cette réponse est informative uniquement.")

        # Recherche du plat dans la réponse et affichage de la description avec l'icône
        for dish, description in dish_descriptions.items():
            if dish in response.lower():
                st.write(description)
                break

if __name__ == "__main__":
    main()
