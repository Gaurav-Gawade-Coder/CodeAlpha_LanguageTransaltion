from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import streamlit as st
import string

# Download necessary NLTK data (runs once)
@st.cache_resource
def download_nltk_data():
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('punkt_tab')


download_nltk_data()

# Page Configuration
st.set_page_config(
    page_title="CodeAlpha: FAQ Chatbot", page_icon="🤖", layout="centered"
)

st.title("🤖 AI FAQ Chatbot")
st.write(
    "Match user questions to FAQs using NLP and Cosine Similarity."
)

# Sample FAQ Knowledge Base
faqs = {
    "What is CodeAlpha?": (
        "CodeAlpha is a software development company that offers educational"
        " tech internships and hands-on project experience in emerging"
        " technologies."
    ),
    "What perks are included in the internship?": (
        "Perks include an internship offer letter, QR-verified completion"
        " certificate, unique ID certificate, letter of recommendation based on"
        " performance, placement support, and resume-building support."
    ),
    "How many tasks are required to complete the internship?": (
        "You must complete a minimum of 2 or 3 out of the listed tasks to be"
        " eligible for the completion certificate."
    ),
    "How do I submit my completed project?": (
        "You need to upload your source code to GitHub, record a video"
        " explanation and post it on LinkedIn tagging @CodeAlpha, and then"
        " submit the details via the official submission form shared in your"
        " WhatsApp group."
    ),
    "What libraries are used for this FAQ chatbot?": (
        "This chatbot uses NLTK for text preprocessing and scikit-learn"
        " (TfidfVectorizer & cosine_similarity) to find the best matching"
        " question."
    ),
}

questions = list(faqs.keys())
answers = list(faqs.values())


# Text Preprocessing Function using NLTK
def preprocess_text(text):
  lemmatizer = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))

  # Tokenize and lowercase
  tokens = word_tokenize(text.lower())

  # Remove punctuation and stopwords, then lemmatize
  cleaned_tokens = [
      lemmatizer.lemmatize(token)
      for token in tokens
      if token not in string.punctuation and token not in stop_words
  ]
  return ' '.join(cleaned_tokens)


# Preprocess all FAQ questions for matching
processed_questions = [preprocess_text(q) for q in questions]

# Vectorize questions using TF-IDF
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

# Initialize chat history in session state if not present
if 'messages' not in st.session_state:
  st.session_state.messages = [{
      'role': 'assistant',
      'content': (
          'Hello! I am your CodeAlpha FAQ assistant. Ask me anything about the'
          ' internship program!'
      ),
  }]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
  with st.chat_message(message['role']):
    st.markdown(message['content'])

# Accept user input via chat input box
if user_query := st.chat_input('Type your question here...'):
  # Add user message to chat history
  st.session_state.messages.append({'role': 'user', 'content': user_query})
  with st.chat_message('user'):
    st.markdown(user_query)

  # Process user query and find best match using Cosine Similarity
  processed_query = preprocess_text(user_query)
  query_vector = vectorizer.transform([processed_query])

  similarity_scores = cosine_similarity(query_vector, question_vectors)
  best_match_idx = similarity_scores.argmax()
  highest_score = similarity_scores[0, best_match_idx]

  # Set a confidence threshold for matching
  if highest_score > 0.15:
    bot_response = answers[best_match_idx]
  else:
    bot_response = (
        "I'm sorry, I couldn't find a matching answer in my FAQ database. Please"
        ' try asking something else related to the internship!'
    )

  # Add assistant response to chat history
  st.session_state.messages.append(
      {'role': 'assistant', 'content': bot_response}
  )
  with st.chat_message('assistant'):
    st.markdown(bot_response)
