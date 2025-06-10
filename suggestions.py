import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import google.generativeai as genai
import plotly.express as px
import pandas as pd 

from gensim.models import Word2Vec
from db_utils import (
    get_saved_phq9_responses,
    get_extra_assessment_data,
    get_user_profile,
    save_user_suggestions,
    get_user_suggestions,
    delete_user_suggestions
)
from preprocessing import transform_single
from model import EnhancedBiLSTMClassifier

# --- Constants & Artifact Loading ---
MODEL_PATH = "best_depression_model.pt"
VOCAB_PATH = "vocabulary.pkl"
CONFIG_PATH = "preprocessing_config.json"
W2V_PATH = "word2vec_model.bin"

PHQ9_QUESTIONS = [
    "1. Anhedonia", "2. Depressed Mood", "3. Sleep Issues", "4. Fatigue",
    "5. Appetite Change", "6. Guilt/Worthlessness", "7. Concentration",
    "8. Psychomotor", "9. Suicidal Ideation"
]

# --- Model & AI Setup (Cached) ---
class DepressionClassifier:
    # ... (This class remains unchanged, keeping it for brevity)
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.load_artifacts()
        self.load_model()

    def load_artifacts(self):
        with open(VOCAB_PATH, 'rb') as f: self.vocab = pickle.load(f)
        with open(CONFIG_PATH, 'r') as f: self.config = json.load(f)
        try: w2v = Word2Vec.load(W2V_PATH)
        except Exception: w2v = None
        vocab_size, emb_dim = len(self.vocab), self.config['embedding_dim']
        self.embedding_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
        for token, idx in self.vocab.items():
            if w2v and token in w2v.wv: self.embedding_matrix[idx] = w2v.wv[token]
            elif token == '<UNK>': self.embedding_matrix[idx] = np.random.normal(scale=0.1, size=(emb_dim,))

    def load_model(self):
        self.model = EnhancedBiLSTMClassifier(emb_matrix=self.embedding_matrix, feature_dim=16, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.5, use_attention=True).to(self.device)
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device)['model_state_dict'])
            self.model.eval()
        else: st.error("❌ Model checkpoint not found!")

    def predict_proba(self, text: str) -> float:
        seq_idxs, feats = transform_single(text)
        if seq_idxs is None or len(seq_idxs) == 0: return 0.0
        x = torch.tensor(seq_idxs[None, :], dtype=torch.long, device=self.device)
        f = torch.tensor(feats[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(x, f)
            probs = F.softmax(logits, dim=1)
        return probs[0, 1].item()
        
@st.cache_resource(show_spinner="Loading AI model...")
def load_depression_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return DepressionClassifier(device=device)

@st.cache_resource
def setup_gemini():
    key = st.secrets.get("gemini", {}).get("GEMINI_API_KEY")
    if not key:
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-1.5-flash")

# --- Helper Functions ---
def classify_depression_level(phq9_score: int, model_prob: float):
    PHQ9_WEIGHT, MODEL_WEIGHT = 0.7, 0.3
    norm_phq9 = min(phq9_score / 27.0, 1.0)
    combined_score = PHQ9_WEIGHT * norm_phq9 + MODEL_WEIGHT * model_prob
    if combined_score < 0.25: level = "Minimal or No Depression"
    elif combined_score < 0.50: level = "Mild Depression"
    elif combined_score < 0.75: level = "Moderate Depression"
    else: level = "Severe Depression"
    return combined_score, level

def create_symptom_charts(phq9_scores):
    df = pd.DataFrame({'Symptom': PHQ9_QUESTIONS, 'Score': phq9_scores})
    symptom_clusters = {
        'Mood/Affective': ['Anhedonia', 'Depressed Mood', 'Guilt/Worthlessness', 'Suicidal Ideation'],
        'Somatic/Physical': ['Sleep Issues', 'Fatigue', 'Appetite Change'],
        'Cognitive': ['Concentration', 'Psychomotor']
    }
    df['Cluster'] = df['Symptom'].apply(lambda x: next((k for k, v in symptom_clusters.items() if x.split('. ')[1] in v), 'Other'))
    cluster_df = df.groupby('Cluster')['Score'].sum().reset_index()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Symptom Breakdown")
        fig1 = px.bar(df, x='Score', y='Symptom', orientation='h', title="PHQ-9 Scores by Question", color='Score', color_continuous_scale=px.colors.sequential.OrRd)
        fig1.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Severity Score", yaxis_title="")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("Symptom Clusters")
        fig2 = px.pie(cluster_df, values='Score', names='Cluster', title="Contribution by Symptom Cluster", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)

def create_text_summary(extra_data):
    if not extra_data or not extra_data.get("assessment_data"): return ""
    assessment = extra_data["assessment_data"]
    summary_points = []
    text_fields = {
        "mood_patterns_input": "Mood patterns", "life_satisfaction_input": "Joys/Challenges",
        "energy_fluctuation_input": "Energy patterns", "protective_factors_input": "Protective factors"
    }
    for key, label in text_fields.items():
        if assessment.get(key): summary_points.append(f"{label}: {assessment[key]}")
    if assessment.get("sleep_disturbances_multi"):
        summary_points.append(f"Sleep issues reported: {', '.join(assessment['sleep_disturbances_multi'])}")
    return ". ".join(summary_points)

# --- Main Page Function ---
def show_suggestions():
    st.title("🧩 Final Report & Suggestions")
    gemini = setup_gemini()
    model = load_depression_model()

    user_id = st.session_state.get("user_id")
    if not user_id:
        st.error("Please log in to view your report.")
        return

    # 1. CHECK FOR AND DISPLAY A SAVED REPORT
    saved_report = get_user_suggestions(user_id)
    if saved_report and "suggestions_data" in saved_report:
        report_data = saved_report["suggestions_data"]
        st.success("Found a previously generated report for you.")
        st.subheader("📊 Your Classification Results")
        c1, c2 = st.columns(2)
        c1.metric("PHQ-9 Score", f"{report_data.get('phq9_score', 'N/A')}")
        c2.metric("Text Analysis", f"{report_data.get('text_prob', 0):.0%}")
        st.progress(report_data.get('combined_score', 0), text=f"Final Assessment: {report_data.get('level', 'N/A')}")
        create_symptom_charts(report_data.get('phq9_scores', [0]*9))
        st.markdown("---")
        st.subheader("💡 Your Personalized Recommendations")
        st.markdown(report_data.get("suggestions", "No suggestions were saved."))
        if st.button("Regenerate My Report", type="primary"):
            delete_user_suggestions(user_id)
            st.rerun()
        return

    # 2. GENERATE A NEW REPORT IF NONE EXISTS
    st.info("Generating a new report based on your latest assessments...")
    phq9_scores = st.session_state.get("scores")
    extra_data = get_extra_assessment_data(user_id)
    profile = get_user_profile(st.session_state.username)

    if not phq9_scores or len(phq9_scores) != 9:
        st.error("PHQ-9 assessment is incomplete. Please complete it to generate a report.")
        return

    text_summary = create_text_summary(extra_data)
    model_prob = model.predict_proba(text_summary) if text_summary else 0.0
    phq9_score = sum(phq9_scores)
    combined_score, level = classify_depression_level(phq9_score, model_prob)

    st.subheader("📊 Your Classification Results")
    c1, c2 = st.columns(2)
    c1.metric("PHQ-9 Score", phq9_score)
    c2.metric("Text Analysis", f"{model_prob:.0%}" if text_summary else "N/A (No text provided)")
    st.progress(combined_score, text=f"Final Assessment: {level}")
    
    create_symptom_charts(phq9_scores)
    st.markdown("---")

    # 3. GENERATE SUGGESTIONS AUTOMATICALLY
    st.subheader("💡 Personalized Recommendations")
    suggestions_text = ""
    
    if gemini:
        prompt = (f"Analyze this mental health data for a {profile.get('age', 'N/A')}-year-old {profile.get('gender', 'N/A')}. "
                  f"PHQ-9 Score: {phq9_score}. Inferred Level: {level}. "
                  f"User's summary of concerns: '{text_summary}'. "
                  "As a compassionate AI assistant, provide a supportive, actionable report in Markdown. "
                  "1. **Validation**: Briefly validate their feelings. "
                  "2. **Immediate Self-Care**: 3 concrete, simple actions for today with a brief 'why'. "
                  "3. **Coping Strategies**: 3 longer-term habits with a 'why'. "
                  "4. **Professional Help**: Clear guidance on when to seek help and a next step. "
                  "Keep it concise, encouraging, and easy to read.")
        with st.spinner("🤖 Generating personalized suggestions..."):
            try:
                suggestions_text = gemini.generate_content(prompt).text
                st.markdown(suggestions_text)
            except Exception as e:
                st.error(f"AI generation failed: {e}. Showing default suggestions.")
    
    # Fallback to default suggestions if AI is unavailable or fails
    if not suggestions_text:
        if not gemini:
            st.warning("AI features are disabled. Showing evidence-based default suggestions.", icon="⚠️")
        suggestions_text = """
### Emotional Validation
It's completely understandable to feel the way you do. Your feelings are valid, and it takes courage to explore them. Remember to be kind to yourself through this process.

### Immediate Self-Care (Actions for Today)
- **Action**: Practice the 5-4-3-2-1 grounding technique (name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste).
  - **Why?**: It pulls your focus away from overwhelming thoughts and into the present moment, calming your nervous system.
- **Action**: Step outside for a 10-minute walk, preferably in sunlight.
  - **Why?**: Sunlight helps regulate your circadian rhythm and boosts Vitamin D, both linked to mood. Gentle movement releases endorphins.
- **Action**: Write down one small, achievable task for today and cross it off when done.
  - **Why?**: This provides a sense of accomplishment and control, countering feelings of helplessness.

### Sustainable Coping Strategies
- **Habit**: Schedule one 'no-cancel' social interaction per week (a call, coffee, or walk).
  - **Why?**: Meaningful social connection is a powerful antidepressant and combats the isolating effects of depression.
- **Habit**: Dedicate 15 minutes before bed to a 'wind-down' routine with no screens (e.g., reading a book, gentle stretching, listening to calm music).
  - **Why?**: Improves sleep quality, which is crucial for emotional regulation and cognitive function.
- **Habit**: Practice 'both/and' thinking. For example, "I feel hopeless, *and* I can still take a shower."
  - **Why?**: It challenges black-and-white thinking by acknowledging difficult feelings without letting them paralyze you.

### When to Seek Professional Help
- **Guidance**: If your PHQ-9 score is 10 or higher, or if these symptoms have persisted for more than two weeks and significantly impact your daily life, it is strongly recommended to consult a professional.
- **Next Step**: Schedule an appointment with your primary care doctor to discuss your symptoms, or use a directory like Psychology Today to find a licensed therapist.
"""
        st.markdown(suggestions_text)

    # 4. SAVE THE GENERATED REPORT
    report = {
        "phq9_score": phq9_score,
        "phq9_scores": phq9_scores,
        "text_prob": model_prob,
        "combined_score": combined_score,
        "level": level,
        "suggestions": suggestions_text,
    }
    if save_user_suggestions(user_id, report):
        st.success("Your report has been saved.")

    if st.button("🏠 Return to Home"):
        st.session_state["page"] = "Home"
        st.rerun()