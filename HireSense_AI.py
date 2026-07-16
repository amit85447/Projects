import os
import re
import fitz  # PyMuPDF
import docx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import mysql.connector
from mysql.connector import Error
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import io

# ==========================================
# 1. STREAMLIT CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="HireSense AI", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #10B981;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #9CA3AF;
    }
    .metric-card {
        background-color: #1F2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #9CA3AF;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #3B82F6;
        font-size: 24px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 10px 24px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    .dataframe {
        background-color: #1F2937;
        color: #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

# ==========================================
# 2. DATABASE MANAGEMENT SYSTEM (MySQL)
# ==========================================
class DatabaseManager:
    def __init__(self):
        self.host = "localhost"
        self.user = "root"
        self.password = "Amit@123" 
        self.database = "hiresense_ai_db"
        self.init_database()

    def get_connection(self):
        return mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def init_database(self):
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            conn.close()

            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password VARCHAR(100) NOT NULL,
                    role VARCHAR(50) DEFAULT 'admin'
                )
            """)
            
            cursor.execute("SELECT * FROM users WHERE username = 'admin'")
            if not cursor.fetchone():
                cursor.execute("INSERT INTO users (username, password, role) VALUES ('admin', 'admin123', 'admin')")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    phone VARCHAR(50),
                    skills TEXT,
                    education TEXT,
                    experience TEXT,
                    certifications TEXT,
                    projects TEXT,
                    resume_score INT,
                    similarity_score FLOAT,
                    selection_probability FLOAT,
                    status VARCHAR(50) DEFAULT 'Pending'
                )
            """)
            conn.commit()
            conn.close()
        except Error as e:
            st.error(f"Database Initialization Error: {e}. Please ensure MySQL is running.")

    def execute_query(self, query, params=None):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
            conn.close()
            return True
        except Error as e:
            st.error(f"Database Execution Error: {e}")
            return False

    def fetch_all(self, query, params=None):
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            result = cursor.fetchall()
            conn.close()
            return result
        except Error as e:
            return []

db = DatabaseManager()

# ==========================================
# 3. NLP & RESUME PARSING ENGINE
# ==========================================
class ResumeParserEngine:
    @staticmethod
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s,.:@\-\(\)]', '', text)
        return text.strip()

    def extract_text(self, uploaded_file):
        text = ""
        extension = uploaded_file.name.split('.')[-1].lower()
        try:
            if extension == 'pdf':
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page in doc:
                    text += page.get_text()
            elif extension == 'docx':
                doc = docx.Document(io.BytesIO(uploaded_file.read()))
                text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error parsing file: {e}")
        return self.clean_text(text)

    def extract_features(self, text):
        skill_bank = [
            "python", "java", "c++", "javascript", "react", "angular", "vue", "node.js", 
            "sql", "mysql", "postgresql", "nosql", "mongodb", "aws", "azure", "gcp", "docker", 
            "kubernetes", "git", "linux", "machine learning", "deep learning", "nlp", "cv",
            "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "tableau", "power bi",
            "excel", "html", "css", "flask", "django", "fastapi", "streamlit", "ci/cd", "devops"
        ]
        
        extracted_skills = [skill for skill in skill_bank if skill in text.lower()]
        
        edu_keywords = ["bachelor", "master", "phd", "b.tech", "m.tech", "b.sc", "m.sc", "university", "college", "degree"]
        exp_keywords = ["experience", "years working", "job title", "manager", "engineer", "developer", "lead", "intern"]
        cert_keywords = ["certified", "certification", "aws certified", "pmp", "scrum master", "coursera", "udemy"]
        proj_keywords = ["project", "github repository", "portfolio", "capstone", "developed an app"]

        lines = text.split('\n')
        education = [line.strip() for line in lines if any(k in line.lower() for k in edu_keywords)]
        experience = [line.strip() for line in lines if any(k in line.lower() for k in exp_keywords)]
        certifications = [line.strip() for line in lines if any(k in line.lower() for k in cert_keywords)]
        projects = [line.strip() for line in lines if any(k in line.lower() for k in proj_keywords)]

        return {
            "skills": ", ".join(set(extracted_skills)) if extracted_skills else "Not Specified",
            "education": "; ".join(education[:3]) if education else "Not Specified",
            "experience": "; ".join(experience[:3]) if experience else "Not Specified",
            "certifications": "; ".join(certifications[:3]) if certifications else "Not Specified",
            "projects": "; ".join(projects[:3]) if projects else "Not Specified"
        }

    def compute_similarity(self, resume_text, jd_text):
        if not resume_text or not jd_text:
            return 0.0, 0.0, []
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        resume_words = set(resume_text.lower().split())
        jd_words = set(jd_text.lower().split())
        missing_keywords = list(jd_words.difference(resume_words))[:8]
        
        ai_score = int((sim_score * 60) + (min(len(resume_words)/100, 1) * 40))
        ai_score = min(max(ai_score, 10), 100)
        
        return float(sim_score), ai_score, missing_keywords

parser_engine = ResumeParserEngine()

# ==========================================
# 4. DEEP LEARNING SYSTEM (PyTorch)
# ==========================================
class SelectionPredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(SelectionPredictionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class PyTorchModelPipeline:
    def __init__(self):
        self.model_path = "hiring_model.pth"
        self.scaler = StandardScaler()
        self.input_dim = 3
        self.model = SelectionPredictionModel(self.input_dim)
        self._init_pipeline()

    def _init_pipeline(self):
        # Always generate synthetic baseline range mapping data to fit the structural Scaler uniformly
        np.random.seed(42)
        X_res_score = np.random.randint(40, 100, size=(200, 1))
        X_sim_score = np.random.uniform(0.2, 0.9, size=(200, 1))
        X_feats_density = np.random.uniform(1, 5, size=(200, 1))
        X = np.hstack((X_res_score, X_sim_score, X_feats_density))
        self.scaler.fit(X)

        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()
                return
            except:
                pass
        
        y = (X[:, 0]*0.5 + X[:, 1]*40 + X[:, 2]*5 > 65).astype(np.float32).reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        self.model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            out = self.model(X_tensor)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer.step()
            
        self.model.eval()
        torch.save(self.model.state_dict(), self.model_path)

    def predict_probability(self, resume_score, similarity_score, extracted_features_dict):
        feat_count = sum([1 for v in extracted_features_dict.values() if v and v != "Not Specified"])
        raw_features = np.array([[resume_score, similarity_score, feat_count]])
        
        # Uses the globally fit scaler context securely without losing mean metrics
        scaled_features = self.scaler.transform(raw_features)
        tensor_input = torch.tensor(scaled_features, dtype=torch.float32)
        
        with torch.no_grad():
            probability = self.model(tensor_input).item()
        return float(probability)

ai_predictor = PyTorchModelPipeline()

# ==========================================
# 5. USER INTERFACE GENERATION & APP ROUTING
# ==========================================
def run_app():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None

    if not st.session_state.logged_in:
        render_login_screen()
    else:
        render_dashboard_layout()

def render_login_screen():
    st.markdown("<h1 style='text-align: center; color: #3B82F6;'>🎯 HireSense AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9CA3AF;'>Intelligent Candidate Selection System</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("HR & Admin Authentication")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Authenticate & Enter System", use_container_width=True):
            user_records = db.fetch_all("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
            if user_records:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Access Granted!")
                st.rerun()
            else:
                st.error("Invalid credentials setup mapping. Access Denied.")

def render_dashboard_layout():
    st.sidebar.markdown("<h2 style='color:#3B82F6; font-weight:bold;'>HireSense AI Console</h2>", unsafe_allow_html=True)
    st.sidebar.markdown(f"👤 **Operator:** `{st.session_state.username}`")
    st.sidebar.markdown("---")
    
    navigation_menu = st.sidebar.radio(
        "Navigation Matrix Options",
        ["System Insights & Metrics", "Candidate Management Console", "Profile Pipeline Evaluator"]
    )
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Terminate Session Context / Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
        
    if navigation_menu == "System Insights & Metrics":
        render_insights_dashboard()
    elif navigation_menu == "Candidate Management Console":
        render_management_console()
    elif navigation_menu == "Profile Pipeline Evaluator":
        render_pipeline_evaluator()

def render_insights_dashboard():
    st.title("📊 System Insights & Analytics Dashboard")
    
    metrics_data = db.fetch_all("SELECT * FROM candidates")
    df = pd.DataFrame(metrics_data)
    
    if df.empty:
        st.info("The metrics subsystem requires functional data layers. Insert data through the Pipeline Evaluator tab.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Applicants</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
    with c2:
        selected_count = len(df[df['status'] == 'Selected'])
        st.markdown(f'<div class="metric-card"><div class="metric-title">Selected Talents</div><div class="metric-value">{selected_count}</div></div>', unsafe_allow_html=True)
    with c3:
        rejected_count = len(df[df['status'] == 'Rejected'])
        st.markdown(f'<div class="metric-card"><div class="metric-title">Rejected Profiles</div><div class="metric-value">{rejected_count}</div></div>', unsafe_allow_html=True)
    with c4:
        avg_score = int(df['resume_score'].mean()) if len(df) > 0 else 0
        st.markdown(f'<div class="metric-card"><div class="metric-title">Average AI Index Score</div><div class="metric-value">{avg_score}%</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("AI Performance Metric Distribution Matrix")
        fig_hist = px.histogram(df, x="resume_score", nbins=15, labels={'resume_score': 'AI Evaluation Score Index'}, color_discrete_sequence=['#3B82F6'])
        fig_hist.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_g2:
        st.subheader("Selection Status Percentages")
        fig_pie = px.pie(df, names="status", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Top Ranked Contenders Ranking Stack")
    top_ranked = df.sort_values(by="resume_score", ascending=False).head(5)
    st.dataframe(top_ranked[['name', 'email', 'resume_score', 'similarity_score', 'selection_probability', 'status']], use_container_width=True)

def render_management_console():
    st.title("🗂️ Candidate Management Console Matrix")
    
    candidates = db.fetch_all("SELECT * FROM candidates")
    if not candidates:
        st.warning("No candidate rows detected inside the relational database framework schema layout.")
        return
        
    df = pd.DataFrame(candidates)
    
    st.subheader("Search & Pipeline Filtering Protocols")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        search_query = st.text_input("Query Names or Specified Extracted Skill Sets")
    with col_f2:
        status_filter = st.selectbox("Pipeline Phase Filter Target", ["All Phases", "Pending", "Selected", "Rejected"])
        
    filtered_df = df.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df['name'].str.contains(search_query, case=False, na=False) |
            filtered_df['skills'].str.contains(search_query, case=False, na=False)
        ]
    if status_filter != "All Phases":
        filtered_df = filtered_df[filtered_df['status'] == status_filter]
        
    st.dataframe(filtered_df[['id', 'name', 'email', 'skills', 'resume_score', 'selection_probability', 'status']], use_container_width=True)
    
    st.markdown("---")
    st.subheader("Workflow Lifecycle State Manipulation Tools")
    col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
    with col_c1:
        target_id = st.number_input("Target Record System ID Index", min_value=1, step=1)
    with col_c2:
        new_status = st.selectbox("Assign Action Status", ["Pending", "Selected", "Rejected"])
    with col_c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Commit Phase Lifecycle Status Shift Matrix Update", use_container_width=True):
            success = db.execute_query("UPDATE candidates SET status=%s WHERE id=%s", (new_status, target_id))
            if success:
                st.success(f"Candidate Reference Entity [{target_id}] state mutated to '{new_status}' successfully.")
                st.rerun()

    st.markdown("---")
    st.subheader("Data Export Utilities Engine")
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Filtered Spreadsheet Report Metrics Stream Asset (.CSV)",
        data=csv_buffer.getvalue(),
        file_name="hiresense_filtered_candidate_report.csv",
        mime="text/csv",
        use_container_width=True
    )

def render_pipeline_evaluator():
    st.title("🧠 Profile Pipeline Evaluator Engine")
    
    with st.form("pipeline_evaluation_form"):
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown("### Profile Identity Metrics Metadata")
            cand_name = st.text_input("Full Legal Profile Name *")
            cand_email = st.text_input("Active Communications Email Channel Address *")
            cand_phone = st.text_input("Contact Primary Phone Line Number")
            uploaded_file = st.file_uploader("Document Profile CV Data Upload Stream (.PDF, .DOCX Asset Types)", type=["pdf", "docx"])
            
        with col_i2:
            st.markdown("### Pipeline Target Context Configuration")
            jd_text_input = st.text_area("Job Profile Requirement / Job Description (JD)", height=280)
            
        st.markdown("---")
        submit_eval = st.form_submit_button("Initiate Evaluation Vector Sequences & Predict Class", use_container_width=True)
        
    if submit_eval:
        if not cand_name or not cand_email or not uploaded_file or not jd_text_input:
            st.error("Missing critical evaluation payload attributes. Form validation constraint failures encountered.")
            return
            
        with st.spinner("Executing document parsing, matrix tokenization, and deep network layer inferences..."):
            parsed_raw_text = parser_engine.extract_text(uploaded_file)
            features = parser_engine.extract_features(parsed_raw_text)
            
            similarity, ai_score, missing_elements = parser_engine.compute_similarity(parsed_raw_text, jd_text_input)
            selection_prob = ai_predictor.predict_probability(ai_score, similarity, features)
            
            insert_query = """
                INSERT INTO candidates (name, email, phone, skills, education, experience, certifications, projects, resume_score, similarity_score, selection_probability, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'Pending')
            """
            params = (
                cand_name, cand_email, cand_phone, 
                features['skills'], features['education'], features['experience'], 
                features['certifications'], features['projects'], 
                ai_score, float(similarity), float(selection_prob)
            )
            
            db.execute_query(insert_query, params)
            st.success("Analysis Matrix successfully mapped, committed, and vectorized!")
            
            st.markdown("---")
            st.subheader("Evaluation Framework Telemetry Readouts Output")
            
            col_o1, col_o2, col_o3 = st.columns(3)
            with col_o1:
                st.metric(label="AI Score Rating Metric Index", value=f"{ai_score} / 100")
            with col_o2:
                st.metric(label="Text Vector Similarity Density Index Ratio", value=f"{round(float(similarity)*100, 2)}%")
            with col_o3:
                st.metric(label="PyTorch Network Selection Probability Confidence", value=f"{round(float(selection_prob)*100, 2)}%")
                
            st.markdown("#### Feature Extraction Parsing Output Metrics")
            st.json(features)
            
            if missing_elements:
                st.warning(f"💡 **Detected Missing Skillsets / Contextual Keywords Target Areas:** {', '.join(missing_elements)}")

# ==========================================
# 6. APPLICATION ENTRY SYSTEM INITIALIZER
# ==========================================
if __name__ == "__main__":
    run_app()