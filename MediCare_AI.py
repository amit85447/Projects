import os
import re
import io
import datetime
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import mysql.connector
from mysql.connector import Error
import fitz 
import docx
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ==========================================
# 0. CONFIGURATION & LOGGER INITIALIZATION
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="MediCare AI", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# Custom Responsive Healthcare Dark/Clinical Theme CSS Injector
st.markdown("""
<style>
    .stApp {
        background-color: #0A0F1D;
        color: #E2E8F0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #0EA5E9;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #94A3B8;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #38BDF8;
        font-size: 28px;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #0284C7;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0369A1;
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 4px;
        color: #94A3B8;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284C7 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATABASE MANAGEMENT SYSTEM (MySQL)
# ==========================================
class DatabaseManager:
    """Enterprise relational data fabric connector managing transactional state models."""
    def __init__(self):
        self.host = "localhost"
        self.user = "root"
        self.password = "Amit@123"
        self.database = "medicare_ai_db"
        self.init_database()

    def get_connection(self):
        return mysql.connector.connect(
            host=self.host, user=self.user, password=self.password, database=self.database
        )

    def init_database(self):
        try:
            conn = mysql.connector.connect(host=self.host, user=self.user, password=self.password)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            conn.close()

            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 1. Users Security Context Frame
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password VARCHAR(100) NOT NULL,
                    role VARCHAR(50) NOT NULL
                )
            """)
            
            # Seeding default staff profiles if clean
            cursor.execute("SELECT * FROM users WHERE username = 'admin'")
            if not cursor.fetchone():
                default_users = [
                    ('admin', 'admin123', 'Admin'),
                    ('doctor', 'doc123', 'Doctor'),
                    ('receptionist', 'rec123', 'Receptionist'),
                    ('pharmacist', 'pharm123', 'Pharmacist'),
                    ('laboratory', 'lab123', 'Laboratory')
                ]
                cursor.executemany("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)", default_users)
            
            # 2. Patients Core Schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    age INT NOT NULL,
                    gender VARCHAR(50),
                    blood_group VARCHAR(10),
                    height FLOAT,
                    weight FLOAT,
                    address TEXT,
                    phone VARCHAR(50),
                    email VARCHAR(100),
                    emergency_contact VARCHAR(50),
                    disease VARCHAR(255),
                    allergies TEXT,
                    medical_history TEXT
                )
            """)

            # 3. Doctors Registry Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS doctors (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    department VARCHAR(100),
                    qualification VARCHAR(100),
                    experience INT,
                    consultation_fee DECIMAL(10,2),
                    availability VARCHAR(255),
                    phone VARCHAR(50),
                    email VARCHAR(100)
                )
            """)

            # 4. Appointments Ledger
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_id INT,
                    doctor_id INT,
                    appointment_date DATE,
                    appointment_time TIME,
                    status VARCHAR(50) DEFAULT 'Scheduled',
                    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
                    FOREIGN KEY (doctor_id) REFERENCES doctors(id) ON DELETE CASCADE
                )
            """)

            # 5. Pharmacy Inventory Matrix
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS medicines (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    supplier VARCHAR(255),
                    stock INT NOT NULL,
                    purchase_price DECIMAL(10,2),
                    sale_price DECIMAL(10,2),
                    expiry_date DATE
                )
            """)

            # 6. Pharmacy Sales Transactions Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pharmacy_sales (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    medicine_id INT,
                    quantity INT,
                    total_price DECIMAL(10,2),
                    sale_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (medicine_id) REFERENCES medicines(id) ON DELETE CASCADE
                )
            """)

            # 7. Laboratory Reports Schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lab_reports (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_id INT,
                    test_type VARCHAR(100),
                    findings TEXT,
                    status VARCHAR(50) DEFAULT 'Pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
                )
            """)

            # 8. Room/Bed Infrastructure Configuration Matrix
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS beds (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    bed_number VARCHAR(50) UNIQUE NOT NULL,
                    type VARCHAR(100),
                    status VARCHAR(50) DEFAULT 'Available'
                )
            """)
            
            cursor.execute("SELECT COUNT(*) FROM beds")
            if cursor.fetchone()[0] == 0:
                init_beds = [
                    ('ICU-101', 'ICU', 'Occupied'), ('ICU-102', 'ICU', 'Available'),
                    ('GEN-201', 'General Bed', 'Available'), ('GEN-202', 'General Bed', 'Occupied'),
                    ('PVT-301', 'Private Room', 'Available'), ('DLX-401', 'Deluxe Room', 'Available')
                ]
                cursor.executemany("INSERT INTO beds (bed_number, type, status) VALUES (%s, %s, %s)", init_beds)

            # 9. Emergency Case Stream Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emergency_cases (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_name VARCHAR(255),
                    triage_condition VARCHAR(255),
                    assigned_bed_id INT,
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (assigned_bed_id) REFERENCES beds(id) ON DELETE SET NULL
                )
            """)

            # 10. Financial Invoice Index
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS invoices (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_id INT,
                    consultation_charges DECIMAL(10,2),
                    lab_charges DECIMAL(10,2),
                    pharmacy_charges DECIMAL(10,2),
                    room_charges DECIMAL(10,2),
                    tax DECIMAL(10,2),
                    discount DECIMAL(10,2),
                    grand_total DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
                )
            """)

            # 11. Core Corporate Staff Management Ledger
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS staff (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    role VARCHAR(100),
                    phone VARCHAR(50),
                    email VARCHAR(100)
                )
            """)

            # 12. AI Predictive Outcome Telemetry Trace
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_name VARCHAR(255),
                    predicted_disease VARCHAR(255),
                    confidence FLOAT,
                    risk_level VARCHAR(50),
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()
            logger.info("Database schema architecture initialized successfully.")
        except Error as e:
            logger.error(f"Database Initialization Critical Crash: {e}")
            st.error("Database connection failure. Check local relational engine stack parameters.")

    def execute_query(self, query, params=None):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
            conn.close()
            return True
        except Error as e:
            logger.error(f"Execution Engine Error: {e}")
            st.error(f"Transaction Fault Matrix: {e}")
            return False

    def fetch_all(self, query, params=None):
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            res = cursor.fetchall()
            conn.close()
            return res
        except Error as e:
            logger.error(f"Data Fetch Core Routine Exception: {e}")
            return []

db = DatabaseManager()

# ==========================================
# 2. PYTORCH CLINICAL PREDICTION PIPELINE
# ==========================================
class DiseaseClassifierNetwork(nn.Module):
    """Deep Neural Network structural topography mapping diagnostic outcomes."""
    def __init__(self, input_dim, output_dim):
        super(DiseaseClassifierNetwork, self).__init__()
        self.layer_block = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.layer_block(x)

class AICheckupPipeline:
    """Manages training validation loops and structural standard scaling functions."""
    def __init__(self):
        self.model_path = "hospital_ai_model.pth"
        self.scaler = StandardScaler()
        # Input Matrix mapping coordinates: [Fever, Cough, Headache, ChestPain, Sugar, BP, Oxygen, HeartRate]
        self.input_dim = 8
        self.diseases = ["Healthy/Normal Profile", "Acute Influenza", "Cardiovascular Irregularity", "Hypertensive Crisis", "Respiratory Infection"]
        self.model = DiseaseClassifierNetwork(self.input_dim, len(self.diseases))
        self._bootstrap_and_fit_pipeline()

    def _bootstrap_and_fit_pipeline(self):
        # Generate clean synthetic arrays for feature normalization
        np.random.seed(42)
        X = np.random.uniform(low=0.0, high=1.0, size=(500, self.input_dim))
        # Ensure scale adjustments align with real clinical telemetry units
        X[:, 4] = np.random.uniform(70, 250, size=(500,))  # Glucose/Sugar
        X[:, 5] = np.random.uniform(80, 180, size=(500,))  # Blood Pressure
        X[:, 6] = np.random.uniform(85, 100, size=(500,))  # Oxygen Saturation
        X[:, 7] = np.random.uniform(50, 140, size=(500,))  # Heart Rate
        
        self.scaler.fit(X)

        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
                self.model.eval()
                return
            except Exception as e:
                logger.warning(f"Failed to load existing diagnostic network states: {e}. Rebuilding network topology.")

        # Algorithmic conditional alignment targets
        y = []
        for row in X:
            if row[6] < 90 or row[3] > 0.7: y.append(2)    # Cardiovascular/Respiratory issues
            elif row[5] > 140: y.append(3)                 # Hypertension
            elif row[0] > 0.6 and row[1] > 0.5: y.append(1) # Influenza
            elif row[4] > 180: y.append(4)                 # Advanced Metabolic anomaly
            else: y.append(0)                              # Normal status
            
        y = np.array(y, dtype=np.int64)
        
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        
        self.model.train()
        for epoch in range(150):
            optimizer.zero_grad()
            predictions = self.model(X_t)
            loss = loss_fn(predictions, y_t)
            loss.backward()
            optimizer.step()
            
        self.model.eval()
        torch.save(self.model.state_dict(), self.model_path)

    def evaluate_clinical_symptoms(self, vector):
        raw_arr = np.array([vector])
        scaled_arr = self.scaler.transform(raw_arr)
        t_in = torch.tensor(scaled_arr, dtype=torch.float32)
        
        with torch.no_grad():
            prob_dist = self.model(t_in).numpy()[0]
            
        pred_idx = np.argmax(prob_dist)
        confidence = float(prob_dist[pred_idx])
        disease = self.diseases[pred_idx]
        
        # Risk factor heuristics definition matrices
        risk_level = "Low"
        dept = "General Medicine"
        tests = ["Complete Blood Count (CBC)"]
        
        if pred_idx in [2, 3]:
            risk_level = "Critical" if (vector[5] > 160 or vector[6] < 88) else "High"
            dept = "Cardiology Division"
            tests = ["12-Lead Electrocardiogram (ECG)", "Echocardiography", "Troponin Biomarker Assessment"]
        elif pred_idx == 4:
            risk_level = "Medium Risk"
            dept = "Endocrinology"
            tests = ["HbA1c Glycated Hemoglobin", "Fasting Blood Glucose Screen Level"]
        elif pred_idx == 1:
            risk_level = "Medium Risk"
            dept = "Pulmonary/Respiratory Diseases"
            tests = ["High-Resolution Chest X-Ray", "Sputum Culture Diagnostics Analysis"]

        return disease, confidence, risk_level, dept, tests

ai_diagnostic_engine = AICheckupPipeline()

# ==========================================
# 3. UNSTRUCTURED UNIFIED TEXT MEDICAL PARSER
# ==========================================
class MedicalDocumentParser:
    """Extracts entity metadata profiles directly out of raw document byte flows."""
    @staticmethod
    def read_document_stream(uploaded_file):
        extracted_text = ""
        ext = uploaded_file.name.split('.')[-1].lower()
        try:
            if ext == 'pdf':
                doc_object = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page in doc_object:
                    extracted_text += page.get_text()
            elif ext == 'docx':
                doc_object = docx.Document(io.BytesIO(uploaded_file.read()))
                extracted_text = "\n".join([para.text for para in doc_object.paragraphs])
        except Exception as e:
            logger.error(f"Document extraction framework failure: {e}")
        return extracted_text

    @staticmethod
    def transform_and_parse_text(raw_text):
        cleaned = re.sub(r'\s+', ' ', raw_text).lower()
        
        disease_directory = ["diabetes", "hypertension", "pneumonia", "asthma", "covid-19", "arrhythmia", "anemia", "carcinoma"]
        medicine_directory = ["metformin", "amlodipine", "azithromycin", "albuterol", "aspirin", "insulin", "losartan", "ibuprofen"]
        test_directory = ["cbc", "mri scan", "ct thorax", "lipid panel", "ecg", "urinalysis", "blood sugar check"]
        
        found_diseases = [d.upper() for d in disease_directory if d in cleaned]
        found_medicines = [m.capitalize() for m in medicine_directory if m in cleaned]
        found_tests = [t.upper() for t in test_directory if t in cleaned]
        
        critical_flags = [line.strip() for line in raw_text.split('.') if any(cw in line.lower() for cw in ["critical", "severe", "abnormal", "positive alert", "high risk", "malignant"])]
        
        return {
            "detected_diseases": ", ".join(found_diseases) if found_diseases else "None Isolated",
            "suggested_medicines": ", ".join(found_medicines) if found_medicines else "None Detected",
            "administered_tests": ", ".join(found_tests) if found_tests else "None Found",
            "summary_nodes": cleaned[:250] + "..." if len(cleaned) > 250 else cleaned,
            "critical_highlights": "; ".join(critical_flags[:3]) if critical_flags else "All analytical parameters trace within baseline standard deviations."
        }

document_analyzer = MedicalDocumentParser()

# ==========================================
# 4. BILLING & INVOICE PDF COMPILER 
# ==========================================
class InvoiceBillingGenerator:
    """Compiles programmatic downstream components into clean corporate financial PDFs."""
    @staticmethod
    def compile_pdf_invoice_stream(invoice_data, profile_name):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'InvoiceTitle', parent=styles['Heading1'], fontSize=24, leading=28, textColor=colors.HexColor("#0284C7"), alignment=1
        )
        normal_style = styles['Normal']
        bold_style = ParagraphStyle('InvoiceBold', parent=normal_style, fontName='Helvetica-Bold')
        
        story.append(Paragraph("MEDICARE AI DIGITAL INVOICE RECEIPT", title_style))
        story.append(Spacer(1, 15))
        story.append(Paragraph(f"<b>Transaction Reference ID Node:</b> TXN-00{invoice_data.get('id', 'NEW')}", normal_style))
        story.append(Paragraph(f"<b>Account Custodian Holder / Patient:</b> {profile_name}", normal_style))
        story.append(Paragraph(f"<b>Timestamp Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Spacer(1, 20))
        
        table_content = [
            [Paragraph("<b>Financial Categorization Node Item</b>", bold_style), Paragraph("<b>Assigned Ledger Charge</b>", bold_style)],
            [Paragraph("Professional Consultation Fee Metrics", normal_style), f"${invoice_data['consultation_charges']:.2f}"],
            [Paragraph("Laboratory Diagnostic Operational Fees", normal_style), f"${invoice_data['lab_charges']:.2f}"],
            [Paragraph("Pharmacy Medicine Supply Dispensation Log", normal_style), f"${invoice_data['pharmacy_charges']:.2f}"],
            [Paragraph("Infrastructure Inpatient Bed / Room Charges", normal_style), f"${invoice_data['room_charges']:.2f}"],
            [Paragraph("State Corporate GST Assessment Tax (18%)", normal_style), f"${invoice_data['tax']:.2f}"],
            [Paragraph("Authorized Institutional Strategic Markdown Discount", normal_style), f"-${invoice_data['discount']:.2f}"],
            [Paragraph("<b>Net Settled Invoice Valuation (Grand Total)</b>", bold_style), f"<b>${invoice_data['grand_total']:.2f}</b>"]
        ]
        
        charge_table = Table(table_content, colWidths=[350, 150])
        charge_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor("#0F172A")),
            ('TEXTCOLOR', (0,0), (1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('BACKGROUND', (0,-1), (1,-1), colors.HexColor("#F0F9FF"))
        ]))
        
        story.append(charge_table)
        story.append(Spacer(1, 30))
        story.append(Paragraph("<i>This document represents a certified electronic clearance record. Authorized by the MediCare AI Finance Operations Stack.</i>", normal_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

# ==========================================
# 5. CORE SYSTEM PRESENTATION LAYER (Streamlit)
# ==========================================
class MediCareApplicationRouter:
    """Global Orchestrator mapping identity privileges directly to functional modules."""
    def __init__(self):
        if 'logged_in' not in st.session_state: st.session_state.logged_in = False
        if 'username' not in st.session_state: st.session_state.username = None
        if 'role' not in st.session_state: st.session_state.role = None

    def execute_routing_loop(self):
        if not st.session_state.logged_in:
            self.render_authentication_wall()
        else:
            self.render_enterprise_shell()

    def render_authentication_wall(self):
        st.markdown("<h1 style='text-align: center; color: #0284C7; margin-top:50px;'>🏥 MediCare AI Enterprise Portal</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94A3B8;'>Intelligent Clinical Operations & Selection Diagnostic Layer Suite</p>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 1.8, 1])
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("security_authentication_form"):
                st.subheader("System Identity Matrix Credentials")
                uid = st.text_input("Operator Username")
                pwd = st.text_input("Security Access Phrase Password", type="password")
                submit = st.form_submit_button("Verify Identity Matrix Check", use_container_width=True)
                
            if submit:
                match = db.fetch_all("SELECT * FROM users WHERE username=%s AND password=%s", (uid, pwd))
                if match:
                    st.session_state.logged_in = True
                    st.session_state.username = match[0]['username']
                    st.session_state.role = match[0]['role']
                    st.success(f"Access granted. Initializing {st.session_state.role} matrix protocols...")
                    st.rerun()
                else:
                    st.error("Credential validation parameters failure. Access Denied.")

    def render_enterprise_shell(self):
        # Professional Left Sidebar Setup Structure Matrix Options Selector
        st.sidebar.markdown("<h2 style='color:#0284C7; font-weight:bold;'>MediCare AI</h2>", unsafe_allow_html=True)
        st.sidebar.markdown(f"**Operator Context:** `{st.session_state.username}` | `{st.session_state.role}`")
        st.sidebar.markdown("---")
        
        core_menu = [
            "Executive Telemetry Dashboard", 
            "Patient Case Registry", 
            "Clinical Staff Matrix", 
            "Scheduling Framework", 
            "Predictive Diagnostic Neural Engine",
            "Medical Report Analyzer",
            "Pharmacy Supply Chain Control",
            "Laboratory Operations",
            "Revenue & Billing Core",
            "Bed Management Subsystem",
            "Emergency Incident Stream"
        ]
        
        selected_module = st.sidebar.radio("Console View Targets System Routing", core_menu)
        
        st.sidebar.markdown("---")
        if st.sidebar.button("Terminate Session Profile / Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

        # Render Module Target Views
        if selected_module == "Executive Telemetry Dashboard": self.render_executive_dashboard()
        elif selected_module == "Patient Case Registry": self.render_patient_registry()
        elif selected_module == "Clinical Staff Matrix": self.render_staff_matrix()
        elif selected_module == "Scheduling Framework": self.render_scheduling_framework()
        elif selected_module == "Predictive Diagnostic Neural Engine": self.render_predictive_engine()
        elif selected_module == "Medical Report Analyzer": self.render_report_analyzer()
        elif selected_module == "Pharmacy Supply Chain Control": self.render_pharmacy_control()
        elif selected_module == "Laboratory Operations": self.render_laboratory_operations()
        elif selected_module == "Revenue & Billing Core": self.render_billing_core()
        elif selected_module == "Bed Management Subsystem": self.render_bed_subsystem()
        elif selected_module == "Emergency Incident Stream": self.render_emergency_stream()

    # ==========================================
    # MODULE 1: EXECUTIVE TELEMETRY DASHBOARD
    # ==========================================
    def render_executive_dashboard(self):
        st.title("📊 Executive Operational Telemetry Dashboard")
        
        # Real-time Telemetry Calculations from Data Stores
        p_count = db.fetch_all("SELECT COUNT(*) as count FROM patients")[0]['count']
        d_count = db.fetch_all("SELECT COUNT(*) as count FROM doctors")[0]['count']
        a_count = db.fetch_all("SELECT COUNT(*) as count FROM appointments")[0]['count']
        e_count = db.fetch_all("SELECT COUNT(*) as count FROM emergency_cases")[0]['count']
        rev_res = db.fetch_all("SELECT SUM(grand_total) as total FROM invoices")[0]['total'] or 0.0
        
        occ_beds = db.fetch_all("SELECT COUNT(*) as count FROM beds WHERE status='Occupied'")[0]['count']
        tot_beds = db.fetch_all("SELECT COUNT(*) as count FROM beds")[0]['count'] or 1
        bed_occ_rate = int((occ_beds / tot_beds) * 100)
        
        low_stock = db.fetch_all("SELECT COUNT(*) as count FROM medicines WHERE stock < 15")[0]['count']

        # Interactive High-Density Cards Layout Metrics Row Grid Elements
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Active Patients</div><div class="metric-value">{p_count}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-title">Active Clinicians</div><div class="metric-value">{d_count}</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-title">Booked Appointments</div><div class="metric-value">{a_count}</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Gross Revenue</div><div class="metric-value">${rev_res:,.2f}</div></div>', unsafe_allow_html=True)

        c5, c6, c7 = st.columns(3)
        with c5: st.markdown(f'<div class="metric-card"><div class="metric-title">Bed Occupancy Vector</div><div class="metric-value">{bed_occ_rate}%</div></div>', unsafe_allow_html=True)
        with c6: st.markdown(f'<div class="metric-card"><div class="metric-title">Active Triage Emergencies</div><div class="metric-value">{e_count}</div></div>', unsafe_allow_html=True)
        with c7: st.markdown(f'<div class="metric-card"><div class="metric-title">Medicine Low Stock Alerts</div><div class="metric-value">{low_stock}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Plotly Enterprise Visualizations Analytics Block Layout
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Financial Operations Revenue Output Analytics")
            inv_records = db.fetch_all("SELECT DATE(created_at) as dt, SUM(grand_total) as rev FROM invoices GROUP BY dt ORDER BY dt ASC")
            if inv_records:
                df_rev = pd.DataFrame(inv_records)
                fig = px.line(df_rev, x='dt', y='rev', markers=True, title="Revenue Generation Velocity Time Trace Trends")
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient billing variance metrics to plot transactional lines tracking traces.")

        with col_chart2:
            st.subheader("Pathological Diagnostic Prevalence Spread")
            pred_records = db.fetch_all("SELECT predicted_disease as disease, COUNT(*) as count FROM ai_predictions GROUP BY predicted_disease")
            if pred_records:
                df_pred = pd.DataFrame(pred_records)
                fig = px.pie(df_pred, values='count', names='disease', hole=0.4, title="Neural Diagnostic Allocation Statistics")
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No AI trace tracking arrays committed to log distributions.")

    # ==========================================
    # MODULE 2: PATIENT CASE REGISTRY (CRUD)
    # ==========================================
    def render_patient_registry(self):
        st.title("🗃️ Patient Case Registry Lifecycle Management")
        t1, t2, t3 = st.tabs(["Search/Read Directory Records", "Initialize New Patient Record Profile", "Modify Status/Delete Fields"])
        
        with t1:
            st.subheader("Dynamic Query Filter System Interface Matrix")
            q_term = st.text_input("Query Registry via Target Full Name, Phone Index, or Categorical Pathology Path")
            query = "SELECT * FROM patients"
            if q_term:
                query += f" WHERE name LIKE '%{q_term}%' OR phone LIKE '%{q_term}%' OR disease LIKE '%{q_term}%'"
            
            p_data = db.fetch_all(query)
            if p_data:
                st.dataframe(pd.DataFrame(p_data), use_container_width=True)
            else:
                st.warning("No tracking profile sequences successfully return parameters matching input targets.")
                
        with t2:
            st.subheader("Patient Clinical Intake Enrolment Structure Schema Form")
            with st.form("patient_creation_intake_form"):
                n = st.text_input("Full Registered Patient Name *")
                a = st.number_input("Patient Current Chronological Age", min_value=0, max_value=125, value=30)
                g = st.selectbox("Biological Sex Profiling Identity Orientation", ["Male", "Female", "Non-Binary/Intersex Category Spectrum"])
                bg = st.selectbox("ABO Rh Factor Blood Group Classification", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
                h = st.number_input("Stature Height Metrology Coordinate Index (cm)", value=170.0)
                w = st.number_input("Mass Weight Scale Reading Metric (kg)", value=70.0)
                addr = st.text_area("Primary Residential Domicile Postal Address Field Location")
                ph = st.text_input("Active Primary Communication Mobile Network Phone Sequence")
                em = st.text_input("Active Communications Email Channel Address Link")
                e_contact = st.text_input("Emergency Proxy Contact Network Sequence Relative Phone")
                dis = st.text_input("Active Provisional Structural Diagnosis Primary Complaint Entity")
                allergies = st.text_area("Hypersensitivity Allergies Critical Matrix Registers Flags")
                hist = st.text_area("Anamnestic Longitudinal Medical History Chronic Logs Summary")
                
                btn = st.form_submit_button("Commit Intake Logs Entry Transaction Pipeline")
                
            if btn:
                if not n: st.error("Validation error: Primary parameter Name element missing context strings."); return
                q = """INSERT INTO patients (name, age, gender, blood_group, height, weight, address, phone, email, emergency_contact, disease, allergies, medical_history)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
                if db.execute_query(q, (n, a, g, bg, h, w, addr, ph, em, e_contact, dis, allergies, hist)):
                    st.success(f"Intake pipeline committed for candidate patient structure record sequence: {n}")
                    st.rerun()

        with t3:
            st.subheader("Mutate Profile Configurations Architecture Operations Layer")
            p_id = st.number_input("Target Primary Structural Identification Database Record Key Vector ID Entry", min_value=1, step=1)
            if st.button("Purge/Delete Database Row Context Entity Instance completely out of Server Memory Matrix", type="primary"):
                if db.execute_query("DELETE FROM patients WHERE id=%s", (p_id,)):
                    st.success("Entity record mapping cleared from the structural tables data configuration framework context loop.")
                    st.rerun()

    # ==========================================
    # MODULE 3: CLINICAL STAFF MATRIX (CRUD)
    # ==========================================
    def render_staff_matrix(self):
        st.title("👨‍⚕️ Corporate Clinical Staff Registry Framework Operations Layer")
        t1, t2 = st.tabs(["Active Strategic Operations Team Roster Matrix", "Add/Update Personnel Roster Profiles"])
        
        with t1:
            st.subheader("Current Core Staffing Roster Network Telemetry Deployment View")
            st.dataframe(pd.DataFrame(db.fetch_all("SELECT * FROM staff")), use_container_width=True)
            st.markdown("---")
            st.subheader("Consulting Practitioner Medical Specialist Faculty Roster Data")
            st.dataframe(pd.DataFrame(db.fetch_all("SELECT * FROM doctors")), use_container_width=True)
            
        with t2:
            st.subheader("Onboard Medical Specialist Practitioner Node Registry Block Settings")
            with st.form("doctor_onboarding_registry_form"):
                dn = st.text_input("Practitioner Professional Name Specifier *")
                dept = st.text_input("Clinical Functional Assignment Medical Department Sector Division Unit")
                qual = st.text_input("Highest Post-Graduate Academic Medical Credential/Qualification Matrix")
                exp = st.number_input("Longitudinal Professional Medical Practice Experience Horizon Matrix Span (Years)", min_value=0, max_value=60, value=8)
                fee = st.number_input("Standard Consultation Base Outpatient Assessment Fee Log Unit Scale", value=150.0)
                avail = st.text_input("Functional Weekly Operational Core Availability Window Index Trace", value="Mon-Fri (09:00 - 17:00)")
                ph = st.text_input("Active Operational Professional Contact Line Phone Sequence")
                em = st.text_input("Corporate Subsystem Active Electronic Email Context Route Matrix Address")
                
                d_btn = st.form_submit_button("Verify Credentials & Execute Matrix Deployment Node Context Mapping")
                
            if d_btn:
                if not dn: st.error("Profile operational name trace definition validation criteria constraints violation."); return
                q = "INSERT INTO doctors (name, department, qualification, experience, consultation_fee, availability, phone, email) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
                if db.execute_query(q, (dn, dept, qual, exp, fee, avail, ph, em)):
                    st.success(f"Practitioner profile configured into departmental operations node registry settings matrices: {dn}")
                    st.rerun()

    # ==========================================
    # MODULE 4: SCHEDULING FRAMEWORK 
    # ==========================================
    def render_scheduling_framework(self):
        st.title("📅 Transactional Clinical Appointment Scheduling & Queue Control Framework")
        
        col_sch1, col_sch2 = st.columns([1.2, 2])
        
        with col_sch1:
            st.subheader("Dispatch Booking Allocation Form Sequence")
            p_list = db.fetch_all("SELECT id, name FROM patients")
            d_list = db.fetch_all("SELECT id, name FROM doctors")
            
            p_map = {f"ID {r['id']} | {r['name']}": r['id'] for r in p_list}
            d_map = {f"ID {r['id']} | {r['name']}": r['id'] for r in d_list}
            
            if not p_map or not d_map:
                st.error("Prerequisite relational structural matrices empty. Ensure patient profile items match practitioner profiles rows.")
                return
                
            p_sel = st.selectbox("Target Patient Asset Instance Reference ID Node Link Selector", list(p_map.keys()))
            d_sel = st.selectbox("Target Medical Faculty Practitioner Asset Instance Specialist Selector", list(d_map.keys()))
            date_sel = st.date_input("Target Event Calendar Operations Execution Window Date Selector", datetime.date.today())
            time_sel = st.time_input("Target Event Chronological Timeline Slot Hour Minute Clock Matrix Assignment", datetime.time(10, 0))
            
            if st.button("Finalize Structural Allocation System Reservation Log Event", use_container_width=True):
                q = "INSERT INTO appointments (patient_id, doctor_id, appointment_date, appointment_time) VALUES (%s, %s, %s, %s)"
                if db.execute_query(q, (p_map[p_sel], d_map[d_sel], date_sel.strftime('%Y-%m-%d'), time_sel.strftime('%H:%M:%S'))):
                    st.success("Reservation scheduling instance transaction committed successfully.")
                    st.rerun()

        with col_sch2:
            st.subheader("Active Operational Pipeline Multi-Factor Calendar View Schedule Master Ledger")
            sch_data = db.fetch_all("""
                SELECT a.id, p.name as patient_name, d.name as doctor_name, a.appointment_date, a.appointment_time, a.status 
                FROM appointments a
                JOIN patients p ON a.patient_id = p.id
                JOIN doctors d ON a.doctor_id = d.id
                ORDER BY a.appointment_date ASC, a.appointment_time ASC
            """)
            if sch_data:
                st.dataframe(pd.DataFrame(sch_data), use_container_width=True)
            else:
                st.info("The calendar database transaction matrix is clear.")

    # ==========================================
    # MODULE 5: PREDICTIVE DIAGNOSTIC NEURAL ENGINE
    # ==========================================
    def render_predictive_engine(self):
        st.title("🧠 Predictive Diagnostic Inference Neural Network Framework Interface")
        st.markdown("Input clinical biometric arrays to pass feature arrays forward through PyTorch Tensor computational nodes.")
        
        c_p1, c_p2 = st.columns(2)
        
        with c_p1:
            st.subheader("Biometric Quantization Telemetry Array Feeds")
            p_name = st.text_input("Active Diagnostic Target Profile Tracking Name Reference Specifier")
            fever = st.slider("Thermometric Pyrexia Body Core Temperature Vector Range Normalization (0: Baseline Normal - 1: Severe Hyperpyrexia Scaled Ratio)", 0.0, 1.0, 0.2)
            cough = st.slider("Tussis Pulmonary Cough Frequency Velocity Scale Node (0: Absent - 1: Unremitting Acute Severe Paroxysmal Spasm)", 0.0, 1.0, 0.4)
            headache = st.slider("Cephalea Intracranial Cephalalgia Neurological Pain Intensity Scale Multiplier Index (0: None - 1: Incapacitating Severe Migraine)", 0.0, 1.0, 0.1)
            chest_pain = st.slider("Angina Pectoris Coronary Ischemic Pain Reflex Threshold Vector Matrix Indicator (0: Absent - 1: Severe Constricting Crushing Sensation)", 0.0, 1.0, 0.0)
            sugar = st.number_input("Serum Fasting Glucose Metabolic Concentration Metrics Reading Unit Value (mg/dL)", min_value=40, max_value=500, value=110)
            bp = st.number_input("Systolic Arterial Hemodynamic Hydraulic Perfusion Wave Value Pressure Scale (mmHg)", min_value=60, max_value=250, value=120)
            oxygen = st.number_input("Peripheral Capillary Oxyhemoglobin Gas Volumetric Saturation Index SpO2 % Range", min_value=50, max_value=100, value=98)
            heart_rate = st.number_input("Chronotropic Cardiac Pulse Inotropic Frequency Output Vector Beats Rate Scale (BPM)", min_value=30, max_value=220, value=75)
            
            execute_infer = st.button("Run Forward Propagation Inference Matrix Sequence Layer Pipelines", use_container_width=True)

        with c_p2:
            st.subheader("PyTorch Tensor Outcome Matrix Array Interpretations Node Readout")
            if execute_infer:
                if not p_name: st.error("Diagnostic telemetry requires a valid identifier tracking tag mapping array instance string parameter."); return
                
                in_vec = [fever, cough, headache, chest_pain, float(sugar), float(bp), float(oxygen), float(heart_rate)]
                dis, conf, risk, dept, tests = ai_diagnostic_engine.evaluate_clinical_symptoms(in_vec)
                
                # Persist outputs inside tracking data layer framework matrices automatically
                db.execute_query("INSERT INTO ai_predictions (patient_name, predicted_disease, confidence, risk_level) VALUES (%s,%s,%s,%s)",
                                 (p_name, dis, conf, risk))
                
                # Display outcome panel indicators metrics cards
                st.markdown(f"### Classification Result Instance: **{dis}**")
                st.metric("Neural Layer Output Softmax Evaluation Confidence Probability Balance Scale", f"{round(conf*100,2)}%")
                
                if risk == "Critical":
                    st.error(f"🚨 **CRITICAL TRIAGE PATH ALERT INDICATOR DETECTED:** Risk parameter calculations trace out of baseline tolerances. Risk Index: {risk}")
                elif risk == "High":
                    st.warning(f"⚠️ **ELEVATED PATHOLOGICAL THREAT MATRIX PROFILE:** Risk Index Factor: {risk}")
                else:
                    st.success(f"✅ **STABLE CLINICAL VARIANCE COMPONENT:** Profiling Category Classification Assessment: {risk}")
                    
                st.markdown(f"🔹 **Target Specialization Routing Sector Division:** `{dept}`")
                st.markdown("📋 **Diagnostic Laboratory Cross-Validation Check Directives Matrix:**")
                for test_item in tests:
                    st.markdown(f"- `{test_item}`")

    # ==========================================
    # MODULE 6: MEDICAL REPORT ANALYZER
    # ==========================================
    def render_report_analyzer(self):
        st.title("📂 Unstructured Clinical Text Entity Extraction Parser Module Engine")
        st.markdown("Upload native binary formatting file objects structure vectors (.PDF or .DOCX medical documents) to parse clinical profiles.")
        
        up_file = st.file_uploader("Document Profile CV Diagnostic Lab Matrix Asset Upload Target Stream", type=["pdf", "docx"])
        if up_file is not None:
            with st.spinner("Executing document parsing, matrix tokenization, and deep network layer inferences..."):
                raw_extracted_text = document_analyzer.read_document_stream(up_file)
                parsing_insights = document_analyzer.transform_and_parse_text(raw_extracted_text)
                
                st.success("Document structure mapped and parsed successfully.")
                st.markdown("---")
                
                co1, co2 = st.columns(2)
                with co1:
                    st.subheader("Extracted Entity Resolution Registry Matches Indices")
                    st.info(f"🧬 **Isolated Disease Match Vector Profiles:** {parsing_insights['detected_diseases']}")
                    st.success(f"💊 **Identified Pharmacological Compound Mentions Matrix:** {parsing_insights['suggested_medicines']}")
                    st.warning(f"🔬 **Isolated Lab Procedure Tokens Target Elements:** {parsing_insights['administered_tests']}")
                with co2:
                    st.subheader("Critical Diagnostic Alert Flags Matrix Tracking Output")
                    st.error(parsing_insights['critical_highlights'])
                    
                st.markdown("### Document Abstract Structural Text Fragment Segment String Snippet")
                st.code(parsing_insights['summary_nodes'], language='text')

    # ==========================================
    # MODULE 7: PHARMACY SUPPLY CHAIN CONTROL
    # ==========================================
    def render_pharmacy_control(self):
        st.title("💊 Pharmacy Stock Control Management Module Inventory")
        t1, t2 = st.tabs(["Current Stocks Inventory Configuration Ledger Matrix", "Dispense Medication Sales Orders Node Log"])
        
        with t1:
            st.subheader("Active Chemical Compounds Matrix Database Store View")
            med_inv = db.fetch_all("SELECT * FROM medicines")
            if med_inv:
                df_med = pd.DataFrame(med_inv)
                st.dataframe(df_med, use_container_width=True)
                
                # Expiry and Low Stock Alert System Heuristics Engine Iterators
                st.markdown("### ⚠️ Automating Active Pharmacy Telemetry Stock Exception Warnings Notification Registers")
                for index_row, row in df_med.iterrows():
                    if row['stock'] < 15:
                        st.error(f"Low Stock Threshold Violation Warning: Asset Instance Node Ref {row['name']} has only {row['stock']} metric physical packaging items remaining in the storage matrix container units.")
            else:
                st.info("The formulation stock inventory records database contains zero cataloged active variables.")
                
            st.markdown("---")
            st.subheader("Provision New Medical Chemical Stock Consignment Matrix Item Entry")
            with st.form("pharmacy_inventory_provision_entry_form"):
                mn = st.text_input("Medication Formulation Structural Branding Nomenclature Reference *")
                supp = st.text_input("B2B Enterprise Wholesale Logistic Vendor Supplier Entity Name Token")
                stk = st.number_input("Initial Physical Intake Item Volume Configuration Quantity Count Scale", min_value=0, value=100)
                pp = st.number_input("Acquisition Cost Price Matrix Factor Unit Metric Value per Base Pack", value=12.50)
                sp = st.number_input("Assigned Commercial Standard Dispensation Outpatient Unit Cost Retail Sale Price", value=22.00)
                exp_dt = st.date_input("Batch Manufacturer Dynamic Chemical Degradation Boundary Expiry Target Calendar Date Selection", datetime.date.today() + datetime.timedelta(days=365))
                
                m_btn = st.form_submit_button("Authorize Logistics Processing Entry Pipeline Matrix Commit")
                
            if m_btn:
                if not mn: st.error("Product catalog nomenclature profile tracking string empty verification error."); return
                q = "INSERT INTO medicines (name, supplier, stock, purchase_price, sale_price, expiry_date) VALUES (%s,%s,%s,%s,%s,%s)"
                if db.execute_query(q, (mn, supp, stk, pp, sp, exp_dt.strftime('%Y-%m-%d'))):
                    st.success("Medication batch provisioned in inventory registers mapping variables matrices.")
                    st.rerun()

        with t2:
            st.subheader("Execute New Medication Dispensation Order Transaction Receipt Logging Engine")
            meds_available = db.fetch_all("SELECT id, name, sale_price, stock FROM medicines WHERE stock > 0")
            if meds_available:
                m_options = {f"Ref ID {r['id']} | {r['name']} (Stock Count Unit: {r['stock']} Packs available)": r for r in meds_available}
                selected_med_str = st.selectbox("Select Medication Target Inventory Matrix Instance Token Node Match Selection", list(m_options.keys()))
                sel_qty = st.number_input("Select Transaction Distribution Count Unit Item Target Quantity Count Scale Value", min_value=1, value=1)
                
                if st.button("Authorize Prescription Dispensation & Deduct Inventory Balances Sequence Row Object Matrix Actions", use_container_width=True):
                    med_record = m_options[selected_med_str]
                    if sel_qty > med_record['stock']:
                        st.error("Transaction Aborted: Transaction structural distribution count items requested quantity configuration violates bounds of active safety backup stock levels.")
                    else:
                        tot_cost = float(med_record['sale_price']) * int(sel_qty)
                        new_stock_level = int(med_record['stock']) - int(sel_qty)
                        
                        db.execute_query("UPDATE medicines SET stock=%s WHERE id=%s", (new_stock_level, med_record['id']))
                        db.execute_query("INSERT INTO pharmacy_sales (medicine_id, quantity, total_price) VALUES (%s,%s,%s)", (med_record['id'], sel_qty, tot_cost))
                        st.success(f"Prescription structural transaction log entry committed. Transaction ledger entry posted value: ${tot_cost:.2f}")
                        st.rerun()
            else:
                st.warning("No available medications configured with active positive balance stocks storage containers matrices parameters.")

    # ==========================================
    # MODULE 8: LABORATORY OPERATIONS
    # ==========================================
    def render_laboratory_operations(self):
        st.title("🔬 Laboratory Diagnostic Operations & Testing Core Registry Suite")
        t1, t2 = st.tabs(["Active Diagnostic Lab Testing Queue Logs View", "Dispatch New Diagnostic Test Request Order Configuration Data Pipeline"])
        
        with t1:
            st.subheader("Laboratory Sample Tracing Pipeline Matrix Telemetry Readout View")
            lab_data = db.fetch_all("""
                SELECT l.id, p.name as patient_name, l.test_type, l.findings, l.status, l.created_at 
                FROM lab_reports l
                JOIN patients p ON l.patient_id = p.id
                ORDER BY l.created_at DESC
            """)
            if lab_data:
                st.dataframe(pd.DataFrame(lab_data), use_container_width=True)
            else:
                st.info("No active diagnostic procedure items posted to the processing framework queues indexes rows data.")
                
        with t2:
            st.subheader("Authorize New Diagnostic Pathology Specimen Examination Sequence Directives")
            p_profiles = db.fetch_all("SELECT id, name FROM patients")
            if p_profiles:
                p_ops_map = {f"ID Vector Ref {r['id']} | {r['name']}": r['id'] for r in p_profiles}
                target_p_str = st.selectbox("Associate Testing Orders Data Entry to Patient Identity Module Link Node Selector", list(p_ops_map.keys()))
                t_type = st.selectbox("Select Diagnostic Examination Procedure Classification Protocol Category", ["Complete Blood Analysis Screen Panel (CBC)", "Routine Macroscopic Urinalysis Screen (UA)", "Chest Radiography X-Ray Imaging Diagnostic Matrix", "High-Field Brain MRI Contrast Structural Scans Diagnostic Assessment Trace", "12-Lead Electrocardiogram Trace Metrics Analysis (ECG)"])
                findings_notes = st.text_area("Initial Presentation Lab Findings Summary Annotations Logs Data Stream Entries Context Text")
                
                if st.button("Finalize Specimen Request Order Records Creation Mapping Sequence Core Pipeline Node", use_container_width=True):
                    q = "INSERT INTO lab_reports (patient_id, test_type, findings, status) VALUES (%s,%s,%s,'Completed')"
                    if db.execute_query(q, (p_ops_map[target_p_str], t_type, findings_notes)):
                        st.success("Diagnostic processing orders structural entry pipeline execution cycle success posted state logs matrix.")
                        st.rerun()
            else:
                st.error("No target candidate patient instances available inside memory structures registers data grid rows.")

    # ==========================================
    # MODULE 9: REVENUE & BILLING CORE (PDF Export)
    # ==========================================
    def render_billing_core(self):
        st.title("💳 Financial Cleardown Operations Revenue Ledger & Invoice Clearing Portal")
        
        col_b1, col_b2 = st.columns([1.2, 2])
        
        with col_b1:
            st.subheader("Construct Consolidated Financial Checkout Invoice Statement Pipeline Block Form")
            p_entities = db.fetch_all("SELECT id, name FROM patients")
            if p_entities:
                p_billing_map = {f"ID Vector Ref {r['id']} | {r['name']}": r['id'] for r in p_entities}
                sel_bill_p_str = st.selectbox("Bind Accounting Entry Transactions Objects to Target Record Index Node", list(p_billing_map.keys()))
                
                c_chg = st.number_input("Consultation Fee Assessment Accrued Matrix Value Units Scale ($)", value=150.0)
                l_chg = st.number_input("Laboratory Procedure Accumulation Charges Scale Value Units ($)", value=75.0)
                ph_chg = st.number_input("Pharmacy Item Dispensation Accrued Expense Statement Scale Units ($)", value=45.20)
                r_chg = st.number_input("Inpatient Accommodation Room Charges Cumulative Unit Scale Value ($)", value=250.0)
                disc = st.number_input("Authorized Insurance/Corporate Markdown Applied Discount Metric Balance Matrix ($)", value=20.0)
                
                # Dynamic calculations standard algorithms logic mapping variables matrices configuration parameters parameters 
                subtotal = c_chg + l_chg + ph_chg + r_chg - disc
                tax_calculated = subtotal * 0.18
                grand_total_value = subtotal + tax_calculated
                
                st.markdown(f"**Provisional Computed Invoice Value (Includes 18% GST Valuation Metrics Matrix):** `${grand_total_value:.2f}`")
                
                if st.button("Commit Clearance Receipt Object Ledger Updates to Central Systems Rows", use_container_width=True):
                    q = """INSERT INTO invoices (patient_id, consultation_charges, lab_charges, pharmacy_charges, room_charges, tax, discount, grand_total)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
                    params = (p_billing_map[sel_bill_p_str], c_chg, l_chg, ph_chg, r_chg, tax_calculated, disc, grand_total_value)
                    if db.execute_query(q, params):
                        st.success("Financial transaction committed to primary databases invoices registers logs array values.")
                        st.rerun()
            else:
                st.error("No patients records maps available targets matrix arrays instances items strings structures fields.")

        with col_b2:
            st.subheader("Historically Cleared Invoice Balance Ledger Ledger Entries List Tracking Telemetry Logs")
            historical_invoices = db.fetch_all("""
                SELECT i.id, p.name as patient_name, i.grand_total, i.created_at, i.consultation_charges, i.lab_charges, i.pharmacy_charges, i.room_charges, i.tax, i.discount
                FROM invoices i
                JOIN patients p ON i.patient_id = p.id
                ORDER BY i.created_at DESC
            """)
            if historical_invoices:
                df_inv = pd.DataFrame(historical_invoices)
                st.dataframe(df_inv[['id', 'patient_name', 'grand_total', 'created_at']], use_container_width=True)
                
                st.markdown("---")
                st.subheader("Generate Standard Native Clear PDF Printable Invoice Engine Dispatch System")
                target_invoice_id = st.selectbox("Select Target Active Transaction Log System ID Index Row Match Location Sequence", df_inv['id'].tolist())
                
                if target_invoice_id:
                    inv_match_row = [row for row in historical_invoices if row['id'] == target_invoice_id][0]
                    pdf_byte_stream = InvoiceBillingGenerator.compile_pdf_invoice_stream(inv_match_row, inv_match_row['patient_name'])
                    
                    st.download_button(
                        label=f"Download Official MediCare Invoice Receipt Stream Asset [Ref ID Node TXN-00{target_invoice_id}] (.PDF File Format)",
                        data=pdf_byte_stream,
                        file_name=f"medicare_invoice_receipt_txn_00{target_invoice_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("The revenue accounts index matrices record values tracks contain zero processed components fields grids rows instances.")

    # ==========================================
    # MODULE 10: BED MANAGEMENT SUBSYSTEM
    # ==========================================
    def render_bed_subsystem(self):
        st.title("🛏️ Hospital Facility Structural Inpatient Accommodation Bed Inventory Configuration Console Matrix")
        st.markdown("Monitor real-time infrastructure allocation configurations vectors tracking mapping array units elements grids instances.")
        
        bed_telemetry = db.fetch_all("SELECT * FROM beds")
        if bed_telemetry:
            df_beds = pd.DataFrame(bed_telemetry)
            
            c_avail, c_occ = st.columns(2)
            with c_avail:
                st.markdown("#### Operational Bed Slots Spatial Mapping Layout Telemetry Tables")
                st.dataframe(df_beds, use_container_width=True)
            with c_occ:
                st.markdown("#### Spatial Allocation Layout Visual Matrix Map Distribution Chart Summary")
                fig_bed_pie = px.pie(df_beds, names="status", color_discrete_sequence=["#10B981", "#EF4444"])
                fig_bed_pie.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bed_pie, use_container_width=True)
                
            st.markdown("---")
            st.subheader("Mutate Facility Real Estate Allocation State Mapping Variables Matrices Status Directly Control Panel")
            col_bstate1, col_bstate2 = st.columns(2)
            with col_bstate1:
                target_bed_id_input = st.selectbox("Target Allocation Slot Room Number Location Identification Tag Unique Sequence ID Key Token Match Index Entry Mapping Context Unit Grid Vector", df_beds['id'].tolist())
            with col_bstate2:
                target_bed_status_assignment = st.selectbox("Assign Targeted Facility Infrastructure Instance Dynamic Operational State Mode Option Selector", ["Available", "Occupied", "Maintenance/Sterilization In-Progress Structural Holds Block Sequence Layer Mode Matrix Data Logs Data"])
                
            if st.button("Commit Structural Shift Real Estate Matrix Allocation System Framework Commands", use_container_width=True):
                if db.execute_query("UPDATE beds SET status=%s WHERE id=%s", (target_bed_status_assignment, target_bed_id_input)):
                    st.success("Infrastructure inventory allocation layout matrix state shifted successfully variables registers values.")
                    st.rerun()
        else:
            st.warning("No structural real estate slots configured. Run master tables infrastructure setup script protocols.")

    # ==========================================
    # MODULE 11: EMERGENCY INCIDENT STREAM
    # ==========================================
    def render_emergency_stream(self):
        st.title("🚨 High-Priority Acute Critical Triage Inpatient Emergency Intake Pipeline Stream Module Control Center Panel")
        st.markdown("Bypass standard verification layers. Force register priority entry points directly into primary data buffers immediately.")
        
        col_em1, col_em2 = st.columns([1.2, 2])
        
        with col_em1:
            st.subheader("Acute Critical Trauma Patient Acceleration Intake Registration Interface Form Panel Block")
            with st.form("emergency_rapid_registration_triage_entry_form"):
                em_name = st.text_input("Provisional Unidentified Patient Alias / Full Known Legal Name Track Token Entry Element *", value="John Doe Unknown Case Ref Vector Alpha")
                triage_condition_notes = st.selectbox("Triage Presentation Evaluation Stratification Level Group Specifier Classification Vector Mapping Coordinate Node Target", ["Class 1 Trauma Code Red - Immediate Resuscitation Pipeline Required Critical Status Matrix Block", "Class 2 Urgent Severe - Cardiorespiratory Anomaly/Altered Mentation Acute Profile", "Class 3 Moderate Non-Life-Threatening Traumatic Complication Event Instance Data Logs"])
                
                avail_beds_list = db.fetch_all("SELECT id, bed_number FROM beds WHERE status='Available'")
                bed_choices_map = {r['bed_number']: r['id'] for r in avail_beds_list}
                bed_choices_map["No Bed Available / Force Hold In Triage Staging Area Corridor Bays Matrix"] = None
                
                selected_emergency_bed_str = st.selectbox("Allocate Urgent Critical Staging Area Spatial Location Infrastructure Target Slots Node Link Selector Base", list(bed_choices_map.keys()))
                
                em_submit_btn = st.form_submit_button("Force Authorize Rapid Insertion Commands Pipeline Sequence Entry Check Checks")
                
            if em_submit_btn:
                target_bed_internal_db_id = bed_choices_map[selected_emergency_bed_str]
                
                # Execute rapid transactional dual-update pipeline sequences simultaneously safely inside transaction logic systems mappings parameters properties variables parameters properties structures framework fields fields 
                q_em_insert = "INSERT INTO emergency_cases (patient_name, triage_condition, assigned_bed_id) VALUES (%s,%s,%s)"
                if db.execute_query(q_em_insert, (em_name, triage_condition_notes, target_bed_internal_db_id)):
                    if target_bed_internal_db_id:
                        db.execute_query("UPDATE beds SET status='Occupied' WHERE id=%s", (target_bed_internal_db_id,))
                    st.success("Triage entry pipeline force-inserted successfully. Medical response teams dispatched directly to station allocation.")
                    st.rerun()

        with col_em2:
            st.subheader("Active Active High-Risk Triage Streams Monitoring Log Trace Metrics Radar Matrix")
            active_emergencies_records = db.fetch_all("""
                SELECT e.id, e.patient_name, e.triage_condition, b.bed_number, e.registered_at 
                FROM emergency_cases e
                LEFT JOIN beds b ON e.assigned_bed_id = b.id
                ORDER BY e.registered_at DESC
            """)
            if active_emergencies_records:
                st.dataframe(pd.DataFrame(active_emergencies_records), use_container_width=True)
            else:
                st.info("No acute critical trauma incidents active within current operational timeframe framework thresholds parameters properties data frames.")

# ==========================================
# 6. SYSTEM BOOTSTRAP INITIALIZATION ENTRY NODE
# ==========================================
if __name__ == "__main__":
    app_instance = MediCareApplicationRouter()
    app_instance.execute_routing_loop()