import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    }
    .hero h1 { font-size: 2rem; font-weight: 700; color: white; margin: 0; }
    .hero p  { color: #a5b4fc; font-size: 0.95rem; margin-top: 0.4rem; }
    .hero-badge {
        display: inline-block; background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2); border-radius: 99px;
        padding: 3px 12px; font-size: 0.75rem; color: #c7d2fe; margin-top: 0.5rem;
    }

    .section-title {
        font-size: 0.7rem; font-weight: 700; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin: 1.4rem 0 0.5rem 0; padding-bottom: 0.4rem;
        border-bottom: 1px solid #f3f4f6;
    }

    .result-low {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 2px solid #10b981; border-radius: 16px;
        padding: 1.8rem; text-align: center;
    }
    .result-high {
        background: linear-gradient(135deg, #fff1f2, #ffe4e6);
        border: 2px solid #f43f5e; border-radius: 16px;
        padding: 1.8rem; text-align: center;
    }
    .result-icon  { font-size: 2.8rem; margin-bottom: 0.4rem; }
    .result-label { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-label-low  { color: #059669; }
    .result-label-high { color: #e11d48; }
    .result-prob { font-size: 1rem; color: #374151; margin-bottom: 0.8rem; }
    .result-note { font-size: 0.82rem; color: #6b7280; font-style: italic; }

    .gauge-track {
        background: #e5e7eb; border-radius: 99px;
        height: 12px; overflow: hidden; margin: 0.6rem 0 0.3rem 0;
    }
    .gauge-fill-low  { background: linear-gradient(90deg,#34d399,#10b981); border-radius:99px; height:100%; }
    .gauge-fill-high { background: linear-gradient(90deg,#fb923c,#f43f5e); border-radius:99px; height:100%; }
    .gauge-labels { display:flex; justify-content:space-between; font-size:0.7rem; color:#9ca3af; }

    .pill-wrap { display:flex; flex-wrap:wrap; gap:7px; margin-top:0.4rem; }
    .pill-risk { background:#fee2e2; color:#b91c1c; border-radius:99px; padding:4px 12px; font-size:0.78rem; font-weight:500; }
    .pill-safe { background:#d1fae5; color:#065f46; border-radius:99px; padding:4px 12px; font-size:0.78rem; font-weight:500; }

    .info-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .disclaimer {
        font-size: 0.78rem; color: #9ca3af; text-align: center;
        margin-top: 1.5rem; padding: 0.8rem;
        background: #f9fafb; border-radius: 8px;
        border: 1px solid #f3f4f6;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white; border: none; border-radius: 10px;
        padding: 0.65rem 2rem; font-size: 1rem;
        font-weight: 600; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants — Cleveland dataset (13 fitur)
# ─────────────────────────────────────────────
FEATURE_COLS  = ['age','sex','cp','trestbps','chol','fbs',
                 'restecg','thalach','exang','oldpeak','slope','ca','thal']
COLS_TO_SCALE = ['age','trestbps','chol','thalach','oldpeak']

# ─────────────────────────────────────────────
# Load model dari pkl, atau retrain dari UCI
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path  = 'knn_heart_model.pkl'
    scaler_path = 'scaler.pkl'

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path,  'rb') as f: model  = pickle.load(f)
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        return model, scaler, None

    # Retrain dari UCI jika pkl tidak ada
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    COL_NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, names=COL_NAMES, na_values='?')
    df['target'] = (df['target'] > 0).astype(int)
    df['ca']   = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    df['ca']   = df['ca'].astype(int)
    df['thal'] = df['thal'].astype(int)

    X = df[FEATURE_COLS]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr = X_train.copy(); X_te = X_test.copy()
    X_tr[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_te[COLS_TO_SCALE] = scaler.transform(X_test[COLS_TO_SCALE])

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_tr, y_train)

    model = KNeighborsClassifier(n_neighbors=11)
    model.fit(X_sm, y_sm)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:,1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc':  roc_auc_score(y_test, y_proba),
        'f1':       f1_score(y_test, y_pred),
    }
    with open(model_path,  'wb') as f: pickle.dump(model,  f)
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
    return model, scaler, metrics

with st.spinner("🫀 Memuat model..."):
    model, scaler, metrics = load_model()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Model Info")
    st.markdown("""
    **Algoritma:** K-Nearest Neighbors
    **Dataset:** UCI Cleveland Heart Disease
    **Pasien:** 303 — data klinis nyata
    **Fitur:** 13 parameter klinis
    **Preprocessing:** StandardScaler + SMOTE
    """)

    if metrics:
        st.markdown("---")
        st.markdown("### 📊 Performa Model")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
        c2.metric("ROC-AUC",   f"{metrics['roc_auc']:.3f}")
        st.metric("F1-Score",  f"{metrics['f1']:.3f}")
    else:
        st.markdown("---")
        st.markdown("### 📊 Performa Model")
        st.success("Model loaded dari file pkl")

    st.markdown("---")
    st.markdown("### 💡 Tips Pengisian")
    st.markdown("""
    **Tidak punya hasil lab?**
    Isi dengan nilai rata-rata normal:
    - Tekanan darah: 120
    - Kolesterol: 200
    - Detak jantung maks: 150
    - ST Depression: 0

    **Belum pernah ke dokter jantung?**
    Pilih opsi pertama untuk bagian
    Hasil Pemeriksaan Dokter.

    **Hasil lebih akurat** jika diisi
    berdasarkan hasil cek kesehatan
    yang sebenarnya.
    """)

    st.markdown("---")
    st.caption("Final Project — Data Science with Pandas  \nBINUS University 2026")

# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🫀 Cek Risiko Penyakit Jantung</h1>
    <p>Isi data kesehatan kamu di bawah ini — kami akan bantu perkirakan risiko penyakit jantungmu
    berdasarkan data dari 303 pasien nyata menggunakan Machine Learning.</p>
    <span class="hero-badge">⏱️ Hanya butuh 2 menit</span>
    <span class="hero-badge" style="margin-left:8px">🔒 Data tidak disimpan</span>
    <span class="hero-badge" style="margin-left:8px">🤖 Akurasi 73.8%</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
with st.form("input_form"):

    st.markdown('<div class="section-title">👤 Data Diri</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    age = c1.number_input("Berapa usia kamu?", 20, 80, 54)
    sex = c2.selectbox("Jenis kelamin", ["Pria", "Wanita"])

    st.markdown('<div class="section-title">💊 Hasil Cek Kesehatan (dari dokter/laboratorium)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    trestbps = c1.number_input("Tekanan darah (angka atas, mmHg)", 90, 200, 130,
                                help="Contoh: kalau tekanan darahmu 130/85, isi 130")
    chol     = c2.number_input("Kolesterol (mg/dL)", 100, 600, 200,
                                help="Didapat dari hasil cek darah di lab. Normal: di bawah 200 mg/dL")
    thalach  = c3.number_input("Detak jantung maks saat olahraga (bpm)", 70, 210, 150,
                                help="Biasanya diukur saat tes treadmill di klinik")

    c1, c2 = st.columns(2)
    fbs = c1.selectbox("Gula darah puasa kamu tinggi?",
                        ["Tidak — di bawah 120 mg/dL", "Ya — di atas 120 mg/dL"],
                        help="Cek dari hasil lab gula darah puasa")
    oldpeak = c2.number_input("Nilai ST Depression (dari hasil EKG)", 0.0, 7.0, 1.0, step=0.1,
                               help="Ada di lembar hasil EKG dari dokter. Kalau tidak tahu, isi 0")

    st.markdown('<div class="section-title">💓 Gejala yang Dirasakan</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    chest_pain = c1.selectbox("Nyeri dada yang sering dirasakan?", [
        "Tidak ada nyeri dada",
        "Nyeri dada saat aktivitas berat, hilang saat istirahat",
        "Nyeri dada tapi tidak khas (tidak selalu saat aktivitas)",
        "Tidak terasa sakit apapun, tapi hasil EKG ada kelainan ⚠️"
    ], help="Pilih yang paling sesuai kondisi kamu")
    exang = c2.selectbox("Saat olahraga atau aktivitas berat, dada terasa sakit/sesak?",
                          ["Tidak", "Ya"],
                          help="Contoh: naik tangga, jalan cepat, lari ringan")

    st.markdown('<div class="section-title">🏥 Hasil Pemeriksaan Dokter</div>', unsafe_allow_html=True)

    st.info("💡 Bagian ini diisi dari hasil pemeriksaan dokter spesialis jantung. "
            "Kalau belum pernah periksa, pilih opsi pertama di setiap pertanyaan.")

    c1, c2, c3 = st.columns(3)
    restecg = c1.selectbox("Hasil rekam jantung (EKG) saat istirahat", [
        "Normal",
        "Ada kelainan ringan (ST-T abnormal)",
        "Jantung membesar (LV Hypertrophy)"
    ])
    slope = c2.selectbox("Pola grafik EKG saat tes olahraga", [
        "Naik (Upsloping) — normal",
        "Datar (Flat) — perlu perhatian",
        "Turun (Downsloping) — perlu waspada ⚠️"
    ])
    ca = c3.selectbox("Berapa pembuluh jantung yang tersumbat? (dari hasil kateterisasi)",
                       ["0 — Tidak ada", "1 — Satu pembuluh", "2 — Dua pembuluh", "3 — Tiga pembuluh"],
                       help="Didapat dari pemeriksaan kateterisasi jantung / fluoroskopi")

    thal = st.selectbox("Hasil tes Thalassemia jantung", [
        "Normal — tidak ada kelainan",
        "Fixed Defect — ada area jantung yang tidak menerima aliran darah (permanen)",
        "Reversible Defect — ada area yang kurang aliran darah saat stres, tapi membaik saat istirahat ⚠️"
    ], help="Didapat dari pemeriksaan nuclear stress test / thallium scan")

    submitted = st.form_submit_button("🫀 Cek Risiko Jantungku")

# ─────────────────────────────────────────────
# Prediksi
# ─────────────────────────────────────────────
if submitted:
    # Parse input
    sex_val = 1 if sex == "Pria" else 0

    cp_map = {
        "Tidak ada nyeri dada": 0,
        "Nyeri dada saat aktivitas berat, hilang saat istirahat": 1,
        "Nyeri dada tapi tidak khas (tidak selalu saat aktivitas)": 2,
        "Tidak terasa sakit apapun, tapi hasil EKG ada kelainan ⚠️": 3,
    }
    cp_val = cp_map[chest_pain]

    fbs_val   = 1 if "Ya" in fbs else 0
    exang_val = 1 if exang == "Ya" else 0

    restecg_map = {
        "Normal": 0,
        "Ada kelainan ringan (ST-T abnormal)": 1,
        "Jantung membesar (LV Hypertrophy)": 2,
    }
    restecg_val = restecg_map[restecg]

    slope_map = {
        "Naik (Upsloping) — normal": 0,
        "Datar (Flat) — perlu perhatian": 1,
        "Turun (Downsloping) — perlu waspada ⚠️": 2,
    }
    slope_val = slope_map[slope]

    ca_val = int(ca[0])

    thal_map = {
        "Normal — tidak ada kelainan": 1,
        "Fixed Defect — ada area jantung yang tidak menerima aliran darah (permanen)": 2,
        "Reversible Defect — ada area yang kurang aliran darah saat stres, tapi membaik saat istirahat ⚠️": 3,
    }
    thal_val = thal_map[thal]

    patient = pd.DataFrame([{
        'age': age, 'sex': sex_val, 'cp': cp_val,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs_val,
        'restecg': restecg_val, 'thalach': thalach, 'exang': exang_val,
        'oldpeak': oldpeak, 'slope': slope_val, 'ca': ca_val, 'thal': thal_val
    }])

    patient_scaled = patient.copy()
    patient_scaled[COLS_TO_SCALE] = scaler.transform(patient[COLS_TO_SCALE])

    proba = model.predict_proba(patient_scaled)[0][1]
    pred  = int(proba >= 0.5)
    pct   = round(proba * 100, 1)

    st.markdown("---")
    left, right = st.columns([1.2, 1])

    # ── Hasil prediksi
    with left:
        if pred == 0:
            st.markdown(f"""
            <div class="result-low">
                <div class="result-icon">✅</div>
                <div class="result-label result-label-low">Risiko Rendah</div>
                <div class="result-prob">Probabilitas penyakit jantung: <strong>{pct}%</strong></div>
                <div class="gauge-track">
                    <div class="gauge-fill-low" style="width:{pct}%"></div>
                </div>
                <div class="gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>
                <br>
                <div class="result-note">
                    Profil pasien ini menunjukkan risiko rendah berdasarkan 13 parameter klinis.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-high">
                <div class="result-icon">⚠️</div>
                <div class="result-label result-label-high">Risiko Tinggi</div>
                <div class="result-prob">Probabilitas penyakit jantung: <strong>{pct}%</strong></div>
                <div class="gauge-track">
                    <div class="gauge-fill-high" style="width:{pct}%"></div>
                </div>
                <div class="gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>
                <br>
                <div class="result-note">
                    Profil ini menunjukkan risiko tinggi. Segera konsultasikan dengan dokter spesialis jantung.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Faktor kontribusi
        st.markdown('<div class="section-title">🔍 Analisis Faktor Risiko</div>', unsafe_allow_html=True)

        risk_factors, safe_factors = [], []

        # Risk factors berdasarkan bukti klinis dari dataset
        if cp_val == 3:
            risk_factors.append("⚠️ Nyeri dada Asymptomatic (risiko tertinggi)")
        elif cp_val == 2:
            risk_factors.append("⚠️ Non-Anginal Pain")
        if thalach < 140:
            risk_factors.append(f"💓 Max HR rendah ({thalach} bpm)")
        if oldpeak >= 2.0:
            risk_factors.append(f"📉 ST Depression tinggi ({oldpeak})")
        if exang_val == 1:
            risk_factors.append("🏃 Angina saat olahraga")
        if ca_val >= 1:
            risk_factors.append(f"🩸 {ca_val} pembuluh tersumbat")
        if thal_val == 3:
            risk_factors.append("🫀 Reversible Defect (Thal)")
        elif thal_val == 2:
            risk_factors.append("🫀 Fixed Defect (Thal)")
        if slope_val == 2:
            risk_factors.append("📊 ST Downsloping")
        if sex_val == 1 and age >= 55:
            risk_factors.append(f"👤 Pria usia {age} tahun")
        if fbs_val == 1:
            risk_factors.append("🩸 Gula darah puasa tinggi")

        # Safe factors
        if cp_val == 0:
            safe_factors.append("✅ Typical Angina (pola normal)")
        if thalach >= 160:
            safe_factors.append(f"💪 Max HR baik ({thalach} bpm)")
        if oldpeak < 1.0:
            safe_factors.append(f"✅ ST Depression minimal ({oldpeak})")
        if exang_val == 0:
            safe_factors.append("✅ Tidak ada angina saat olahraga")
        if ca_val == 0:
            safe_factors.append("✅ Tidak ada pembuluh tersumbat")
        if thal_val == 1:
            safe_factors.append("✅ Thalassemia Normal")
        if chol < 200:
            safe_factors.append(f"✅ Kolesterol normal ({chol})")
        if fbs_val == 0:
            safe_factors.append("✅ Gula darah puasa normal")

        col_r, col_s = st.columns(2)
        with col_r:
            st.markdown("**🔴 Faktor Risiko**")
            if risk_factors:
                pills = "".join(f'<span class="pill-risk">{f}</span>' for f in risk_factors)
                st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)
            else:
                st.success("Tidak ada faktor risiko signifikan")

        with col_s:
            st.markdown("**🟢 Faktor Protektif**")
            if safe_factors:
                pills = "".join(f'<span class="pill-safe">{f}</span>' for f in safe_factors)
                st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)
            else:
                st.info("Tidak ada faktor protektif teridentifikasi")

    # ── Ringkasan pasien
    with right:
        st.markdown('<div class="section-title">📋 Ringkasan Data Pasien</div>', unsafe_allow_html=True)

        cp_labels   = {0:"Tidak ada", 1:"Nyeri saat aktivitas", 2:"Tidak khas", 3:"Tanpa gejala ⚠️"}
        thal_labels = {1:"Normal", 2:"Ada sumbatan permanen", 3:"Kurang aliran darah ⚠️"}
        slope_labels= {0:"Naik (Normal)", 1:"Datar", 2:"Turun ⚠️"}
        restecg_lbl = {0:"Normal", 1:"Kelainan ringan", 2:"Jantung membesar"}

        summary = [
            ("👤 Usia / Kelamin",       f"{age} tahun / {sex}"),
            ("💉 Tekanan Darah",        f"{trestbps} mmHg"),
            ("🧪 Kolesterol",           f"{chol} mg/dL"),
            ("💓 Detak Jantung Maks",   f"{thalach} bpm"),
            ("📉 ST Depression",        f"{oldpeak}"),
            ("🫀 Nyeri Dada",           cp_labels[cp_val]),
            ("📊 Pola EKG Olahraga",    slope_labels[slope_val]),
            ("🩸 Pembuluh Tersumbat",   f"{ca_val} pembuluh"),
            ("🔬 Kondisi Jantung",      thal_labels[thal_val]),
            ("🏃 Sesak saat Olahraga", "Ya" if exang_val else "Tidak"),
            ("🍬 Gula Darah Puasa",    "Tinggi" if fbs_val else "Normal"),
        ]

        for label, value in summary:
            st.markdown(f"""
            <div class="info-card">
                <span style="color:#6b7280">{label}</span><br>
                <strong>{value}</strong>
            </div>
            """, unsafe_allow_html=True)

    # ── Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚕️ Alat ini bersifat <strong>edukatif</strong> dan tidak menggantikan diagnosis medis profesional.<br>
    Selalu konsultasikan hasil ini dengan dokter atau tenaga medis yang berwenang.
    </div>
    """, unsafe_allow_html=True)
