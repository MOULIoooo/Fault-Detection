import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ✅ Set page config
st.set_page_config(page_title="Fault Detection", page_icon="☠️", layout="centered")

# ✅ Add styling
page_bg_img = '''
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://energytheory.com/wp-content/uploads/2023/04/JAN23-What-is-HT-Line.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}

/* White box center */
.block-container {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 1rem;
    margin-top: 6vh;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}

/* Input labels */
label, .stNumberInput label {
    font-weight: 700;
    color: #000000;
}

/* Input styling */
input[type="number"] {
    background-color: #f0f7ff !important;
    border-radius: 6px;
    color: black;
}

/* Hide + - buttons */
input::-webkit-outer-spin-button,
input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ✅ App header
st.markdown("<h1 style='text-align: center;'>⚡ Fault Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Know  Fault Type</h4>", unsafe_allow_html=True)

# ✅ Input Section
with st.container():
    st.subheader("Enter Voltage and Current Readings")
    col1, col2 = st.columns(2)
    with col1:
        f1 = st.number_input("Va (Volts)", value=0.0, format="%.4f")
        f2 = st.number_input("Vb (Volts)", value=0.0, format="%.4f")
        f3 = st.number_input("Vc (Volts)", value=0.0, format="%.4f")
    with col2:
        f4 = st.number_input("Ia (Amps)", value=0.0, format="%.4f")
        f5 = st.number_input("Ib (Amps)", value=0.0, format="%.4f")
        f6 = st.number_input("Ic (Amps)", value=0.0, format="%.4f")

# ✅ Train model only once — CACHED!
@st.cache_data(show_spinner=False)
def load_model():
    df = pd.read_csv("https://raw.githubusercontent.com/MOULIoooo/Fault-Detection/main/fault_detection.csv")
    df['fault_type'] = df[['G', 'C', 'B', 'A']].apply(lambda row: ''.join(row.astype(str)), axis=1)
    df = df.drop(['G', 'C', 'B', 'A'], axis=1)
    fault_map = {
        '0000': 'NO Fault',
        '1001': 'LG Fault',
        '0110': 'LL Fault',
        '1011': 'LLG Fault',
        '0111': 'LLL Fault',
        '1111': 'LLLG Fault',
    }
    df['fault_type'] = df['fault_type'].map(fault_map)
    le = LabelEncoder()
    df['fault_type'] = le.fit_transform(df['fault_type'])

    samples = [df[df.fault_type == i].sample(n=1000, random_state=42) for i in range(6)]
    df = pd.concat(samples)

    x = df.iloc[:, 0:6]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# ✅ Load cached model & accuracy
rfmodel, ac = load_model()

# ✅ Accuracy above button
left, _, _ = st.columns([1, 1, 1])
with left:
    st.markdown(f"<h5 style='color: green;'>🎯 Accuracy: {ac:.2f}</h5>", unsafe_allow_html=True)

# ✅ Predict Button in center
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🚀 Predict Fault Status"):
        input_data = np.array([[f4, f5, f6, f1, f2, f3]])  # Order as in dataset
        fault_mapping = {
            0: "NO Fault",
            1: "LL Fault",
            2: "LLL Fault",
            3: "LG Fault",
            4: "LLG Fault",
            5: "LLLG Fault"
        }
        pred = rfmodel.predict(input_data)
        prediction = fault_mapping.get(pred[0], "Unknown Fault")
        st.success(f"✅ **Predicted Fault Type:** {prediction}")
