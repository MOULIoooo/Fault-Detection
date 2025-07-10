import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
x = df.drop(['species'], axis=1)
y = df['species']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(max_depth=3, min_samples_split=4, random_state=42)
model.fit(x_train, y_train)

# Prediction and accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.set_page_config(page_title="Iris Predictor", page_icon="🌸")
st.title("🌸 Iris Flower Species Predictor")
st.write("Using Decision Tree Classifier")

# Show dataset
with st.expander("📊 See Raw Dataset"):
    st.dataframe(df)

# Show model accuracy
st.markdown(f"### ✅ Model Accuracy: `{accuracy:.2f}`")

# User input
st.sidebar.header("🔍 Enter Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Make prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=x.columns)
prediction = model.predict(input_data)[0]

st.markdown(f"### 🔮 Predicted Species: **{prediction}**")

# Plot decision tree
st.subheader("🌲 Decision Tree Structure")
fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(model, feature_names=x.columns, class_names=model.classes_, filled=True)
st.pyplot(fig)
