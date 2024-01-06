# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tpot import TPOTClassifier

# Load the dataset
@st.cache_resource
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/Ozodiy/Diabets-prediction/main/diabetes.csv')
    return data

data = load_data()

# Preprocessing
X = data.drop('Outcome', axis=1)  
y = data['Outcome']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "TPOT": TPOTClassifier(generations=5, population_size=20, verbosity=2)  
}

# Streamlit title
st.title('Diabetes Prediction App')

# Model selection
model_option = st.selectbox("Select the model for prediction", list(models.keys()))
model = models[model_option]

# Train and predict
if model_option != "TPOT":  # TPOT has its own fit/predict procedure
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
else:
    # TPOT fitting (this can be very slow, consider running separately or simplifying)
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)

# Display the accuracy
st.write(f"Accuracy of {model_option}: {accuracy:.2%}")

# Prediction section
st.subheader("Predict Your Diabetes Risk")

# Collecting user input for each feature
user_data = [st.slider(f"{feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean())) for feature in X.columns]

# Making a prediction
if st.button('Predict'):
    user_data_scaled = scaler.transform([user_data]) 
    prediction = model.predict(user_data_scaled)
    st.write(f"Predicted Class: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")

# Visualization section
st.subheader('Data Visualizations')

# Visualization 1: Correlation Heatmap
if st.checkbox('Show Correlation Heatmap'):
    st.write('Correlation Heatmap')
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

# Visualization 2: Pairplot 
if st.checkbox('Show Pairplot of Features'):
    st.write('Pairplot of Features')
    fig = sns.pairplot(data, hue='Outcome')
    st.pyplot(fig)

