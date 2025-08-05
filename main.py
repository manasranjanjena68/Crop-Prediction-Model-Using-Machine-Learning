from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key
app.config['SESSION_PERMANENT'] = True  # Keep session active

# Dummy user credentials (Replace with database authentication)
USER_CREDENTIALS = {
    "manas": "manas123"
}

# Load dataset and train model
dataset = pd.read_csv("Crop_recommendation.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route('/')
def home():
    if 'user' not in session:  # Check if user is logged in
        return redirect(url_for('login'))  
    return render_template('index.html')  # Render crop prediction page

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            session['user'] = username  # Store user in session
            session.permanent = True  # Keep session active
            return redirect(url_for('home'))  
        else:
            return render_template('login.html', error="Invalid Username or Password")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove user from session
    return redirect(url_for('login'))  

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:  
        return jsonify({'error': 'Unauthorized access, please log in'}), 401  

    data = request.get_json()
    features = np.array([
        data['N'], data['P'], data['K'], 
        data['temperature'], data['humidity'], 
        data['ph'], data['rainfall']
    ]).reshape(1, -1)

    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
