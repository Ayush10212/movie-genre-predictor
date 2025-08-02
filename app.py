from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('genre_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    plot = request.form['plot']
    if not plot:
        return render_template('index.html', prediction="Please enter a movie plot.")
    
    vec_plot = vectorizer.transform([plot])
    prediction = model.predict(vec_plot)[0]
    
    return render_template('index.html', prediction=f"Predicted Genre: {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
