import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample dataset
data = {
    'plot': [
        "A spy goes on a secret mission to save the world from terrorists.",
        "Two best friends fall in love while attending college.",
        "A haunted house terrorizes a family with supernatural forces.",
        "A team of superheroes unite to fight against alien invaders.",
        "A man learns to cook after a painful breakup and finds joy.",
        "A detective tries to solve a complex murder mystery.",
        "Aliens attack Earth and one man tries to stop them.",
        "A woman finds love and healing on a road trip across Italy.",
        "The story of a war veteran struggling with PTSD.",
        "A hilarious adventure of a group of misfits in high school."
    ],
    'genre': [
        'Action', 'Romance', 'Horror', 'Action', 'Romance',
        'Thriller', 'Action', 'Romance', 'Drama', 'Comedy'
    ]
}

df = pd.DataFrame(data)

# Preprocessing
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['plot'])
y = df['genre']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, 'genre_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved.")
