import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from flask import Flask, request, jsonify

cred = credentials.Certificate('data/ecpeazy-firebase.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

CACHE_SIZE = 128

app = Flask('ECPeazy Recommender')

@lru_cache(maxsize=CACHE_SIZE)
def fetch_extracurriculars():
    extracurriculars_ref = db.collection('ECs')
    extracurriculars = []
    for doc in extracurriculars_ref.stream():
        extracurriculars.append(doc.to_dict())
    return extracurriculars

@lru_cache(maxsize=CACHE_SIZE)
def fetch_user_portfolio(user_id):
    portfolio_ref = db.collection('Portfolios').document(user_id)
    portfolio = portfolio_ref.get()
    if portfolio.exists:
        return portfolio.to_dict()
    else:
        return None

def preprocess_portfolio(portfolio, extracurriculars):
    activities = portfolio.get('activities', [])
    combined_text = []
    for activity in activities:
        matching_ecs = [ec for ec in extracurriculars if ec['name'] == activity['activity']]
        description = activity['description']
        subjects = ' '.join(matching_ecs[0]['subjects']) if matching_ecs else ''
        difficulty = ' '.join(matching_ecs[0]['skill_levels']) if matching_ecs else ''
        combined_text.append(f"{description} {subjects} {difficulty}")
    
    return ' '.join(combined_text)

def preprocess_extracurriculars(extracurriculars):
    descriptions = []
    for ec in extracurriculars:
        desc = f"{ec['name']} {ec['description']} {' '.join(ec['subjects'])} {' '.join(ec['skill_levels'])} {' '.join(ec['org_types'])} {' '.join(ec['locations'])} {' '.join(ec['grades'])} {ec['categories']}"
        descriptions.append(desc)
    return descriptions

def recommend_extracurriculars(user_id):
    portfolio = fetch_user_portfolio(user_id)
    if not portfolio:
        print("No portfolio found for the user.")
        return []

    extracurriculars = fetch_extracurriculars()
    
    user_activities = set(activity['activity'] for activity in portfolio.get('activities', []))

    filtered_extracurriculars = [ec for ec in extracurriculars if ec['name'] not in user_activities]

    portfolio_text = preprocess_portfolio(portfolio, filtered_extracurriculars)
    extracurricular_texts = preprocess_extracurriculars(filtered_extracurriculars)

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([portfolio_text] + extracurricular_texts)

    similarity_matrix = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    top_indices = similarity_matrix.argsort()[-5:][::-1]
    recommendations = [extracurriculars[i] for i in top_indices]

    return recommendations

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    recommendations = recommend_extracurriculars(user_id)
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)