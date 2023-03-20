import json, pickle
from sklearn.feature_extraction.text import TfidfVectorizer

with open('qna_queries.json', 'r', encoding='utf-8') as f:
    data = json.loads(f.read())

questions = []
for i in data:
    questions.append(i['question']['en'])

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)
model = (vectorizer, tfidf_matrix)
with open("QnA_Rec.model", "wb") as f:
    pickle.dump(model, f)