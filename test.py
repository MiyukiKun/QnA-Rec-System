import sys
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open('qna_queries.json', 'r', encoding='utf-8') as f:
    data = json.loads(f.read())

with open("QnA_Rec.model", 'rb') as f:
    model = pickle.load(f)

vectorizer = model[0]
tfidf_matrix = model[1]

params = sys.argv
query = params[1]
query_matrix = vectorizer.transform([query])
similarity_scores = cosine_similarity(query_matrix, tfidf_matrix)

top_indices = similarity_scores[0].argsort()[::-1][:10]

result = []
for i in top_indices:
    data[i]['similarity_score'] = similarity_scores[0][i]
    result.append(data[i])

weights = {'no_of_likes': 0.05, 'no_of_answers': 0.1, 'no_of_views': 0.001}    # + 1 if expert reply present

def sort_key(question):
    score = 0
    for param, weight in weights.items():
        score += question.get(param, 0) * weight
    if question.get('isExpertReplied'):
        score += 0.5

    score += question['similarity_score']
    return score

sorted_questions = sorted(result, key=sort_key, reverse=True)
sorted_questions = sorted_questions[:5]

for i in sorted_questions:
    print(f"Question Id : {i['_id']['$oid']} : {i['question']['en']}: Score : {sort_key(i)}")