# Q&A Recommendation System

## Data available
    {
        "id": { "$oid": "63673f2efc801e51390a870d" },
        "parentId": "6366c5c7d50293000e400a2e",
        "profileVisibility": "public",
        "queryVisibility": "all",
        "text": "Query for notification",
        "tags": [],
        "isImageAvailable": false,
        "no_of_likes": 4,
        "no_of_answers": 3,
        "no_of_views": 168,
        "isExpertReplied": true,
        "isReported": false,
        "userReportCount": 0,
        "postedAt": { "$date": "2022-11-06T04:59:26Z" },
        "lastUpdatedDtm": { "$date": "2023-02-21T13:14:48Z" },
        "imageList": [],
        "_v": 0,
        "expertId": "63594c4ff84b26000e171646",
        "expertReplyId": { "$oid": "6367673ee6514a42d26f6427" },
    }

## Insights in the data
- With the data available the relevant fields to make a recommendation system are:
    - Text of the question.
    - The number of likes.
    - If the answers has expert reply. 
    - The number of answers.
    - The number of Views.

- These Data point can be used to make a search-based recommendation model that takes in user queries and returns a list of recommended questions based on their relevance to the query
- Techniques such as TF-IDF (term frequency-inverse document frequency) can be used to calculate the relevance score of each question based on its keywords.
- To classify questions based on their keywords, you can use natural language processing techniques such as tokenization, stopword removal, and stemming to extract important words from the question text. This can help identify the main topics or themes of the question.
- This comes under Content-Based Filtering Recommendation systems


## Implementation Plan
- Code for preprocessing the questions and search queries 
    - Tokenization
    - Lemmatizing
    
    Both can be achieved by NLTK Natural Language Toolkit, a python library

- TF-IDF vectorization method, which computes a weight for each word in the question based on its frequency and relevance to the overall corpus of questions.

- TF-IDF takes in input query and compares it with the overall corpus and returns top 10 questions with thier similarity score. Then the responses are sorted based on external factors such as likes, views, answers and expert replies

- Python file takes query input with cli input, so it can be run directly from any backend using subprocess calls and output can be taken from stdoutput

## Run the code
- `train.py` file needs to be run to train the model once every week or day, depending on how frequently the questions data is updated.
- `test.py` file neeeds to be run with query parameter for output of question ids.