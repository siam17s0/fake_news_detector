# fake_news_detector
Machine Learning based Fake News Detection System using NLP

Live Demo:https://fake-news-detector-ml-app.streamlit.app

## Overview
This project is a Machine Learning-based Fake News Detection system that classifies news articles as Real or Fake using Natural Language Processing (NLP) techniques.
The model analyzes the textual content of news and predicts its authenticity with high accuracy.

## Models used
- Logistic Regression
- Naive Bayes
- Linear Support Vector Classifier (Linear SVC)
- Random Forest

## Dataset used:https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

## Features
- Text preprocessing (lowercasing, remove url,remove html, remove punctuation, stopword removal, lemmatization)
- TF-IDF vectorization
- Model training and evaluation

## Results
| Model               | Accuracy   | Precision  | Recall     | F1         |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 98.94%     | 0.985      | 0.992      | 0.988      |
| SVM                 | 99.49%     | **0.9925** | 0.9969     | 0.9947     |
| Linear SVC          | 99.44%     | 0.9922     | 0.9959     | 0.9941     |
| **Random Forest**   | **99.51%** | 0.9922     | **0.9974** | **0.9948** |

### Winner: Random Forest (slightly)
Highest accuracy (99.51%),
Highest recall (99.74%),
Best F1 score (very slightly)

### “Although Random Forest achieved the highest accuracy, I selected Linear SVC as the final model because it generalizes better on high-dimensional sparse text data.”

## How It Works
1.Input news text from user
2.Preprocess the text (cleaning + lemmatization)
3.Convert text into numerical form using TF-IDF
4.Pass it into trained ML model
5.Output prediction: Real / Fake

## How to Run Locally
Step 1: Clone the repository
git clone https://github.com/siam17s0/fake_news_detector.git
cd fake-news-detector-ml
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Run the app
streamlit run app.py

## Example news(prediction: real news)
The government has announced a new education reform policy that will be implemented nationwide next year. Officials said the plan aims to improve the quality of education and provide better training for teachers.

