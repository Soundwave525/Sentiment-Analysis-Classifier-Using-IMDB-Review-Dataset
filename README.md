## 📌 Project Overview

This project builds a sentiment analysis classifier that automatically determines whether a movie review is **positive or negative** using Natural Language Processing (NLP) and Machine Learning techniques.

---

## 🚀 Features Implemented

### Text Preprocessing

* Lowercasing text
* Punctuation removal
* Tokenization (NLTK)
* Stopword removal
* Lemmatization

### Feature Extraction

1. **TF-IDF Vectorization**

   * Unigrams and bigrams (n-grams)
   * Captures word importance and word sequences

2. **Word2Vec Embeddings**

   * Dense semantic vector representations
   * Captures contextual relationships between words

---

## 🤖 Machine Learning Models Used

* Logistic Regression
* Random Forest Classifier
* Linear Support Vector Machine (Linear SVM)
* Multinomial Naive Bayes

---

## 📊 Evaluation Metrics

Models are evaluated using:

* Accuracy score
* Precision, Recall, F1-score
* Confusion Matrix
* Classification Report

---

## 📈 Key Observations

* TF-IDF with n-grams generally performs best for classical NLP sentiment tasks.
* Linear models such as Logistic Regression and SVM handle sparse text data effectively.
* Word2Vec embeddings capture semantics but may not outperform TF-IDF with traditional classifiers.

---

## 📂 Dataset

IMDB Movie Reviews Dataset:

* 50,000 labeled movie reviews
* Balanced positive and negative samples

**Note:** Dataset is not included in this repository due to size.
Download from: https://www.kaggle.com/datasets/pawankumargunjan/imdb-review

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install nltk scikit-learn gensim numpy pandas
```

Download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## ▶️ How to Run

1. Download and extract IMDB dataset.
2. Place it in project folder:

```
aclImdb/
   train/
   test/
```

3. Run the Python script:

```bash
python sentiment_analysis.py
```

---
