import numpy as np
import pandas as pd

# Load datasets
True_news = pd.read_csv(r"C:\Users\diaca\Desktop\plasmid internship final\True.csv")
Fake_news = pd.read_csv(r"C:\Users\diaca\Desktop\plasmid internship final\Fake.csv")

# Add labels
True_news["label"] = 0
Fake_news["label"] = 1

# Select relevant columns
dataset1 = True_news[["text", "label"]]
dataset2 = Fake_news[["text", "label"]]

# Concatenate datasets
dataset = pd.concat([dataset1, dataset2])

# Check dataset shape
print(dataset.shape)

# Check for null values
print(dataset.isnull().sum())

# Check label distribution
print(dataset["label"].value_counts())

# Shuffle the dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

# NLP Libraries
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stopwords
ps = WordNetLemmatizer()
stopwords = stopwords.words("english")

# Download necessary NLTK data
nltk.download("wordnet")

# Text cleaning function
def clean_row(row):
    row = row.lower()
    row = re.sub("[^a-zA-Z]", " ", row)  # Corrected regex to keep only alphabetic characters
    tokens = row.split()
    news = [ps.lemmatize(word) for word in tokens if word not in stopwords]
    cleaned_news = " ".join(news)
    return cleaned_news

# Apply text cleaning
dataset["text"] = dataset["text"].apply(lambda x: clean_row(x))

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))

x = dataset["text"]
y = dataset["label"]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=0)

# Vectorize the text data
vec_train_data = vectorizer.fit_transform(train_data).toarray()
vec_test_data = vectorizer.transform(test_data).toarray()

# Convert the vectorized data into DataFrames
train_data_df = pd.DataFrame(vec_train_data, columns=vectorizer.get_feature_names_out())
test_data_df = pd.DataFrame(vec_test_data, columns=vectorizer.get_feature_names_out())

# Model training
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()  # Corrected line 38
clf.fit(train_data_df, train_label)

# Model prediction
y_pred = clf.predict(test_data_df)

# Calculate accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_label, y_pred)
print(f"Model Accuracy: {accuracy}")

# Test the model with a new input
txt = input("Enter News: ")
news = clean_row(txt)
pred = clf.predict(vectorizer.transform([news]).toarray())

if pred == 0:
    print("News is correct")
else:
    print("News is Fake")
