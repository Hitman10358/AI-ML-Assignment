#Step 1: Scraping News Articles


import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_cricket_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    articles = []
    article_elements = soup.find_all('div', class_='news-card')

    for article in article_elements:
        # Extract article title, content, and section (if available)
        title = article.find('a', class_='headingfour').get_text() if article.find('a', class_='headingfour') else None
        content = article.find('div', class_='para-txt').get_text() if article.find('div', class_='para-txt') else None
        section = article.find('a', class_='category').get_text() if article.find('a', class_='category') else None

        # Check if the article is related to cricket
        if section and 'cricket' in section.lower():
            articles.append({'title': title, 'content': content, 'section': section})

    return articles

# URL of Hindustan Times website
url_to_scrape = 'https://www.hindustantimes.com/'

# Scrape cricket-related articles
cricket_articles = scrape_cricket_news(url_to_scrape)

# Convert to DataFrame and save as CSV
df = pd.DataFrame(cricket_articles)
df.to_csv('cricket_articles.csv', index=False)



#Step 2: Text Classification

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the cricket-related articles scraped earlier
data = pd.read_csv('cricket_articles.csv')

# Prepare the data for classification
X = data['content']  # Using article content as features
y = data['section']  # Assuming 'section' contains the target labels (e.g., 'cricket')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vect, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_vect)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Classification report
print(classification_report(y_test, y_pred))

# Save classification report to a CSV file
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('classification_report.csv')
