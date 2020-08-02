from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('movie_plot.csv')

genre_mapper = {'other': 0, 'action': 1, 'adventure': 2, 'comedy':3, 'drama':4, 'horror':5, 'romance':6, 'sci-fi':7, 'thriller': 8}
df['genre'] = df['genre'].map(genre_mapper)

df.drop('id',axis = 1,inplace = True)

import re
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,df.shape[0]):
    print(i)
    line = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df['text'][i])
    line = line.lower()
    line = line.split()
    ps = PorterStemmer()
    line = [ps.stem(words) for words in line if not words in set(stopwords.words('english'))]
    line = ' '.join(line)
    corpus.append(line)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=10000,ngram_range=(1,2))
X = cv.fit_transform(corpus).toarray()
y = df['genre'].values

pickle.dump(cv, open('cv-transform.pkl', 'wb'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train,y_train)

filename = 'movie-genre-mnb-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))