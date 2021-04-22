"""
Title: NLP Classifier of Pennsylvania Legislature
Author: Angelina Wang
Date: April 21 2021
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#First, we need to import our cleaned PA bills dataset.

PA_bills_cleaned = pd.read_csv("https://github.com/choudhs/NLP-on-US-Legislature/blob/main/PA_bills_cleaned.csv?raw=True")
PA_bills_cleaned = PA_bills_cleaned.drop(columns=["Unnamed: 0"])
PA_bills_cleaned

#Now, let's standardize the text by tokenizing it, converting it all to lowercase, and lemmatizing it.

import en_core_web_sm
import nltk
from nltk.tokenize import RegexpTokenizer

def standardize_text(df, text_field):
  # step 1 - remove irrelevant characters
  df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")

  # step 2 - tokenize our text (lowercasing it, removing the punctuation and splitting on spaces)
  tokenizer = RegexpTokenizer("[\w']+")
  tokenizer.tokenize(text_field)

  # step 3 - convert all characters to lowercase
  df[text_field] = df[text_field].str.lower()

  # step 4 - lemmatize which is the process of reducing words to their basic stem
  nlp = en_core_web_sm.load()
  lemmas = nlp(text_field)
  lemmas = [t.lemma_ for t in lemmas if (t.is_alpha and not (t.is_stop or t.like_num))]
  lemmas = " ".join(lemmas)
  return df

PA_bills_cleaned = standardize_text(PA_bills_cleaned, "title")
PA_bills_cleaned

#Let's remove the stop words of common English words from the titles and find the most frequently used words in the bill titles.

cleaned_titles = PA_bills_cleaned['title']

nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stops =  set(stopwords.words('english') + ['com']) # set stop words to be common English words
co = CountVectorizer(stop_words=stops) # initialize count vectorizer
counts = co.fit_transform(cleaned_titles)

top_freq_words = pd.DataFrame(counts.sum(axis=0), columns=co.get_feature_names()).T.sort_values(by=0, ascending=False).head(20)
top_freq_words = top_freq_words.rename(columns={0: "Top 20 Most Frequent Words in Titles"})
top_freq_words

#After this initial EDA, we have noticed that a lot of these most common words are not very useful, such as "PA" (stands for Pennsylvania) and "effective". Let's filter out these words manually and see what we get.

pd.DataFrame({"Top 5 Filtered Most Frequent Words in Titles":
[top_freq_words.loc["omnibus"],
top_freq_words.loc["vehicle"],
top_freq_words.loc["crimes"],
top_freq_words.loc["judicial"],
top_freq_words.loc["property"]]})

#Now, we can see that after filtering, the top 5 most frequent words in bill titles are omnibus, crimes, judicial, enactment, and vehicle. We can keep this information and these skills in mind as we proceed with building our classifier model.

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_titles_tfidf = tfidf_transformer.fit_transform(counts)
X_titles_tfidf.shape
#Here, we have found the TF-IDF (Term Frequency times inverse document frequency) of our training data.

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_titles_tfidf, cleaned_titles)
#The above code will train the Naive Bayes (NB) classifier on the training data we provided.

predicted = clf.predict(counts)
np.mean(predicted == cleaned_titles)

#Testing the model on our data, we see our NB model was able to correctly predict the title 64.63% of the time. In order to improve this, let's change the fit_prior parameter from True to False. When set to false for MultinomialNB, a uniform prior will be used.

new_clf = MultinomialNB(fit_prior=False).fit(X_titles_tfidf, cleaned_titles)
new_predicted = new_clf.predict(counts)
np.mean(new_predicted == cleaned_titles)

#After this improvement, our NB Model correctly predicts the title of a bill 79.27% of the time, which is a significant improvement. For future improvements, we could explore building a Support Vector Machines (SVM) model or use Grid Search to obtain optimal performance.
