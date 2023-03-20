import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
#pip install scikit-learn pour l'installation au lieu de sklearn
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import re
from nltk.corpus import stopwords
import string

#data import and data check
data = pd.read_csv("jobs.csv")
        #dataset cleaned from https://statso.io/jobs-dataset/
print(data.head())
print(data.columns)

#drop the unnamed column
data = data.drop("Unnamed: 0",axis=1)

#check for null values
print(data.isnull().sum())

#create a skills wordcloud
text1 = " ".join(i for i in data["Key Skills"])
stopwords = set(STOPWORDS)
#STOPWORDS built in list of words to ignore in Wordcloud
#ex : "the", "a", "an", "in"
#we can add custom stopwords with STOPWORDS.update(["newword1", "newword2"])
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text1)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("skills.png")  #adding a save
#plt.show()

#create a functional areas wordcloud
text2 = " ".join(i for i in data["Functional Area"])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text2)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("funct_areas.png") #adding a save
#plt.show()

#create a job title wordcloud
text3 = " ".join(i for i in data["Job Title"])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text3)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("job_titles.png")  # adding a save
#plt.show()

#create a similarity matrix from the key skills column
from sklearn.feature_extraction import text
feature = data["Key Skills"].tolist()
tfidf = text.TfidfVectorizer(input='content', stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
#fit_transform learn vocabulary and idf, and return document-term matrix
#print(type(tfidf_matrix))
similarity = cosine_similarity(tfidf_matrix)

#using the job title column as index
indices = pd.Series(data.index, index=data['Job Title']).drop_duplicates()

#recommendation function
def jobs_recommendation(Title, similarity = similarity):
    index = indices[Title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[::], reverse=True)
    similarity_scores = similarity_scores[0:5]
    newsindices = [i[0] for i in similarity_scores]
    return data[['Job Title', 'Job Experience Required', 
                 'Key Skills']].iloc[newsindices]

#testing for a recommendation
print(jobs_recommendation("Software Developer"))