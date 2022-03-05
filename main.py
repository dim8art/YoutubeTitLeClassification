import random
import math
import numpy as np
import pandas as pd
import time
import re
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from wordcloud import WordCloud

DATA = pd.read_csv("Youtube Video Dataset.csv")
DATA = DATA.drop(["Videourl", "Description"],axis=1)
DATA["Category"]=DATA["Category"].map({"travel blog":0,"Science&Technology":1,"Food":2,"Art&Music":3,"manufacturing":4,"History":5})

word_density = dict()
DATA["Title"] = DATA["Title"].str.lower()
DATA["Title"] = DATA["Title"].str.replace(r'[^a-z ]', '')

data_letters = DATA.copy()
data_letters["Title"] = data_letters["Title"].apply(lambda a: [a.count(chr(i)) for i in range(ord('a'), ord('z')+1)])
data_density_letters = DATA.copy()
data_density_letters["Title"] = data_density_letters["Title"].apply((lambda a: [a.count(chr(i))/len(a)-a.count(' ') for i in range(ord('a'), ord('z')+1)]))
word_counter = dict()
for title in DATA["Title"]:
    for word in title.split(' '):
        if word in word_counter.keys():
            word_counter[word] += 1
        else:
            word_counter[word] = 1
top_words = sorted(word_counter, key=word_counter.get)[-101:-1]
top_words.reverse()

wordcloud=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(top_words))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

data_words = DATA.copy()
data_words["Title"] = data_words["Title"].apply(lambda a: [a.count(i) for i in top_words])
data_density_words = DATA.copy()
data_density_words["Title"] = data_density_words["Title"].apply(lambda a: [a.count(i)/len(a.split(' ')) for i in top_words])

letter_classifier = RandomForestClassifier()
density_letter_classifier = RandomForestClassifier()
words_classifier = RandomForestClassifier()
density_words_classifier = RandomForestClassifier()

l_sample = data_letters.sample(n=2000, axis="rows")
l_sample_x = l_sample["Title"].to_list()
l_sample_y = l_sample["Category"].to_list()

dl_sample = data_density_letters.sample(n=2000, axis="rows")
dl_sample_x = dl_sample["Title"].to_list()
dl_sample_y = dl_sample["Category"].to_list()

w_sample = data_words.sample(n=2000, axis="rows")
w_sample_x = w_sample["Title"].to_list()
w_sample_y = w_sample["Category"].to_list()

dw_sample = data_density_words.sample(n=2000, axis="rows")
dw_sample_x = dw_sample["Title"].to_list()
dw_sample_y = dw_sample["Category"].to_list()


letter_classifier.fit(l_sample_x, l_sample_y)
density_letter_classifier.fit(dl_sample_x, dl_sample_y)
words_classifier.fit(w_sample_x, w_sample_y)
density_words_classifier.fit(dw_sample_x, dw_sample_y)

predict = letter_classifier.predict(data_letters["Title"].to_list())
matrix = confusion_matrix(data_letters["Category"], predict)
sn.set(font_scale=1)
sn.heatmap(matrix, annot=True, fmt = "d")

plt.title("Accuracy " + str(100*letter_classifier.score(data_letters["Title"].to_list(), data_letters["Category"].to_list()))[:5]+"%")
plt.suptitle("Value of letters classifier")
plt.show()


predict = density_letter_classifier.predict(data_density_letters["Title"].to_list())
matrix = confusion_matrix(data_density_letters["Category"], predict)
sn.set(font_scale=1)
sn.heatmap(matrix, annot=True, fmt = "d")

plt.title("Accuracy " + str(100*density_letter_classifier.score(data_density_letters["Title"].to_list(), data_density_letters["Category"].to_list()))[:5]+"%")
plt.suptitle("Density of letters classifier")
plt.show()


predict = words_classifier.predict(data_words["Title"].to_list())
matrix = confusion_matrix(data_words["Category"], predict)
print(words_classifier.score(data_words["Title"].to_list(), data_words["Category"].to_list()))
sn.set(font_scale=1)
print(matrix)
sn.heatmap(matrix, annot=True, fmt = "d")

plt.title("Accuracy " + str(100*letter_classifier.score(data_letters["Title"].to_list(), data_letters["Category"].to_list()))[:5]+"%")
plt.suptitle("Value of words classifier")
plt.show()


predict = density_words_classifier.predict(data_density_words["Title"].to_list())
matrix = confusion_matrix(data_density_words["Category"], predict)
print(density_words_classifier.score(data_density_words["Title"].to_list(), data_density_words["Category"].to_list()))
sn.set(font_scale=1)
print(matrix)
sn.heatmap(matrix, annot=True, fmt = "d")

plt.title("Accuracy " + str(100*density_words_classifier.score(data_density_words["Title"].to_list(), data_density_words["Category"].to_list()))[:5]+"%")
plt.suptitle("Density of words classifier")
plt.show()
