# -*- coding: utf-8 -*-
"""Extract Bag of Words (BoW) Features from Course Textual Content.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1v4PhfygEG4n17i64tG7wCJFSxgR3vo1_
"""

# Install the necessary packages
!pip install nltk==3.6.7
!pip install gensim==4.1.2
!pip install scipy==1.7.3

# Import libraries
import gensim
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

course_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv"
course_content_df = pd.read_csv(course_url)

course_content_df.iloc[0, :]

#join those two text columns together.
# Merge TITLE and DESCRIPTION title
course_content_df['course_texts'] = course_content_df[['TITLE', 'DESCRIPTION']].agg(' '.join, axis=1)
course_content_df = course_content_df.reset_index()
course_content_df['index'] = course_content_df.index

course_content_df.iloc[0, :]

#Tokenize the course content:
def tokenize_course(course, keep_only_nouns=True):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(course)
    # Remove English stop words and numbers
    word_tokens = [w for w in word_tokens if (not w.lower() in stop_words) and (not w.isnumeric())]
    # Only keep nouns
    if keep_only_nouns:
        filter_list = ['WDT', 'WP', 'WRB', 'FW', 'IN', 'JJR', 'JJS', 'MD', 'PDT', 'POS', 'PRP', 'RB', 'RBR', 'RBS',
                       'RP']
        tags = nltk.pos_tag(word_tokens)
        word_tokens = [word for word, pos in tags if pos not in filter_list]

    return word_tokens

#tokenize_course() method to tokenize all courses in course_content_df['course_texts'].
tokenized_courses = [tokenize_course(course_text) for course_text in course_content_df['course_texts']]
tokenized_courses[:1]

#Utokens_dict = gensim.corpora.Dictionary(tokenized_courses)
tokens_dict = gensim.corpora.Dictionary(tokenized_courses)
tokens_dict.token2id

#Use tokens_dict.doc2bow() to generate BoW features for each tokenized course
courses_bow = [tokens_dict.doc2bow(course) for course in tokenized_courses]
courses_bow[:1]

# Create a new course_bow dataframe based on the extracted BoW features.

course_content_df

doc_index = []
doc_id = []
bags_of_token = []
bow = []

for idx, bag in enumerate(courses_bow):
    for word in bag:
        token = tokens_dict[word[0]]
        doc_index.append(idx)
        doc_id.append(course_content_df['COURSE_ID'][idx])
        bags_of_token.append(token)
        bow.append(word[1])


bow_dicts = {"doc_index": doc_index,
           "doc_id": doc_id,
            "token": bags_of_token,
            "bow": bow}
pd.DataFrame(bow_dicts)