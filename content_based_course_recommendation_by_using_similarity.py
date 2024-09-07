# Required imports
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
import nltk
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
bows_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/courses_bows.csv"
bows_df = pd.read_csv(bows_url)
bows_df = bows_df[['doc_id', 'token', 'bow']]

course_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv"
course_df = pd.read_csv(course_url)

# Function to get recommendations based on cosine similarity
def get_recommendations(course_title, course_data, genre_data, threshold=0.5):
    # Find the selected course
    target_course = course_data[course_data['COURSE_TITLE'] == course_title]

    # If the course is not found
    if target_course.empty:
        return []

    # Extract the BoW features for the target course
    target_bow = genre_data[genre_data['doc_id'] == target_course.iloc[0]['COURSE_ID']]

    # List to store recommendations
    recommendations = []

    # Iterate through all other courses
    for _, course in course_data.iterrows():
        if course['COURSE_TITLE'] != course_title:
            other_bow = genre_data[genre_data['doc_id'] == course['COURSE_ID']]

            # Ensure that other_bow has entries
            if not other_bow.empty:
                # Calculate cosine similarity
                similarity = 1 - cosine(target_bow['bow'], other_bow['bow'])
                if similarity >= threshold:
                    recommendations.append(course['COURSE_TITLE'])

    return recommendations
