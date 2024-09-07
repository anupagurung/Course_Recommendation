from flask import Flask, render_template, request, jsonify
import pandas as pd
from content_based_course_recommendation_by_using_similarity import get_recommendations

app = Flask(__name__)

# Load datasets
course_data = pd.read_csv('course_processed.csv')
genre_data = pd.read_csv('course_genre.csv')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend_courses():
    user_data = request.get_json()  # Get JSON data from the request
    course_title = user_data.get('course_title')  # Extract course_title
    
    if course_title:
        # Call the recommendation function
        recommended_courses = get_recommendations(course_title, course_data, genre_data)
        return jsonify(recommended_courses)
    else:
        return jsonify({"error": "No course title provided."}), 400

if __name__ == '__main__':
    app.run(debug=True)
