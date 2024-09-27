import json
import openai
import os
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient


def load_courses(file_path: str) -> List[Dict]:
    """
    Load course data from a JSON file.

    :param file_path: Path to the JSON file containing course data.
    :return: List of courses as dictionaries.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            courses = json.load(f)
        print(f"Loaded {len(courses)} courses from {file_path}.")
        return courses
    except Exception as e:
        print(f"Error loading courses: {e}")
        return []


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text for better understanding.

    :param text: Original text.
    :return: Cleaned text.
    """
    # Remove excessive whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def prepare_course_embeddings(courses: List[Dict]) -> np.ndarray:
    """
    Generate embeddings for each course based on title and description.

    :param courses: List of course dictionaries.
    :return: Numpy array of embeddings.
    """
    texts = []
    for course in courses:
        title = preprocess_text(course.get('title', ''))
        description = preprocess_text(course.get('description', ''))
        combined_text = f"{title}. {description}"
        texts.append(combined_text)

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def generate_llm_prompt(user_query: str, courses_subset: List[Dict]) -> str:
    """
    Create a prompt for the LLM to process the user query against the courses.

    :param user_query: The user's search query.
    :param courses_subset: A subset of courses to consider for recommendations.
    :return: A string prompt for the LLM.
    """
    formatted_courses = prepare_course_data(courses_subset)
    prompt = f"""
You are an intelligent assistant specialized in recommending educational courses.

**User Query:** "{user_query}"

**Available Courses:**
{formatted_courses}

**Task:**
Based on the user's query, provide a list of the top 5 most relevant courses from the above list. For each recommended course, include:
- **Title**
- **Reason for Recommendation:** A brief explanation of why this course is relevant to the user's query.

**Response Format:**
Please respond in JSON format as a list of objects, each containing "title" and "reason".

**Example:**
[
  {{
    "title": "Introduction to Data Science",
    "reason": "Provides foundational knowledge in data science, aligning with the user's interest in data analysis."
  }},
  ...
]
"""
    return prompt


def prepare_course_data(courses_subset: List[Dict]) -> str:
    """
    Format course data into a structured string for the LLM prompt.

    :param courses_subset: List of course dictionaries.
    :return: Formatted string representing the subset of courses.
    """
    formatted_courses = ""
    for idx, course in enumerate(courses_subset, 1):
        title = preprocess_text(course.get('title', 'No Title'))
        description = preprocess_text(course.get('description', 'No Description'))
        formatted_courses += f"{idx}. **{title}**\n{description}\n\n"
    return formatted_courses


def call_openai_api(prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """
    Call the OpenAI API with the given prompt.

    :param prompt: The prompt to send to the LLM.
    :param max_tokens: Maximum number of tokens in the response.
    :param temperature: Sampling temperature.
    :return: The assistant's response as a string.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that recommends relevant courses based on user queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None
        )
        assistant_reply = response.choices[0].message.content.strip()
        return assistant_reply
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}")
        return ""


def parse_llm_response(response: str) -> List[Dict]:
    """
    Parse the LLM's JSON response into a Python list.

    :param response: The raw response string from the LLM.
    :return: List of recommended courses with titles and reasons.
    """
    try:
        # Extract JSON content from the response
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        json_str = response[json_start:json_end]
        recommendations = json.loads(json_str)
        return recommendations
    except json.JSONDecodeError:
        print("Failed to parse JSON from the LLM response.")
        print("LLM Response:")
        print(response)
        return []
    except Exception as e:
        print(f"Unexpected error during parsing: {e}")
        return []


def smart_search(user_query: str, courses: List[Dict], course_embeddings: np.ndarray, shortlist_limit: int = 50) -> \
List[Dict]:
    """
    Perform a smart search to find relevant courses based on the user's query using semantic similarity.

    :param user_query: The user's search input.
    :param courses: List of all available courses.
    :param course_embeddings: Precomputed embeddings for all courses.
    :param shortlist_limit: Number of courses to shortlist for the LLM.
    :return: List of recommended courses with titles and reasons.
    """
    if not user_query:
        print("Empty query provided.")
        return []

    # Preprocessing the user query
    clean_query = preprocess_text(user_query)

    # Generate embedding for the user query
    query_embedding = model.encode([clean_query], convert_to_numpy=True)

    # Compute cosine similarity between query and all courses
    similarities = cosine_similarity(query_embedding, course_embeddings)[0]

    # Get indices of courses sorted by similarity (highest first)
    sorted_indices = np.argsort(similarities)[::-1]

    # Shortlist top N courses based on similarity
    shortlisted_indices = sorted_indices[:shortlist_limit]
    keyword_matched_courses = [courses[idx] for idx in shortlisted_indices]

    # Generate the LLM prompt
    prompt = generate_llm_prompt(clean_query, keyword_matched_courses)

    # Call the OpenAI API
    llm_response = call_openai_api(prompt)

    if not llm_response:
        print("No response from LLM.")
        return []

    # Parse the LLM's JSON response
    recommendations = parse_llm_response(llm_response)

    return recommendations


def main():
    # Load course data
    courses = load_courses('all_courses.json')

    if not courses:
        print("No courses available to search.")
        return

    # Prepare course embeddings
    print("Generating course embeddings...")
    course_embeddings = prepare_course_embeddings(courses)
    print("Course embeddings generated.")

    while True:
        # Get user input
        user_query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
        if user_query.lower() == 'exit':
            print("Exiting Smart Search. Goodbye!")
            break

        # Perform smart search
        print("Searching for relevant courses...")
        recommendations = smart_search(user_query, courses, course_embeddings)

        # Display recommendations
        if recommendations:
            print("\n**Recommended Courses:**")
            for idx, course in enumerate(recommendations, 1):
                title = course.get('title', 'No Title')
                reason = course.get('reason', 'No Reason Provided')
                print(f"{idx}. **{title}**\n   *{reason}*\n")
        else:
            print("No relevant courses found for your query.")


if __name__ == "__main__":
    main()
