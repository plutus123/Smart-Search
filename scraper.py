import json
from openai import OpenAI
import re
from dotenv import load_dotenv
from typing import List, Dict


load_dotenv()

client = OpenAI()
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
    Clean and preprocess text for better LLM understanding.

    :param text: Original text.
    :return: Cleaned text.
    """
    # Remove excessive whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def prepare_course_data(courses: List[Dict]) -> str:
    """
    Format course data into a structured string for the LLM prompt.

    :param courses: List of course dictionaries.
    :return: Formatted string representing all courses.
    """
    formatted_courses = ""
    for idx, course in enumerate(courses, 1):
        title = preprocess_text(course.get('title', 'No Title'))
        description = preprocess_text(course.get('description', 'No Description'))
        formatted_courses += f"{idx}. **{title}**\n{description}\n\n"
    return formatted_courses

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
                {"role": "system", "content": "You are a helpful assistant that recommends relevant courses based on user queries."},
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

def smart_search(user_query: str, courses: List[Dict], shortlist_limit: int = 50) -> List[Dict]:
    """
    Perform a smart search to find relevant courses based on the user's query.

    :param user_query: The user's search input.
    :param courses: List of all available courses.
    :param shortlist_limit: Number of courses to shortlist for the LLM.
    :return: List of recommended courses with titles and reasons.
    """
    if not user_query:
        print("Empty query provided.")
        return []

    # Preprocessing the user query
    clean_query = preprocess_text(user_query)

    # Define stopwords
    stopwords = {'the', 'is', 'at', 'which', 'on', 'for', 'and', 'a', 'an', 'of', 'to', 'in', 'with', 'best', 'course'}

    # Tokenize query and remove stopwords
    query_tokens = [word for word in re.findall(r'\w+', clean_query.lower()) if word not in stopwords]

    if not query_tokens:
        print("No meaningful keywords in query after removing stopwords. Considering all courses.")
        keyword_matched_courses = courses[:shortlist_limit]
    else:
        keyword_matched_courses = []
        for course in courses:
            title = course.get('title', '').lower()
            description = course.get('description', '').lower()
            for token in query_tokens:
                if token in title or token in description:
                    keyword_matched_courses.append(course)
                    break  # Avoid duplicate adds
            if len(keyword_matched_courses) >= shortlist_limit:
                break

    # If no matches found via keyword, consider all courses or use more advanced filtering
    if not keyword_matched_courses:
        print("No keyword matches found. Considering all courses.")
        keyword_matched_courses = courses[:shortlist_limit]

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

    while True:
        # Get user input
        user_query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
        if user_query.lower() == 'exit':
            print("Exiting Smart Search. Goodbye!")
            break

        # Perform smart search
        print("Searching for relevant courses...")
        recommendations = smart_search(user_query, courses)

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
