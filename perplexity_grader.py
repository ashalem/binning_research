from typing import List, Tuple
import requests
import re

PERPLEXITY_TOKEN = "<FILL IN BASED ON INSTRUCTIONS IN SITE>"

def evaluate_grouping_perplexity(
    feature_name: str,
    groupings: List[Tuple[float, float]],
    feature_range: Tuple[float, float],
    perplexity_token: str
) -> tuple:
    """
    Evaluates a grouping using the Perplexity API and the requested prompt, returning
    the grade and explanation.

    Parameters:
        feature_name (str): The name of the feature.
        feature_range (Tuple[float, float]): The entire range of values for the feature (min, max).
        groupings (List[Tuple[float, float]]): The groupings to evaluate, provided as a list of (low, high) tuples.
        perplexity_token (str): Your Perplexity API token.

    Returns:
        tuple: A tuple containing:
            - grade (int): An integer (1, 2, 3, or 4) indicating how commonly the grouping is used.
            - explanation (str): The explanation or reasoning from the model's response.
    """

    # Convert the list of tuples into a readable string for the prompt
    grouping_str = ", ".join([f"{low}-{high}" if low != float('-inf') and high != float('inf') else f"lower than {high}" if low == float('-inf') else f"more than {low}" for (low, high) in groupings])

    # Define the new prompt
    prompt = f"""
ask context: You are tasked with evaluating the semantic value of a given grouping for a specified feature, by providing a metric of the number of times the provided grouping method is used with the stated feature.

You must also provide links to two of those references for validation purposes.

For a standardized metric you will use the following defined sources:

Google Scholar

PubMed

JSTOR

Office for National Statistics (ONS)

You will use this context to evaluate the grouping provided.

You will also add a level to this response based on how common the usage of a specific grouping is in the sources, picked from these 4 grades:

1. Not used at all
2. Has seen very few references to it
3. Rare but used
4. Very commonly used

Your reply template:
- Grade:
- Reference Count In Sources:
- Reference Example Links:
- Explanation:

Task:
Feature: {feature_name}
Range: {feature_range}
Grouping: {grouping_str}
    """

    # Build the request payload for Perplexity
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "Be precise and concise."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        # Optional parameters you can tune
        "max_tokens": 300,
        "temperature": 0.2,
        "top_p": 0.9,
        "search_domain_filter": [],         # e.g. ["perplexity.ai"] or leave empty
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",   # e.g. "month", "year", or "all"
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }

    headers = {
        "Authorization": f"Bearer {perplexity_token}",
        "Content-Type": "application/json"
    }

    # Send the request to the Perplexity API
    response = requests.post(url, json=payload, headers=headers)

    # Basic error handling for non-200 responses
    if response.status_code != 200:
        raise RuntimeError(
            f"Perplexity API returned status code {response.status_code}: {response.text}"
        )

    data = response.json()
    # The assistant's reply text (where we expect the grading format)
    try:
        model_response = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Unexpected response format from Perplexity: {data}"
        ) from e

    # Extract the return values from the model response
    # NOTE: This should be changed if changing the prompt format
    perplexity_pattern = re.compile(
    r"(?s)- \*\*Grade:\*\*\s*(\d+)(?:[^\r\n]*)?\s*?"
    r"- \*\*Reference Count In Sources:\*\*\s*([\s\S]*?)\s*?"
    r"- \*\*Reference Example Links:\*\*\s*([\s\S]*?)\s*?"
    r"- \*\*Explanation:\*\*\s*(.*)"
)
    match = perplexity_pattern.search(model_response)
    if match:
        grade = int(match.group(1))  
        reference_count = match.group(2).strip()
        reference_links = match.group(3).strip()
        explanation     = match.group(4).strip()
    else:
        print("No match found in:\n" + model_response)
        raise RuntimeError("No match found in the model response.")


    return grade, explanation, reference_count, reference_links


def main():
    """
    Main function to test the evaluate_grouping function with predefined examples.
    """
    # Predefined features and groupings
    test_cases = [
        ("Monthly Salary in Dollars", [(0, 1000), (1000, 10000), (11000, float('inf'))], (0, float('inf'))),
        ("Temperature in Celsius", [(-30, 0), (1, 15), (16, 30), (31, 50), (51, float('inf'))], (float('-inf'), float('inf'))),
        ("Temperature in Celsius", [(float('-inf'), 0), (1, 35), (36, 50), (51, float('inf'))], (float('-inf'), float('inf'))),
        ("Age in Years", [(0, 12), (13, 19), (20, 64), (65, 100)], (0,100)),
        ("Height in Centimeters", [(0, 100), (101, 150), (151, 200), (201, float('inf'))], (0, 230)),
        ("BMI", [(0, 18.5), (18.5, 24.9), (25, 29.9), (30, 50)], (0, 50)),
        ("BMI", [(0, 10), (11, 20), (21, 30), (31, 50)], (0, 50)),
        ("Temperature in Celsius", [(-50, -10), (-9, 0), (1, 30), (31, float('inf'))], (float('-inf'), float('inf'))),
        ("Temperature in Celsius", [(float('-inf'), 0), (0, 10), (10, 25), (25, float('inf'))], (float('-inf'), float('inf'))),
        ("Age in Years", [(0, 18), (19, 35), (36, 50), (51, 70), (71, 100)], (0,100)),
        ("Age in Years", [(0, 1), (2, 5), (6, 12), (13, 17), (18, 24), (25, 100)], (0,100))
    ]

    # Iterate through test cases and print results
    for feature, grouping, feature_range in test_cases:
        try:
            grade, explanation, reference_count, reference_links = evaluate_grouping_perplexity(feature, grouping, feature_range, PERPLEXITY_TOKEN)
            print(f"Feature: {feature}, Grouping: {grouping}\nGrade: {grade}, Explanation: {explanation}\nReference Count: {reference_count}\nReference Links: {reference_links}\n")
        except RuntimeError as e:
            print(f"Error evaluating feature '{feature}' with grouping '{grouping}': {e}\n")

main()