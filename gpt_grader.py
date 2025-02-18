from typing import List, Tuple
from openai import OpenAI
import re

client = OpenAI(api_key='sk-proj-b8ymyvv75LlzBOisUYH8T3BlbkFJPeh2KB99bpIZ4WihDJQs')

def evaluate_groupings_gpt(
    tasks: List[Tuple[str, List[Tuple[float, float]], Tuple[float, float]]],
) -> List[tuple]:
    """
    Evaluates multiple groupings using GPT-4 API, returning grades and explanations for each.

    Parameters:
        tasks (List[Tuple]): List of tasks, where each task is a tuple containing:
            - feature_name (str): The name of the feature
            - groupings (List[Tuple[float, float]]): The groupings to evaluate
            - feature_range (Tuple[float, float]): The entire range of values

    Returns:
        List[tuple]: List of tuples, each containing:
            - grade (int): Integer (1-4) indicating how commonly the grouping is used
            - explanation (str): The explanation from the model
            - reference_count (str): Count of references found
            - reference_links (str): Example reference links
    """

    # Build the prompt for all tasks
    full_prompt = "Evaluate multiple feature groupings for their semantic value and usage frequency.\n\n"
    
    for i, (feature_name, groupings, feature_range) in enumerate(tasks, 1):
        grouping_str = ", ".join([
            f"{low}-{high}" if low != float('-inf') and high != float('inf')
            else f"lower than {high}" if low == float('-inf')
            else f"more than {low}" for (low, high) in groupings
        ])
        
        full_prompt += f"""Task {i}:
Feature: {feature_name}
Range: {feature_range}
Grouping: {grouping_str}

"""

    system_prompt = """You are tasked with evaluating the semantic value of given groupings for specified features. For each task:

1. Evaluate how commonly the grouping method is used with the stated feature
2. Search academic and statistical sources
3. Provide a grade from:
   1 = Not used at all
   2 = Very few references
   3 = Rare but used
   4 = Very commonly used
4. Provide reference counts and example links

Format your response for each task as:
Task N:
- Grade: [1-4]
- Reference Count In Sources: [count]
- Reference Example Links: [links]
- Explanation: [brief explanation]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        model_response = response.choices[0].message.content.strip()
        
        # Parse the response for each task
        results = []
        task_pattern = re.compile(
            r"Task \d+:\s*"
            r"- Grade:\s*(\d+)(?:[^\r\n]*)?\s*"
            r"- Reference Count In Sources:\s*([\s\S]*?)\s*"
            r"- Reference Example Links:\s*([\s\S]*?)\s*"
            r"- Explanation:\s*((?:(?!Task \d+:)[\s\S])*)",
            re.MULTILINE
        )
        
        matches = task_pattern.finditer(model_response)
        for match in matches:
            grade = int(match.group(1))
            reference_count = match.group(2).strip()
            reference_links = match.group(3).strip()
            explanation = match.group(4).strip()
            results.append((grade, explanation, reference_count, reference_links))
        
        if not results:
            raise RuntimeError("No valid results found in the model response.")
        
        return results

    except Exception as e:
        raise RuntimeError(f"Error in GPT API call: {str(e)}")

def main():
    """
    Main function to test the evaluate_groupings function with predefined examples.
    """
    # Test multiple groupings for the same feature in a single call
    test_cases = [
        # Multiple groupings for Monthly Salary
        [
            ("Monthly Salary in Dollars", [(0, 1000), (1000, 10000), (11000, float('inf'))], (0, float('inf'))),
            ("Monthly Salary in Dollars", [(0, 2000), (2000, 5000), (5000, 10000), (10000, float('inf'))], (0, float('inf'))),
            ("Monthly Salary in Dollars", [(0, 3000), (3000, 6000), (6000, 9000), (9000, float('inf'))], (0, float('inf')))
        ],
        
        # Multiple groupings for Temperature
        [
            ("Temperature in Celsius", [(-30, 0), (1, 15), (16, 30), (31, 50)], (float('-inf'), float('inf'))),
            ("Temperature in Celsius", [(-50, 0), (0, 25), (25, 50)], (float('-inf'), float('inf'))),
            ("Temperature in Celsius", [(float('-inf'), 0), (0, 20), (20, 40), (40, float('inf'))], (float('-inf'), float('inf')))
        ],
        
        # Multiple groupings for Age
        [
            ("Age in Years", [(0, 12), (13, 19), (20, 64), (65, 100)], (0, 100)),
            ("Age in Years", [(0, 18), (19, 35), (36, 50), (51, 65), (66, 100)], (0, 100)),
            ("Age in Years", [(0, 21), (22, 40), (41, 60), (61, 80), (81, 100)], (0, 100))
        ]
    ]

    try:
        # Test each group of related test cases
        for test_group in test_cases:
            print(f"\nTesting multiple groupings for {test_group[0][0]}:")
            print("-" * 80)
            
            # Evaluate all test cases in the group in one API call
            results = evaluate_groupings_gpt(test_group)
            
            # Print results for each test case in the group
            for (feature, grouping, feature_range), (grade, explanation, reference_count, reference_links) in zip(test_group, results):
                print(f"\nFeature: {feature}")
                print(f"Grouping: {grouping}")
                print(f"Feature Range: {feature_range}")
                print(f"Grade: {grade}")
                print(f"Explanation: {explanation}")
                print(f"Reference Count: {reference_count}")
                print(f"Reference Links: {reference_links}")
                print("-" * 80)
            
    except RuntimeError as e:
        print(f"Error evaluating groupings: {e}")

if __name__ == "__main__":
    main() 