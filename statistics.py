import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from scipy.stats import kendalltau, pearsonr
from scipy.spatial.distance import euclidean
from statsmodels.stats.inter_rater import aggregate_raters
import krippendorff
from scipy.stats import spearmanr



def calculate_krippendorffs_alpha(df, question_columns):
    """
    Calculates Krippendorff's Alpha for the given DataFrame.

    Parameters:
    - df: pandas DataFrame containing the ratings.
    - question_columns: List of column names corresponding to questions.

    Returns:
    - Dictionary of Krippendorff's Alpha values for each question.
    """
    # Placeholder reshaping (modify as per your data)
    # df_transposed = df[question_columns].transpose()
    ratings_matrix = df[question_columns].values
    
    
    # Check for missing values
    if np.isnan(ratings_matrix).any():
        print("Warning: Missing values detected. They will be handled by Krippendorff's Alpha.")

    alpha = krippendorff.alpha(reliability_data=ratings_matrix, level_of_measurement='ordinal')
    return alpha

# Function to calculate Tau-a
def kendall_tau_a(x, y):
    n = len(x)
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] and y[i] == y[j]:
                continue  # Tie in both
            elif x[i] == x[j]:
                ties_x +=1
            elif y[i] == y[j]:
                ties_y +=1
            else:
                concordant += (x[i] - x[j]) * (y[i] - y[j]) > 0
                discordant += (x[i] - x[j]) * (y[i] - y[j]) < 0
    tau_a = (concordant - discordant) / (0.5 * n * (n - 1))
    return tau_a

def main():
    # Replace 'GS_pre_responses.csv' with the path to your CSV file
    csv_file = './GS_responses.csv'
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # remoce the first column
    df = df.iloc[:, 1:]
    
    # Assume the first column is 'ID' and the rest are question columns
    id_column = df.columns[0]
    question_columns = df.columns[1:]
    
    # Number of categories (1, 2, 3, 4)
    categories = [1,2,3,4]
    
    # Convert all question columns to integers
    df[question_columns] = df[question_columns].astype(int)
    
    # Check for invalid entries
    # invalid_entries = ~df[question_columns].isin(categories).all(axis=None)
    # if invalid_entries:
    #     print("Error: Found responses outside the expected range (1-4). Please clean your data.")
    #     return
    
    # Convert the df to a numpy array with the ID column removed
    # Ensure all responses are integers
    data = df[question_columns].astype(int).values
    
    # Adjust category numbering from 1-4 to 0-3
    # fleiss_data = (data.copy().T - 1)
    
    # Use the aggregate_raters function to get the Fleiss' Kappa matrix
    # fleiss_matrix = aggregate_raters(fleiss_data)
    # Instead of using aggregate_raters, compute counts manually
    fleiss_matrix = []
    for question_idx, question_responses in enumerate(data.T):
        # Count the number of responses in each category (1, 2, 3, 4)
        counts = [np.sum(question_responses == category) for category in categories]
        fleiss_matrix.append(counts)
    

    # Calculate Fleiss' Kappa
    kappa = fleiss_kappa(fleiss_matrix, method='fleiss')
    print(f"Fleiss' Kappa: {kappa:.4f}")
    
    # Calculate Krippendorff's Alpha
    alpha = calculate_krippendorffs_alpha(df, question_columns)
    print(f"Krippendorff's Alpha: {alpha:.4f}")
    
    # Calculate quiz_vector: average response for each question
    quiz_vector = df[question_columns].mean().values
    print(f"Quiz Vector (Average Responses):\n{quiz_vector}")
    
    # Define constant vectors
    # Replace these with your actual rankings
    # Ensure that the length matches the number of question columns
    # PRE TASK: gpt_ranking = np.array([3,4,2,4,1])
    # PRE TASK: perplexity_ranking = np.array([3,4,2,4,1])
    
    # real task:
    gpt_ranking = np.array([
    4,  # Blood Pressure: “Less than 120, 120 to 129, 130 to 139, 140 or higher”
    2,  # Blood Pressure: “Less than 120, 110 or higher”
    4,  # BMI: “Less than 18.5, 18.5 to 24.9, 25 to 29.9, 30 or higher”
    1,  # BMI: “Less than 5, 5 to 15, higher than 15”
    4,  # AQI: “0–50, 51–100, 101–150, 151–200, 201–300, greater than 300”
    2,  # AQI: “0–60, 61–120, 121–180, 181–240, 241–300, 301–500”
    2,  # AQI: “0–40, 41–80, 81–120, 121–300, 301–500”
    3,  # AQI: “0–75, 76–150, 151–225, 226–300, 301–375, 376–450, 451+”
    4,  # AQI: “0–50, 51–100, 101–200, 201–300, 300+”
    3,  # Temperature: “Less than 0, 1 to 99, greater than 100”
    2,  # Temperature: “Less than 5, 5 to 25, greater than 25”
    1,  # Temperature: “Less than 3, 3 or greater”
    3,  # Temperature: “Less than or equal to -100, -100 to 0, 1 to 100, 101 to 200, 201 to 300, and so on”
    3,  # Temperature: “Less than 10, 10 to 15, 15 to 20, 20 to 25, 25 to 30, more than 30”
    4,  # Body Temperature: “Less than 36.1, 36.1 to 37.5, greater than 37.5”
    3,  # Body Temperature: “35 to 37, 37 to 39, 39 to 41, greater than 41”
    2,  # GNI per capita: “Less than 5, 5 or greater”
    3,  # GNI per capita: “Less than 1, 1 to 4, 4 to 7, 7 to 10, greater than 10”
    1,  # GNI per capita: “Less than 4.515, greater than or equal to 4.515”
    3,  # GNI per capita: “Less than 1.145, 1.146 to 4.515, 4.516 to 14.005, greater than 14.005”
    3,  # Movie Ratings: “0 to 1, 1 to 2, 2 to 3, and so on up to 9 to 10”
    2,  # Movie Ratings: “Less than or equal to 7, greater than 7”
    4,  # Movie Ratings: “Less than or equal to 4, 4 to 7, 7 to 9, greater than or equal to 9”
    4,  # Heart Rate: “50% to 60%, 60% to 70%, 70% to 80%, 80% to 90%, 90% to 100%”
    3,  # Heart Rate: “50% to 85%, 85% to 100%”
])
    perplexity_ranking = np.array([
        4,  # Blood Pressure: “Less than 120, 120 to 129, 130 to 139, 140 or higher”
        2,  # Blood Pressure: “Less than 120, 110 or higher”
        4,  # BMI: “Less than 18.5, 18.5 to 24.9, 25 to 29.9, 30 or higher”
        1,  # BMI: “Less than 5, 5 to 15, higher than 15”
        4,  # AQI: “0–50, 51–100, 101–150, 151–200, 201–300, greater than 300”
        3,  # AQI: “0–60, 61–120, 121–180, 181–240, 241–300, 301–500”
        2,  # AQI: “0–40, 41–80, 81–120, 121–300, 301–500”
        2,  # AQI: “0–75, 76–150, 151–225, 226–300, 301–375, 376–450, 451+”
        4,  # AQI: “0–50, 51–100, 101–200, 201–300, 300+”
        3,  # Temperature: “Less than 0, 1 to 99, greater than 100”
        2,  # Temperature: “Less than 5, 5 to 25, greater than 25”
        1,  # Temperature: “Less than 3, 3 or greater”
        3,  # Temperature: “Less than or equal to -100, -100 to 0, 1 to 100, 101 to 200, 201 to 300, and so on”
        3,  # Temperature: “Less than 10, 10 to 15, 15 to 20, 20 to 25, 25 to 30, more than 30”
        4,  # Body Temperature: “Less than 36.1, 36.1 to 37.5, greater than 37.5”
        3,  # Body Temperature: “35 to 37, 37 to 39, 39 to 41, greater than 41”
        2,  # GNI per capita: “Less than 5, 5 or greater”
        3,  # GNI per capita: “Less than 1, 1 to 4, 4 to 7, 7 to 10, 10 or greater”
        4,  # GNI per capita: “Less than 4.515, greater than or equal to 4.515”
        3,  # GNI per capita: “Less than 1.145, 1.146 to 4.515, 4.516 to 14.005, greater than 14.005”
        3,   # Movie Ratings: “0 to 1, 1 to 2, 2 to 3, and so on up to 9 to 10”
        3,  # Movie Ratings: “Less than or equal to 7, greater than 7”
        4,  # Movie Ratings: “Less than or equal to 4, 4 to 7, 7 to 9, greater than or equal to 9”
        4,  # Heart Rate: “50% to 60%, 60% to 70%, 70% to 80%, 80% to 90%, 90% to 100%”
        3,  # Heart Rate: “50% to 85%, 85% to 100%”
    ])
    
    # Ensure the constant vectors have the same length as quiz_vector
    if len(gpt_ranking) < len(quiz_vector):
        gpt_ranking = np.pad(gpt_ranking, (0, len(quiz_vector) - len(gpt_ranking)), 'wrap')
    elif len(gpt_ranking) > len(quiz_vector):
        gpt_ranking = gpt_ranking[:len(quiz_vector)]
        
    if len(perplexity_ranking) < len(quiz_vector):
        perplexity_ranking = np.pad(perplexity_ranking, (0, len(quiz_vector) - len(perplexity_ranking)), 'wrap')
    elif len(perplexity_ranking) > len(quiz_vector):
        perplexity_ranking = perplexity_ranking[:len(quiz_vector)]
    
    # Function to calculate metrics
    def calculate_metrics(const_vector, name):
        # Euclidean Distance
        distance = euclidean(const_vector, quiz_vector)
        
        # Kendall Tau
        tau, p_value = kendalltau(const_vector, quiz_vector)
        
        # Kendall Tau-a
        # tau_a = kendall_tau_a(const_vector, quiz_vector)
        
        print(f"\nMetrics for {name}:")
        print(f"Euclidean Distance: {distance:.4f}")
        
        # Calculate and print average distance after accumulating
        avg_distance = distance / len(quiz_vector)
        print(f"Average Euclidean Distance: {avg_distance:.4f}")
        print(f"Kendall Tau: {tau:.4f}")
        # print(f"Kendall Tau-a: {tau_a:.4f}")
        
        # Calculate Spearman's Rank Correlation
        corr, p_value = spearmanr(const_vector, quiz_vector)

        print(f"Spearman's Rank Correlation: {corr:.4f}")
        print(f"P-value: {p_value:.12f}")
        
        # calculate pearson correlation
        corr, p_value = pearsonr(const_vector, quiz_vector)
        print(f"Pearson Correlation: {corr:.4f}")
        print(f"P-value: {p_value:.12f}")
    
    # Calculate metrics for GPT Ranking
    calculate_metrics(gpt_ranking, "GPT Ranking")
    
    # Calculate metrics for Perplexity Ranking
    calculate_metrics(perplexity_ranking, "Perplexity Ranking")

if __name__ == "__main__":
    main()