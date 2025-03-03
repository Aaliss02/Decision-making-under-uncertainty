from data import get_fixed_data
from WindProcess import wind_model
import numpy as np
from sklearn.cluster import KMeans

def clustering(current_wind, previous_wind, data):
    # Monte Carlo simulation
    np.random.seed(42)
    n_simulations = 1000  # Number of Monte Carlo samples
    n_clusters = 42       # Number of representative scenarios

    # Generate wind scenarios (1D array)
    scenarios = np.zeros(n_simulations)

    for i in range(n_simulations):
        scenarios[i] = wind_model(current_wind, previous_wind, data)  # Single wind value per scenario

    # Reshape for clustering (2D array with one column)
    X = scenarios.reshape(-1, 1)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Compute probabilities for reduced scenarios
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()  # Normalize probabilities

    # Print wind values and probabilities for each scenario
    print("\n=== Clustered Wind Scenarios ===")
    for cluster, prob in zip(unique, probabilities):
        wind_value = kmeans.cluster_centers_[cluster][0]  # Get cluster center (wind value)
        print(f"Scenario {cluster}: Wind = {wind_value:.2f}, Probability = {prob:.2%}")

    return labels, probabilities  # Return clusters and probabilities

# Example call
current_wind = 10
previous_wind = 8
data = get_fixed_data()
clustering(current_wind, previous_wind, data)




