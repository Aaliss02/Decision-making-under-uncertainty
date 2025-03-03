from data import get_fixed_data
from WindProcess import wind_model
import numpy as np
from sklearn.cluster import KMeans

def clustering(current_wind, previous_wind):
    # Monte Carlo simulation
    np.random.seed(42)
    n_simulations = 1000  # Numero di scenari
    n_clusters = 41       # Numero di scenari rappresentativi
    data = get_fixed_data()

    # Generazione degli scenari (ora è un array 1D)
    scenarios = np.zeros(n_simulations)

    for i in range(n_simulations):
        scenarios[i] = wind_model(current_wind, previous_wind, data)  # Ora è uno scalare

    # Reshape per clustering (array 2D con una colonna)
    X = scenarios.reshape(-1, 1)

    # Clustering con K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Calcolo delle probabilità degli scenari ridotti
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()  # Normalizzazione

    # Stampa delle probabilità degli scenari
    for cluster, prob in zip(unique, probabilities):
        print(f"Scenario {cluster}: Probabilità {prob:.2%}")

    return labels, probabilities  # Ritorna i cluster e le probabilità



