from WindProcess import wind_model
from PriceProcess import price_model
import numpy as np
from sklearn.cluster import KMeans

def clustering(n_clusters, current_wind, previous_wind, current_price, previous_price, problemData):
    # Monte Carlo simulation
    np.random.seed(42)
    n_simulations = 1000  # Number of Monte Carlo samples

    # Generate wind scenarios (1D array)
    scenarios = np.zeros(n_simulations)

    wind = [wind_model(current_wind, previous_wind, problemData) for i in range(n_simulations)]
    price = [price_model(current_price, previous_price, wind[i], problemData) for i in range(n_simulations)]
    data = list(zip(wind,price))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    prob = [0 for i in range(n_clusters)]
    for i in range(n_simulations):
        prob[labels[i]] = prob[labels[i]] + 1
    
    for i in range(n_clusters):
        prob[i] = prob[i]/n_simulations
    
    return centers, prob



