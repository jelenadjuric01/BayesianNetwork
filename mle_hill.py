from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'dataset.csv'  # Replace with your dataset file path
data = pd.read_csv(file_path, delimiter=';')

# Convert categorical data to discrete integers

# Step 1: Perform Hill Climbing with K2Score as the scoring metric
hc = HillClimbSearch(data)
estimated_model = hc.estimate(scoring_method=K2Score(data))

# Print estimated edges
print("Estimated edges:", estimated_model.edges)



# Step 2: Learn parameters using Maximum Likelihood Estimation
model = BayesianNetwork(estimated_model.edges)
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Print the learned structure and parameters
print("Learned Structure:")
print(model.edges)

print("\nLearned Parameters:")
for cpd in model.get_cpds():
    print(cpd)



# Visualize the Bayesian Network
plt.figure(figsize=(10, 8))
networkx_model = nx.DiGraph(model.edges)
pos = nx.spring_layout(networkx_model)
nx.draw(
    networkx_model,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=10,
    font_weight="bold",
    arrowsize=20
)
plt.title("Learned Bayesian Network")
plt.savefig("plots/mle_hill_k2_network.png", format="png", bbox_inches="tight")
plt.close()
print("\nNetwork visualization saved as 'mle_hill_k2_network.png'")
