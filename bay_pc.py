from pgmpy.estimators import PC
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'dataset.csv'  # Replace with your dataset file path
data = pd.read_csv(file_path, delimiter=';')



# Step 1: Perform structure learning using the PC Algorithm
pc = PC(data)
estimated_model = pc.estimate()

# Step 2: Learn parameters using Bayesian Estimator
model = BayesianNetwork(estimated_model.edges)
model.fit(data, estimator=BayesianEstimator)

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
plt.savefig("plots/bayesian_pc_network.png", format="png", bbox_inches="tight")
plt.close()
print("\nNetwork visualization saved as 'bayesian_pc_network.png'")
