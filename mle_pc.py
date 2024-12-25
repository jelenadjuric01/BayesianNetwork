from pgmpy.estimators import PC, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.base import DAG
from pgmpy.inference import VariableElimination


# Load the dataset with proper delimiter
file_path = 'dataset.csv'  # Replace with your actual file path
data = pd.read_csv(file_path, delimiter=';')

# Display the first few rows of the dataset
print("Original Dataset:")
print(data.head())

for column in data.columns:
    print(column, data[column].unique(), data[column].dtype)

#MLE and PC
pc = PC(data)
estimated_model = pc.estimate()

# Step 2: Learn parameters using Maximum Likelihood Estimation
model = BayesianNetwork(estimated_model.edges)  # Convert to Bayesian Network
model.fit(data, estimator=MaximumLikelihoodEstimator)

print("\nLearned Parameters:")
for cpd in model.get_cpds():
    print(cpd)

#Save the model
model_path = 'models/mle_pc_model.bif'
model.save(model_path)

graph = nx.DiGraph(model.edges)

# Plot the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(graph)  # Layout for better visualization
nx.draw(
    graph,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=10,
    font_weight="bold",
    arrowsize=20 
)
# Save the plot as a picture
output_path = 'plots/mle_pc.png'
plt.title("Learned Bayesian Network")
plt.savefig(output_path, format='png', bbox_inches='tight')
plt.close()  # Close the plot to avoid displaying it

print(f"Network visualization saved as '{output_path}'")

G = DAG()
G.add_nodes_from(model.nodes())
G.add_edges_from(model.edges())
print("Independencies:")
print(G.get_independencies())

# Define tests for d-separation
tests = [
    ("maint", "safety", ["buying", "persons", "lug_boot"]),
    ("safety", "lug_boot", ["maint", "persons", "buying"]),
    ("buying", "persons", ["maint", "safety", "lug_boot"]),
    ("lug_boot", "persons", ["car"]),
    ("buying", "safety", ["car"]),
    ("safety", "buying", ["car"]),
    ("maint", "lug_boot", ["car"]),
    ("persons", "buying", ["safety", "lug_boot", "maint"]),
    ("lug_boot", "safety", ["persons", "maint", "buying"])
]

# Test each case and print results
for x, y, z in tests:
    result = G.is_dconnected(x, y, observed=z)
    print(f"Is '{x}' d-connected to '{y}' given {z}? {'Yes' if result else 'No'}")

for node in model.nodes():
    markov_blanket = model.get_markov_blanket(node)
    print(f"Markov blanket for '{node}': {markov_blanket}")

inference = VariableElimination(model)


scenarios = [
    {"evidence": {"safety": "high"}, "query": "car"},
    {"evidence": {"maint": "high"}, "query": "car"},
    {"evidence": {"persons": "more", "safety": "low"}, "query": "car"},
    {"evidence": {"buying": "low"}, "query": "car"},
    {"evidence": {"buying": "low", "maint": "med", "persons": "4", "safety": "med"}, "query": "car"},
    {"evidence": {"buying": "high", "maint": "low", "safety": "high", "persons": "4"}, "query": "car"}
]

# Compute baseline probabilities for queries
print("Baseline Probabilities:")
for scenario in scenarios:
    query = scenario["query"]
    baseline = inference.query(variables=[query])
    print(f"Query: {query}")
    print(baseline, "\n")

# Compute probabilities with evidence
print("Probabilities with Evidence:")
for i, scenario in enumerate(scenarios, 1):
    evidence = scenario["evidence"]
    query = scenario["query"]
    result = inference.query(variables=[query], evidence=evidence)
    print(f"Scenario {i}: Evidence={evidence}, Query={query}")
    print(result, "\n")
