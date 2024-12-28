from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.base import DAG
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

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

model_path = 'models/mle_hill_model.bif'
model.save(model_path)

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



G = DAG()
G.add_nodes_from(model.nodes())
G.add_edges_from(model.edges())
print("Independencies:")
print(G.get_independencies())

tests = [
    # Impact of Indirect Relationships
    ("persons", "lug_boot", ["car", "safety"]),
    ("maint", "safety", ["car", "lug_boot"]),
    
    # Joint Effects on Key Nodes
    ("doors", "lug_boot", ["car", "persons"]),
    ("maint", "persons", ["car", "safety"]),
    
    # Higher-Order Conditioning
    ("doors", "maint", ["lug_boot", "car", "safety"]),
    ("lug_boot", "maint", ["car", "persons", "doors"]),
    
    # Testing Causal Paths
    ("safety", "maint", ["persons", "car"]),
    ("doors", "safety", ["lug_boot", "car"]),
    
    # Testing for Redundancy
    ("persons", "maint", ["lug_boot", "car"]),
    ("doors", "lug_boot", ["safety", "car"]),
    
    # Effects of Removing Conditioning
    ("maint", "doors", ["car"]),
    ("lug_boot", "persons", ["safety"]),
    
    # Symmetry and Reversals
    ("maint", "safety", ["lug_boot"]),
    ("doors", "lug_boot", ["maint", "car"])
]

# Test each case and print results
for x, y, z in tests:
    result = G.is_dconnected(x, y, observed=z)  # Replace G with your network object
    print(f"Is '{x}' d-connected to '{y}' given {z}? {'Yes' if result else 'No'}")

for node in model.nodes():
    markov_blanket = model.get_markov_blanket(node)
    print(f"Markov blanket for '{node}': {markov_blanket}")



inference = VariableElimination(model)

parent_nodes = ["safety", "persons","lug_boot", "buying"]

print("Baseline Probabilities:")
for parent in parent_nodes:
    result = inference.query(variables=[parent])
    print(f"Query: {parent}")
    print(result, "\n")
#Predictive reasoning
scenarios = [
   {"evidence": {"car": "acc"}, "query": "safety"},
   {"evidence": {"safety": "med"}, "query": "lug_boot"},
    {"evidence": {"car": "unacc","safety":"med"}, "query": "persons"},
   {"evidence": {"maint": "low", "safety": "high"}, "query": "buying"},
    {"evidence": {"doors": "4", "persons": "more"}, "query": "buying"},
    {"evidence": {"car": "good", "maint": "med"}, "query": "buying"}
]

print("Probabilities with Evidence:")
for i, scenario in enumerate(scenarios, 1):
    evidence = scenario["evidence"]
    query = scenario["query"]
    result = inference.query(variables=[query], evidence=evidence)
    print(f"Scenario {i}: Evidence={evidence}, Query={query}")
    print(result, "\n")

#Diagnostic reasoning

parent_nodes = ["maint", "car","safety"]

print("Baseline Probabilities:")
for parent in parent_nodes:
    result = inference.query(variables=[parent])
    print(f"Query: {parent}")
    print(result, "\n")

scenarios = [
    {"evidence": {"buying": "vhigh"}, "query": "maint"},
    {"evidence": {"buying": "vhigh"}, "query": "car"},
    {"evidence": {"buying": "vhigh"}, "query": "safety"},
    {"evidence": {"lug_boot": "small"}, "query": "car"},
    {"evidence": {"persons": "2"}, "query": "car"},
    {"evidence": {"safety": "low"}, "query": "car"},
    {"evidence": {"lug_boot": "big"}, "query": "safety"}
]

print("Probabilities with Evidence:")
for i, scenario in enumerate(scenarios, 1):
    evidence = scenario["evidence"]
    query = scenario["query"]
    result = inference.query(variables=[query], evidence=evidence)
    print(f"Scenario {i}: Evidence={evidence}, Query={query}")
    print(result, "\n")

#Intercausal reasoning
parent_nodes = ["doors", "maint","safety"]

print("Baseline Probabilities:")
for parent in parent_nodes:
    result = inference.query(variables=[parent])
    print(f"Query: {parent}")
    print(result, "\n")

scenarios = [
    {"evidence": {"maint": "high", "buying": "low"}, "query": "doors"},
    {"evidence": {"doors": "4", "buying": "vhigh"}, "query": "maint"},
    {"evidence": {"lug_boot": "big", "car": "unacc"}, "query": "safety"},
    {"evidence": {"car": "vgood", "persons": "4"}, "query": "safety"}
]

print("Probabilities with Evidence:")
for i, scenario in enumerate(scenarios, 1):
    evidence = scenario["evidence"]
    query = scenario["query"]
    result = inference.query(variables=[query], evidence=evidence)
    print(f"Scenario {i}: Evidence={evidence}, Query={query}")
    print(result, "\n")