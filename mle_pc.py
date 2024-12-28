from pgmpy.estimators import PC, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.base import DAG
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling


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
parent_nodes = ["car"]

print("Baseline Probabilities:")
for parent in parent_nodes:
    result = inference.query(variables=[parent])
    print(f"Query: {parent}")
    print(result, "\n")

# Compute probabilities with evidence
print("Probabilities with Evidence:")
for i, scenario in enumerate(scenarios, 1):
    evidence = scenario["evidence"]
    query = scenario["query"]
    result = inference.query(variables=[query], evidence=evidence)
    print(f"Scenario {i}: Evidence={evidence}, Query={query}")
    print(result, "\n")

parent_nodes = ["buying", "maint", "lug_boot", "persons", "safety"]

print("Baseline Probabilities:")
for parent in parent_nodes:
    result = inference.query(variables=[parent])
    print(f"Query: {parent}")
    print(result, "\n")

scenarios = [
    {"evidence": {"car": "unacc"}, "query": ["buying", "maint", "lug_boot", "persons", "safety"]},
    {"evidence": {"car": "acc"}, "query": ["buying", "maint", "lug_boot", "persons", "safety"]},
    {"evidence": {"car": "vgood"}, "query": ["buying", "maint", "lug_boot", "persons", "safety"]},
    {"evidence": {"car": "good"}, "query": ["buying", "maint", "lug_boot", "persons", "safety"]}
]

# Perform diagnostic reasoning for each scenario
for i, scenario in enumerate(scenarios, 1):
    evidence = scenario["evidence"]
    queries = scenario["query"]
    print(f"Scenario {i}: Evidence={evidence}")
    for query in queries:
        result = inference.query(variables=[query], evidence=evidence)
        print(f"Query: {query}")
        print(result, "\n")

parent_nodes = ["buying", "maint", "lug_boot", "safety"]

print("Baseline Probabilities:")
for parent in parent_nodes:
    result = inference.query(variables=[parent])
    print(f"Query: {parent}")
    print(result, "\n")


    # Define scenarios for intercausal reasoning
scenarios = [
    {'evidence': {'car': 'acc', 'buying': 'low'}, 'query': ['safety', 'maint','lug_boot']},
    {'evidence': {'car': 'unacc', 'safety': 'low'}, 'query': ['buying', 'maint','lug_boot']},
    {'evidence': {'car': 'vgood', 'maint': 'low'}, 'query': ['lug_boot', 'safety','buying']},
    {'evidence': {'car': 'good', 'lug_boot': 'big'}, 'query': ['buying', 'safety','maint']}
]

# Run inference for each scenario and display the results
for i, scenario in enumerate(scenarios):
    evidence = scenario['evidence']
    query = scenario['query']

    print(f"Scenario {i+1}: Evidence={evidence}")
    for var in query:
        result = inference.query(variables=[var], evidence=evidence)
        print(f"\nQuery: {var}\n{result}")
    print("-" * 50)



sampler = BayesianModelSampling(model)

# ------------------------------
# Maximum A Posteriori (MAP)
# ------------------------------
# Query 1: MAP for `buying`, `maint`, and `safety` given `car = vgood`
map_query_1 = inference.map_query(variables=["buying", "maint", "safety"], evidence={"car": "vgood"})
print("MAP Query 1:", map_query_1)



# Query 3: MAP for `car` given favorable conditions
map_query_3 = inference.map_query(variables=["car"], evidence={"buying": "low", "maint": "low", "safety": "high"})
print("MAP Query 3:", map_query_3)

# ------------------------------
# Sensitivity Analysis
# ------------------------------
# Query 1: Probability of `car = vgood` as `maint` shifts from `low` to `high`
sensitivity_results = {}
for maint_level in ["low", "med", "high"]:
    prob_vgood = inference.query(variables=["car"], evidence={"maint": maint_level, "buying": "low", "safety": "high"})
    sensitivity_results[maint_level] = prob_vgood.values[3]  # Index for 'vgood'

print("Sensitivity Analysis - Maint Levels:", sensitivity_results)

# Query 2: Probability of `car = acc` as `lug_boot` changes
sensitivity_results_lug_boot = {}
for lug_boot_size in ["small", "med", "big"]:
    prob_acc = inference.query(variables=["car"], evidence={"lug_boot": lug_boot_size, "persons": "4"})
    sensitivity_results_lug_boot[lug_boot_size] = prob_acc.values[1]  # Index for 'acc'

print("Sensitivity Analysis - Lug Boot:", sensitivity_results_lug_boot)

# ------------------------------
# Approximate Reasoning
# ------------------------------
# Query 1: Approximate probability of `car = acc`
sample_data = sampler.likelihood_weighted_sample(evidence={"safety": "med", "persons": "more"}, size=1000)
approx_acc_prob = len(sample_data[sample_data["car"] == "acc"]) / len(sample_data)
print("Approximate Probability (car = acc):", approx_acc_prob)

# Query 2: Simulated probability of `car = vgood` with random feature distribution
random_samples = sampler.forward_sample(size=1000)
vgood_prob = len(random_samples[random_samples["car"] == "vgood"]) / len(random_samples)
print("Simulated Probability (car = vgood):", vgood_prob)

## Query 3: Approximate probability of `car = good`
sample_data_good = sampler.likelihood_weighted_sample(evidence={"lug_boot": "small", "buying": "med"}, size=1000)
approx_good_prob = len(sample_data_good[sample_data_good["car"] == "good"]) / len(sample_data_good)
print("Approximate Probability (car = good):", approx_good_prob)