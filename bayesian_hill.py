from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BDeuScore
from pgmpy.models import BayesianNetwork
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.base import DAG
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.readwrite import BIFWriter

# Load the dataset
file_path = 'dataset.csv'  # Replace with your dataset file path
data = pd.read_csv(file_path, delimiter=';')



# Step 1: Perform structure learning using Hill Climbing with BDeuScore
hc = HillClimbSearch(data)
estimated_model = hc.estimate(scoring_method=BDeuScore(data))  # Adjust equivalent_sample_size as needed

# Step 2: Learn parameters using Bayesian Estimator
model = BayesianNetwork(estimated_model.edges)
model.fit(data, estimator=BayesianEstimator)

# Print the learned structure and parameters
print("Learned Structure:")
print(model.edges)

print("\nLearned Parameters:")
for cpd in model.get_cpds():
    print(cpd)

model_path = 'models/bayesian_hill_model.bif'
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
plt.savefig("plots/bayesian_hill_bdeu_network.png", format="png", bbox_inches="tight")
plt.close()
print("\nNetwork visualization saved as 'bayesian_hill_bdeu_network.png'")


G = DAG()
G.add_nodes_from(model.nodes())
G.add_edges_from(model.edges())
print("Independencies:")
print(G.get_independencies())

tests = [
    # 1. Are maintenance and buying independent?
    ("maint", "buying", []),
    
    # 2. Is safety independent of maintenance cost?
    ("safety", "maint", []),
    ("safety", "maint", ["lug_boot"]),

    # 3. Does car quality break all dependencies?
    ("buying", "maint", ["car", "safety"]),
    ("buying", "lug_boot", ["car", "safety"]),

    # 4. Is safety independent of lug_boot under complex conditions?
    ("lug_boot", "safety", ["persons", "maint"]),

    # 5. Are persons and maintenance cost independent?
    ("persons", "maint", ["lug_boot", "safety"]),
]

# Test each case and print results
for x, y, z in tests:
    result = G.is_dconnected(x, y, observed=z)  # Replace G with your network object
    print(f"Is '{x}' d-connected to '{y}' given {z}? {'Yes' if result else 'No'}")

for node in model.nodes():
    markov_blanket = model.get_markov_blanket(node)
    print(f"Markov blanket for '{node}': {markov_blanket}")

inference = VariableElimination(model)
#car (), persons (car,safety), safety (car,lug_boot), lug_boot (car), buying (car), maint (car,buying) and the format is node (parents of the node)
parent_nodes = ["safety", "persons","lug_boot", "buying", 'maint']

print("Baseline Probabilities:")
for parent in parent_nodes:
    result = inference.query(variables=[parent])
    print(f"Query: {parent}")
    print(result, "\n")

scenarios = [
    # Core Queries
    {"evidence": {"lug_boot": "big"}, "query": "safety"},
    {"evidence": {"buying": "high"}, "query": "maint"},
    {"evidence": {"safety": "high"}, "query": "persons"},

    # Scenario-Based Queries
    {"evidence": {"car": "vgood"}, "query": "buying"},
    {"evidence": {"lug_boot": "big", "car": "good"}, "query": "persons"},

    # Impact of Safety and Luggage Size
    {"evidence": {"safety": "low"}, "query": "persons"},
    {"evidence": {"lug_boot": "big"}, "query": "safety"},

    # Complex Queries
    {"evidence": {"car": "acc"}, "query": "buying"},
    {"evidence": {"car": "acc"}, "query": "lug_boot"},

    # Counterfactual Queries
    {"evidence": {"lug_boot": "small"}, "query": "persons"}
]

print("Probabilities with Evidence:")
for i, scenario in enumerate(scenarios, 1):
    evidence = scenario["evidence"]
    query = scenario["query"]
    result = inference.query(variables=[query], evidence=evidence)
    print(f"Scenario {i}: Evidence={evidence}, Query={query}")
    print(result, "\n")

parent_nodes = ["safety", "car","lug_boot", "buying"]

print("Baseline Probabilities:")
for parent in parent_nodes:
    result = inference.query(variables=[parent])
    print(f"Query: {parent}")
    print(result, "\n")
# Diagnostic Reasoning Queries
scenarios = [
    {"evidence": {"persons": "more"}, "query": "car"},
    {"evidence": {"safety": "low"}, "query": "lug_boot"},
    {"evidence": {"persons": "4"}, "query": "safety"},
    {"evidence": {"maint": "high"}, "query": "buying"}
]

# Execute and print results
print("Diagnostic Reasoning Results:")
for i, scenario in enumerate(scenarios, 1):
    evidence = scenario["evidence"]
    query = scenario["query"]
    result = inference.query(variables=[query], evidence=evidence)
    print(f"Scenario {i}: Evidence={evidence}, Query={query}")
    print(result, "\n")

parent_nodes = ["safety","lug_boot", "buying"]

print("Baseline Probabilities:")
for parent in parent_nodes:
    result = inference.query(variables=[parent])
    print(f"Query: {parent}")
    print(result, "\n")
# Intercausal Queries
scenarios = [
    {"evidence": {"maint": "high", "car": "vgood"}, "query": "buying"},
    {"evidence": {"persons": "4", "car": "good"}, "query": "safety"},
    {"evidence": {"safety": "high", "car": "acc"}, "query": "lug_boot"}
]

# Execute and print results
print("Intercausal Reasoning Results:")
for i, scenario in enumerate(scenarios, 1):
    evidence = scenario["evidence"]
    query = scenario["query"]
    result = inference.query(variables=[query], evidence=evidence)
    print(f"Scenario {i}: Evidence={evidence}, Query={query}")
    print(result, "\n")


# MAP Queries
map_queries = [
    {"query": ["buying", "maint"], "evidence": {"safety": "high", "lug_boot": "small"}},
    {"query": ["safety", "persons"], "evidence": {"buying": "high", "car": "acc"}},
    {"query": ["lug_boot", "maint"], "evidence": {"car": "vgood", "persons": "4"}},
    {"query": ["safety", "buying"], "evidence": {"maint": "med", "lug_boot": "big"}},
    {"query": ["car", "persons"], "evidence": {"maint": "high", "safety": "low"}}
]

# Execute and print results
print("MAP Query Results:")
for i, scenario in enumerate(map_queries, 1):
    query = scenario["query"]
    evidence = scenario["evidence"]
    result = inference.map_query(variables=query, evidence=evidence)
    print(f"Scenario {i}: Query={query}, Evidence={evidence}")
    print(f"Most likely values: {result}\n")


# Initialize approximate inference
sampler = BayesianModelSampling(model)


# Sensitivity Analysis 1: Impact of safety on car acceptability for maint='high'
v_evidence = [("maint", "high")]
samples_sa1 = sampler.likelihood_weighted_sample(evidence=v_evidence, size=1000)
prob_car_sa1 = samples_sa1.groupby("safety")["car"].value_counts(normalize=True)
print(f"Sensitivity 1 (impact of safety on car acceptability for maint='high'):\n{prob_car_sa1}")

# Sensitivity Analysis 2: Relationship between seating capacity and safety for car='good'
v_evidence = [("car", "good")]
samples_sa2 = sampler.likelihood_weighted_sample(evidence=v_evidence, size=1000)
prob_safety_sa2 = samples_sa2.groupby("persons")["safety"].value_counts(normalize=True)
print(f"Sensitivity 2 (relationship between seating capacity and safety for car='good'):\n{prob_safety_sa2}")

# Sensitivity Analysis 3: Impact of luggage size on buying price for safety='high'
v_evidence = [("safety", "high")]
samples_sa3 = sampler.likelihood_weighted_sample(evidence=v_evidence, size=1000)
prob_buying_sa3 = samples_sa3.groupby("lug_boot")["buying"].value_counts(normalize=True)
print(f"Sensitivity 3 (impact of luggage size on buying price for safety='high'):\n{prob_buying_sa3}")

# Sensitivity Analysis 4: Interaction between maintenance and car quality for lug_boot='big'
v_evidence = [("lug_boot", "big")]
samples_sa4 = sampler.likelihood_weighted_sample(evidence=v_evidence, size=1000)
prob_car_sa4 = samples_sa4.groupby("maint")["car"].value_counts(normalize=True)
print(f"Sensitivity 4 (interaction between maintenance and car quality for lug_boot='big'):\n{prob_car_sa4}")

# Sensitivity Analysis 5: Impact of seating capacity on buying price for maint='med'
v_evidence = [("maint", "med")]
samples_sa5 = sampler.likelihood_weighted_sample(evidence=v_evidence, size=1000)
prob_buying_sa5 = samples_sa5.groupby("persons")["buying"].value_counts(normalize=True)
print(f"Sensitivity 5 (impact of seating capacity on buying price for maint='med'):\n{prob_buying_sa5}")
