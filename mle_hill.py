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


sampler = BayesianModelSampling(model)
# MAP Query 1
map_result_1 = inference.map_query(variables=['maint', 'lug_boot'], evidence={'car': 'good', 'safety': 'med'})
print("MAP Result 1 (maint and lug_boot for car='good' and safety='med'):", map_result_1)

# MAP Query 2
map_result_2 = inference.map_query(variables=['persons', 'doors'], evidence={'car': 'unacc', 'lug_boot': 'small'})
print("MAP Result 2 (persons and doors for car='unacceptable' and lug_boot='small'):", map_result_2)

# MAP Query 3
map_result_3 = inference.map_query(variables=['car'], evidence={'safety': 'high', 'persons': 'more', 'maint': 'low'})
print("MAP Result 3 (car classification for safety='high', persons='more', and maint='low'):", map_result_3)

# MAP Query 4
map_result_4 = inference.map_query(variables=['buying', 'lug_boot'], evidence={'maint': 'vhigh', 'safety': 'high'})
print("MAP Result 4 (buying and lug_boot for maint='vhigh' and safety='high'):", map_result_4)
print("Sensitive Analysis")
v_evidence = [("maint", "med")]
samples_sa1 = sampler.likelihood_weighted_sample(evidence=v_evidence, size=1000)
prob_lug_boot_sa1 = samples_sa1.groupby("lug_boot")["buying"].value_counts(normalize=True)
print(f"Sensitivity 1 (dependency between buying and lug_boot for maint='med'):\n{prob_lug_boot_sa1}")

v_evidence = [("safety", "high"), ("buying", "low")]
samples_sa2 = sampler.likelihood_weighted_sample(evidence=v_evidence, size=1000)
prob_doors_sa2 = samples_sa2.groupby("doors")["persons"].value_counts(normalize=True)
print(f"Sensitivity 2 (relationship between doors and persons for safety='high' and buying='low'):\n{prob_doors_sa2}")

v_evidence = [("persons", "4"), ("doors", "4")]
samples_sa3 = sampler.likelihood_weighted_sample(evidence=v_evidence, size=1000)
prob_lug_boot_sa3 = samples_sa3.groupby("maint")["lug_boot"].value_counts(normalize=True)
print(f"Sensitivity 3 (impact of maint on lug_boot for persons='4' and doors='4'):\n{prob_lug_boot_sa3}")

v_evidence = [("buying", "vhigh")]
samples_sa4 = sampler.likelihood_weighted_sample(evidence=v_evidence, size=1000)
prob_safety_sa4 = samples_sa4.groupby("doors")["safety"].value_counts(normalize=True)
print(f"Sensitivity 4 (relationship between safety and doors for buying='vhigh'):\n{prob_safety_sa4}")


v_evidence = [("maint", "low"), ("doors", "5more")]
samples_sa5 = sampler.likelihood_weighted_sample(evidence=v_evidence, size=1000)
prob_lug_boot_sa5 = samples_sa5.groupby("persons")["lug_boot"].value_counts(normalize=True)
print(f"Sensitivity 5 (interaction between lug_boot and persons for maint='low' and doors='5more'):\n{prob_lug_boot_sa5}")
