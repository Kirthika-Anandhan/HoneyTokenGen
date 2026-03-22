import math
from collections import Counter
import random


# Entropy Calculation
def calculate_entropy(token):

    prob = [n_x / len(token) for x, n_x in Counter(token).items()]
    entropy = -sum(p * math.log2(p) for p in prob)

    return round(entropy, 3)


# Similarity Score Simulation
def similarity_score(similarity):
    return round(similarity + random.uniform(-0.05, 0.05), 3)


# Discriminator Score (GAN-style simulation)
def discriminator_score():
    return round(random.uniform(0.85, 0.99), 3)
