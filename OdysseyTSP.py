# Import nessecary libraries
import numpy as np
import pandas as pd
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Read data from CSV
df = pd.read_csv("ulysis22TSP.csv", index_col=0)


class Simulated_Annealing:
    
    def __init__(self, cities, max_iter=10000, init_temp=100, alpha=10):
        self.cities = cities
        self.max_iter = max_iter
        self.init_temp = init_temp
        self.alpha = alpha
        self.progress = []
        self.temp_history = [init_temp]
    
    # Returns euclidean distance between a pair of cities' coordinates
    def distance_between_cities(self, c1, c2):
        return math.dist(c1,c2)

    # Returns the total traversed distance of a candidate route
    def total_distance(self, route):
        distance = 0
        for i in range(len(route)-1):
            distance += self.distance_between_cities(route[i], route[i+1])
        distance += self.distance_between_cities(route[-1], route[0])
        return distance
    
    # Returns updated temperature according to defined annealing schedule.
    def temp_fn(self, temp, iter):
        return temp*(1-self.alpha/self.max_iter)

    # Returns a new candidate route using 2-opt
    def get_neighbour(self, route):
        i = random.randint(0, len(route) - 1)
        j = random.randint(i + 1, len(route))
        new_route = np.concatenate((route[:i], route[i:j+1][::-1], route[j+1:])) # Generate two cities and reverse traversal order between them
        return new_route

    # Performs simulated annealing
    def simulated_annealing(self):
        route = self.cities
        best_route = route
        current_distance = self.total_distance(best_route)
        best_distance = [current_distance]
        self.progress.append(current_distance)
        temp = self.init_temp
        iter = 0
        while iter < self.max_iter: # Continue until desired number of iterations is reached
            new_route = self.get_neighbour(route) # Get new candidate
            new_distance = self.total_distance(new_route)
            delta = new_distance - current_distance
            if delta < 0 or np.random.random() < np.exp(-delta/temp): # If candiate is better, accept it. If not better, sometimes accept anyway
                route = new_route                                                 
                current_distance = new_distance
                self.progress.append(current_distance)
            if current_distance < best_distance[-1]: # Check if this is the best so far
                best_route = route
                best_distance.append(current_distance)
            temp = self.temp_fn(temp, iter) # Update temperature
            self.temp_history.append(temp)
            iter += 1
        return best_distance, best_route
    

class GeneticAlgorithm:
    
    def __init__(self, cities, population_size=50, num_parents=20, beta=2):
        self.num_parents = num_parents
        self.cities = cities
        self.population_size = population_size
        self.beta = beta
        self.alpha = 2 - self.beta
        self.max_iter = math.floor((10000-self.population_size)/self.num_parents)
        self.pool = []
        self.progress = []
        self.fitness_calculations = 0
        for i in range(self.population_size):
            permutation = np.random.permutation((self.cities))
            self.pool.append([permutation, self.total_distance(permutation)])
             
    # Returns euclidean distance between a pair of cities' coordinates
    def distance_between_cities(self, c1, c2):
        return math.dist(c1,c2)

    # Returns the total traversed distance of a candidate route
    def total_distance(self, route):
        distance = 0
        for i in range(len(route)-1):
            distance += self.distance_between_cities(route[i], route[i+1])
        distance += self.distance_between_cities(route[-1], route[0])
        self.fitness_calculations += 1
        return distance
    
    # Returns a selection of parents with probability linearly proportional to ranking
    def select_parents(self, pool):
        pool.sort(key = lambda x: x[1], reverse=True) # Sort pool by total distance in descending order
        probabilities = np.empty(len(pool))
        for rank in range(len(pool)):
            probabilities[rank] = (self.alpha + (self.beta - self.alpha)*(rank/(len(pool)-1)))/len(pool) # Calculate linear ranking probabilities
        parents = []
        for i in range(self.num_parents):
            parents.append(pool[np.random.choice(len(pool), p=probabilities)]) # Randomly select parents proportional to probabilities
        return parents

    # Returns a number of mutations of the selected parents according to desired population size and number of selected parents
    def mutate(self, parents):
        mutations = []
        for parent in parents:
            i = random.randint(0, len(parent[0]) - 1)
            j = random.randint(i + 1, len(parent[0]))
            child = np.concatenate((parent[0][:i], parent[0][i:j+1][::-1], parent[0][j+1:])) # Generate two cities and perform 2-opt to mutate
            distance = self.total_distance(child)
            mutations.append([child, distance])
        return mutations
    
    # Performs Genetic Algorithm
    def genetic_algorithm(self):
        pool = self.pool
        iter = 0
        best_route = [sorted(pool, key = lambda x: x[1])[0]] # Sort by total distance in ascending order and take best route of pool
        while iter < self.max_iter:
            next_gen = []
            parents = self.select_parents(pool)
            children = self.mutate(parents)
            next_gen.extend(parents)
            next_gen.extend(children)
            for i in range(self.population_size - 2*self.num_parents): # Fill rest of pool to reach population size with randomly selected candidates from previous generation
                next_gen.append(pool[np.random.choice(len(pool))])
            pool = next_gen
            best_candidate = sorted(pool, key = lambda x: x[1])[0] # Find best candidate of pool
            self.progress.append(best_candidate) 
            if best_candidate[1] < best_route[-1][1]: # If candidate is the best found so far, add it as the new best route
                best_route.append(best_candidate)
            iter += 1
        return best_route
    

cities = df.T.to_numpy()


# 30 runs of Simulated Annealing
sa_distances = []
print('Performing 30 runs of Simulated Annealing...')
for i in range(30):
    odyssey = Simulated_Annealing(cities)
    best_distance, best_route = odyssey.simulated_annealing()
    sa_distances.append(best_distance[-1])

# Calculate mean and standard deviation
sa_distances = np.array(sa_distances)
mean_simulated_annealing = np.mean(sa_distances)
std_simulated_annealing = np.std(sa_distances)
print('Simulated Annealing: Mean =', mean_simulated_annealing, 'Standard Deviation =', std_simulated_annealing, '\n')


# 30 runs of Genetic Algorithm
ga_distances = []
('Performing 30 runs of Genetic Algorithm...')
for i in range(30):
    odyssey = GeneticAlgorithm(cities, beta = 2)
    best_distance = odyssey.genetic_algorithm()[-1][1]
    ga_distances.append(best_distance)

# Calculate mean and standard deviation
ga_distances = np.array(ga_distances)
mean_genetic_algorithm = np.mean(ga_distances)
std_genetic_algorithm = np.std(ga_distances)
print('Genetic Algorithm: Mean = ', mean_genetic_algorithm, 'Standard Deviation = ', std_genetic_algorithm, '\n')


# Wilcoxon Signed-Rank Test
print('Wilcoxon Signed-Rank Test, 5% significance level:\n')
diffs = sa_distances - ga_distances
alpha = 0.025
statistic, p = wilcoxon(diffs, alternative='two-sided')
print('Test statistic = ', statistic)
print('p =', p)
if p < alpha:
    print(p, '<', alpha, 'So reject null hypothesis. Significant difference.')
else:
    print(p, '>', alpha, 'So cannot reject null hypothesis. No significant difference.')


#Individual test SA
print('Performing SA for some nice graphs...')
odyssey = Simulated_Annealing(cities)
best_distance, best_route = odyssey.simulated_annealing()
progress = odyssey.progress

plt.title("Search Trajectory")
plt.xlabel("Iteration")
plt.ylabel("Total distance of Current Candidate")
plt.plot(progress, color="red")
plt.show()

plt.title("Best Found Distance by Number of Improvements")
plt.xlabel("Improvements Found")
plt.ylabel("Best Distance")
plt.plot(best_distance, color="blue")
plt.show()

G = nx.Graph()
for i, coord in enumerate(best_route):
    G.add_node(i, pos=coord)
for i in range(len(best_route)):
    G.add_edge(i, (i+1) % len(best_route))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=False, node_size=25)
plt.show()


#Individual test GA
print('Performing GA for some nice graphs...')
odyssey = GeneticAlgorithm(cities, population_size=50, num_parents=20, beta = 2)
best_route = odyssey.genetic_algorithm()
print(best_route[-1][1])

progress = [tup[1] for tup in odyssey.progress] # get the second element of each tuple
plt.title("Search Trajectory")
plt.xlabel("Generation")
plt.ylabel("Total distance of Best Candidate in Pool")
plt.plot(progress, color="red")
plt.show()

best_distances = [tup[1] for tup in best_route]
plt.title("Best Found Distance by Number of Improvements")
plt.xlabel("Improvements Found")
plt.ylabel("Best Distance")
plt.plot(best_distances, color="red")
plt.show()

G = nx.Graph()
for i, coord in enumerate(best_route[-1][0]):
    G.add_node(i, pos=coord)
for i in range(len(best_route[-1][0])):
    G.add_edge(i, (i+1) % len(best_route[-1][0]))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=False, node_size=25)
plt.show()