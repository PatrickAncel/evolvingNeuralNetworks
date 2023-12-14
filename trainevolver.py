import torch
import torchvision
import tqdm
import random
import time
import numpy as np

from parameters import *
import genotype

data_transform = torchvision.transforms.ToTensor()

dataset_train = torchvision.datasets.CIFAR10("./data", download = True, train = True, transform = data_transform)
dataset_test = torchvision.datasets.CIFAR10("./data", download = True, train = False, transform = data_transform)

dataset_train = torch.utils.data.Subset(dataset_train, range(30_000))

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False)


class TrainableNetwork(genotype.NetworkType2):
    def mutate(self):
        for layer in self.layers:
            layer.mutate_rescale()
        self._mutate_insert_remove_layer()
        self.repair_all()
    def accept_training(self, phen_network):
        '''Accepts new weights and biases from a trained model.'''
        i = 0
        for phen_layer in phen_network.layers:
            if isinstance(phen_layer, torch.nn.Linear):
                geno_layer = self.layers[i]
                i += 1
                # Copies the weights and biases from the phenotype layer.
                geno_layer.W = phen_layer.weight.detach().numpy()
                geno_layer.b = phen_layer.bias.detach().numpy()
                # Reshapes the bias vector.
                geno_layer.b = geno_layer.b.reshape((geno_layer.b.shape[0], 1))
    @classmethod
    def generate(cls, layer_count, input_size, output_size):
        if layer_count < 1:
            raise ValueError(F"Cannot create a NN with {layer_count} layers.")
        layers = []
        # Generates the size of each layer, including the input layer.
        # The input and output layers will very likely have incorrect sizes.
        mean = global_parameters["initial_layer_size_mean"]
        sigma = global_parameters["initial_layer_size_sigma"]
        layer_sizes = np.random.default_rng().normal(mean, sigma, (layer_count + 1,))
        # Fixes any layers that are too small.
        minimum_sizes = np.ones((layer_count + 1,)) * global_parameters["min_layer_size"]
        layer_sizes = np.maximum(layer_sizes, minimum_sizes)
        # Fixes any layers that are too large.
        maximum_sizes = np.ones((layer_count + 1,)) * global_parameters["max_layer_size"]
        layer_sizes = np.maximum(layer_sizes, maximum_sizes)
        # Fixes the sizes of the input and output layers.
        layer_sizes[0] = input_size
        layer_sizes[-1] = output_size
        # Iterates over the layer indices, starting from the first non-input layer.
        for layer_index in range(1, layer_count + 1):
            this_layer_length = round(layer_sizes[layer_index])
            previous_layer_length = round(layer_sizes[layer_index - 1])
            layers.append(genotype.Layer.generate(this_layer_length, previous_layer_length))
        return TrainableNetwork(layer_count, input_size, output_size, layers)
    def copy(self):
        '''Returns a copy of this network.'''
        # Copies every layer.
        copied_layers = [layer.copy() for layer in self.layers]
        # Copies this network.
        return TrainableNetwork(self.layer_count, self.input_size, self.output_size, copied_layers)

def train(geno_network):
    '''Trains a network for a few epochs.'''
    epoch_count = 2
    # Generates a phenotype object.
    phen_network = NetworkPhenotype(geno_network.get_layers())
    optimizer = torch.optim.Adam(phen_network.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    phen_network.train()
    # Trains the phenotype.
    for i in range(epoch_count):
        for batch, label in tqdm.tqdm(dataloader_train):
            optimizer.zero_grad()
            output = phen_network(batch)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
    # Inserts the trained parameters back into the genotype.
    geno_network.accept_training(phen_network)

def population_train(population):
    print("Training Population Members")
    members_trained = 0
    for geno_network in population:
        train(geno_network)
        print(F"Trained {members_trained} Network(s)")
        members_trained += 1

## MOSTLY DUPLICATE CODE FROM phenotype.py
########################################################################

class NetworkPhenotype(torch.nn.Module):
    def __init__(self, layers):
        super(NetworkPhenotype, self).__init__()
        self.layers = []
        self.input_size = layers[0].len0
        for i in range(len(layers)):
            # Genotype layer.
            geno_layer = layers[i]
            # Phenotype layer.
            phen_layer = torch.nn.Linear(geno_layer.len0, geno_layer.len1)
            # Copies the weights and biases of the genotype layer.
            with torch.no_grad():
                phen_layer.weight = torch.nn.Parameter(torch.from_numpy(geno_layer.W).float())
                phen_layer.bias = torch.nn.Parameter(torch.from_numpy(geno_layer.b[:,0]).float())
            self.layers.append(phen_layer)
            if i < len(layers) - 1:
                # ReLU Activation
                self.layers.append(torch.nn.ReLU())
            else:
                # Sigmoid Activation
                #self.layers.append(torch.nn.Sigmoid())
                pass
        self.layers = torch.nn.ModuleList(self.layers)
    
    def forward(self, input):
        output = input.view(-1, self.input_size)
        for layer in self.layers:
            output = layer(output)
        return output

## MOSTLY DUPLICATE CODE FROM evolver.py
########################################################################

def calc_fitness(geno_network):
    # Converts the genotype into the phenotype.
    phen_network = NetworkPhenotype(geno_network.get_layers())
    fitness = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch, label in tqdm.tqdm(dataloader_train):
            output = phen_network(batch)
            fitness += criterion(output, label)
    print(F"Fitness: {fitness}")
    print(F"Layer Count: {len(geno_network.get_layers())}")
    return fitness

def shuffle(li):
    '''Performs a Fisher-Yates shuffle on a list in place.'''
    for i in range(len(li) - 1):
        j = random.randint(i, len(li) - 1)
        li[i], li[j] = li[j], li[i]

def select_parents(population):
    print("Performing Selection")
    parents_to_mate = []
    # Iterates over the population twice.
    known_fitnesses = dict()
    best_fitness = float("inf")
    best_solution = None
    for run in range(2):
        # Shuffles the population.
        shuffle(population)
        # Iterates over the population in pairs.
        for i in range(0, len(population), 2):
            if run == 0:
                fitness1 = calc_fitness(population[i])
                fitness2 = calc_fitness(population[i+1])
                # Stores the fitnesses for the next run.
                known_fitnesses[population[i]] = fitness1
                known_fitnesses[population[i+1]] = fitness2
                if fitness1 < best_fitness:
                    best_fitness = fitness1
                    best_solution = population[i]
                if fitness2 < best_fitness:
                    best_fitness = fitness2
                    best_solution = population[i+1]
            else:
                # Uses the previously computed fitnesses.
                fitness1 = known_fitnesses[population[i]]
                fitness2 = known_fitnesses[population[i+1]]
            # Selects the individual with the lower (better) fitness.
            # The second individual is favored in a tie.
            if fitness1 < fitness2:
                parents_to_mate.append(population[i].copy())
            else:
                parents_to_mate.append(population[i+1].copy())
    return (parents_to_mate, best_fitness, best_solution)

def population_crossover(selected_parents):
    '''Performs crossover on the selected parents, replacing them with children.'''
    print("Performing Crossover")
    # Shuffles the parents.
    shuffle(selected_parents)
    # Iterates over the parents in pairs.
    for i in range(0, len(selected_parents), 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i+1]
        parent1.crossover(parent2)

def population_mutate(population):
    '''Performs mutation on the whole population.'''
    print("Performing Mutation")
    for i in range(len(population)):
        population[i].mutate()

def evaluate_solution(solution):
    '''Evaluates the accuracy of a network on the test dataset.'''
    print("Evaluating Best Solution")
    score = 0
    phen_network = NetworkPhenotype(solution.get_layers())
    with torch.no_grad():
        for batch, label in tqdm.tqdm(dataloader_test):
            output = phen_network(batch)
            choices = torch.argmax(output,dim=1)
            score += (choices==label).sum().item()
    accuracy = score / len(dataloader_test.dataset)
    print(F"Accuracy on Test Set: {accuracy}")
    return accuracy

def save_results(best_fitnesses, accuracy, time_elapsed):
    filename = F"trainevolve results/{time.time()}.txt"
    f = open(filename, "w")
    f.write(str(global_parameters))
    f.write(F"\n\n")
    f.write(str(best_fitnesses))
    f.write(F"\n\nAccuracy: {accuracy}\n\nTime Elapsed: {time_elapsed}")
    f.close()

def evolve(network_class):
    start_time = time.time()
    generation_count = global_parameters["generation_count"]
    population_size = global_parameters["population_size"]
    initial_layer_count = global_parameters["initial_layer_count"]
    input_size = 32 * 32 * 3
    output_size = 10
    best_fitnesses = [] # Tracks the best fitness at each generation.
    # Generates an initial population.
    print(F"Generating Initial Population: 0/{population_size}", end="")
    population = []
    for i in range(population_size):
        # Generates a population member.
        network = network_class.generate(initial_layer_count, input_size, output_size)
        population.append(network)
        print(F"\rGenerating Initial Population: {i+1}/{population_size}", end="")
    print("\nBeginning Evolution")
    for i in range(generation_count):
        print(F"Generation: {i}")
        # Selects parents of the next generation.
        next_generation, best_fitness, _ = select_parents(population)
        best_fitnesses.append(best_fitness.item())
        # Performs crossover on the parents.
        population_crossover(next_generation)
        # Performs mutation on the children.
        population_mutate(next_generation)
        population = next_generation
        population_train(population)
    _, best_fitness, best_solution = select_parents(population)
    best_fitnesses.append(best_fitness.item())
    accuracy = evaluate_solution(best_solution)
    time_elapsed = time.time() - start_time
    save_results(best_fitnesses, accuracy, time_elapsed)
    print("Done Evolving")

if __name__ == "__main__":
    evolve(TrainableNetwork)
