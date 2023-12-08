import genotype
import phenotype
from parameters import global_parameters
import torch
import torchvision
import tqdm
import random

data_transform = torchvision.transforms.ToTensor()

dataset_train = torchvision.datasets.CIFAR10("./data", download = True, train = True, transform = data_transform)
dataset_test = torchvision.datasets.CIFAR10("./data", download = True, train = False, transform = data_transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()

def calc_fitness(geno_network):
    # Converts the genotype into the phenotype.
    phen_network = phenotype.NetworkPhenotype(geno_network.layers)
    fitness = 0
    with torch.no_grad():
        for batch, label in tqdm.tqdm(dataloader_train):
            output = phen_network(batch)
            fitness += criterion(output, label)
    print(F"Fitness: {fitness}")
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
    return parents_to_mate

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

def evolve(network_class):
    generation_count = global_parameters["generation_count"]
    population_size = global_parameters["population_size"]
    initial_layer_count = global_parameters["initial_layer_count"]
    input_size = 32 * 32 * 3
    output_size = 10
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
        next_generation = select_parents(population)
        # Performs crossover on the parents.
        population_crossover(next_generation)
        # Performs mutation on the children.
        population_mutate(next_generation)
        population = next_generation
    print("Done Evolving")

evolve(genotype.NetworkType2)