import genotype
import phenotype
from parameters import global_parameters
import torch
import torchvision
import tqdm
import random
import time

data_transform = torchvision.transforms.ToTensor()

dataset_train = torchvision.datasets.CIFAR10("./data", download = True, train = True, transform = data_transform)
dataset_test = torchvision.datasets.CIFAR10("./data", download = True, train = False, transform = data_transform)

dataset_train = torch.utils.data.Subset(dataset_train, range(30_000))

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False)

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
    phen_network = phenotype.NetworkPhenotype(solution.layers)
    with torch.no_grad():
        for batch, label in tqdm.tqdm(dataloader_test):
            output = phen_network(batch)
            choices = torch.argmax(output,dim=1)
            score += (choices==label).sum().item()
    accuracy = score / len(dataloader_test.dataset)
    print(F"Accuracy on Test Set: {accuracy}")
    return accuracy

def save_results(best_fitnesses, accuracy, time_elapsed):
    filename = F"results/{time.time()}.txt"
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
    _, best_fitness, best_solution = select_parents(population)
    best_fitnesses.append(best_fitness.item())
    accuracy = evaluate_solution(best_solution)
    time_elapsed = time.time() - start_time
    save_results(best_fitnesses, accuracy, time_elapsed)
    print("Done Evolving")

if __name__ == "__main__":
    if global_parameters["network_type"] == 2:
        network_class = genotype.NetworkType2
    else:
        raise ValueError(F"Invalid network type: {global_parameters['network_type']}")
    evolve(network_class)
