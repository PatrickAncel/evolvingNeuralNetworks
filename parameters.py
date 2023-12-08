global_parameters = {
    # POPULATION PARAMETERS
    ######################################################################################
    # The number of individuals in the population.
    "population_size": 25,
    # The number of generations to run the GA for.
    "generation_count": 50,
    # NETWORK PARAMETERS
    ######################################################################################
    # The minimum acceptable number of nodes a layer can have.
    "min_layer_size": 5,
    # The mean size of a randomly generated neural network layer.
    "initial_layer_size_mean": 400,
    # The standard deviation of the size of a randomly generated neural network layer.
    "initial_layer_size_sigma": 100,
    # The size of an initial population member.
    "initial_layer_count": 5,
    # SINGLE-LAYER MUTATION AND CROSSOVER PARAMETERS
    ######################################################################################
    # Standard deviation of variable-wise Gaussian mutation of weights.
    "gaussian_mutation_weight_sigma": 0.05,
    # Standard deviation of variable-wise Gaussian mutation of biases.
    "gaussian_mutation_bias_sigma": 0.05,
    # Probability that a network layer will be resized during mutation.
    "rescale_mutation_probability": 0.05,
    # The standard deviation of the number of nodes added/removed during rescale mutation.
    "rescale_mutation_sigma": 100,
    # The alpha value for BLX.
    "blend_crossover_alpha": 0.5,
    # The eta value for SBX.
    "simulated_binary_crossover_eta": 2.0,
    # Whether to keep some values from the parent during SBX. This is slow.
    "simulated_binary_crossover_keep": False,
    # Valid values: "blx", "sbx"
    "layer_level_crossover_type": "sbx",
    # NETWORK-LEVEL MUTATION AND CROSSOVER PARAMETERS
    ######################################################################################
    # Probability that a network layer will be added or removed during mutation.
    "layer_insertion_removal_probability": 0.1,
    # Probability that a pair of layers will be swapped during mutation.
    "layer_swap_probability": 0.1,
    # Probability that two networks will undergo network-split crossover.
    "network_split_crossover_probability": 0.1,
    # Probability that two networks will undergo network-mix crossover.
    "network_mix_crossover_probability": 0.15
}