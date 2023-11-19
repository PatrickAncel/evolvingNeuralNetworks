'''
A solution is a sequence of array pairs (W,b):
    W: Weight Matrix
    b: bias vector

    Output: activation(Wx + b)
    W.shape = (this_layer_length, previous_layer_length)
    x.shape = (previous_layer_length, 1)
    b.shape = (this_layer_length, 1)

    The size of a layer is determined by two parameters:
    *   this_layer_length       (len1)
    *   previous_layer_length   (len0)

    It is a constraint on the network that len0[i + 1] = len1[i]
    Neighboring layers must have compatible dimensions.
'''

import numpy as np
import random

global_parameters = {
    # NETWORK PARAMETERS
    ######################################################################################
    # The minimum acceptable number of nodes a layer can have.
    "min_layer_size": 5,
    # The mean size of a randomly generated neural network layer.
    "initial_layer_size_mean": 400,
    # The standard deviation of the size of a randomly generated neural network layer.
    "initial_layer_size_sigma": 100,
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
    # Valid values: "blx", "sbx"
    "layer_level_crossover_type": "sbx",
    # NETWORK-LEVEL MUTATION AND CROSSOVER PARAMETERS
    ######################################################################################
    # Probability that a network layer will be added or removed during mutation.
    "layer_insertion_removal_probability": 0.05,
    # Probability that a pair of layers will be swapped during mutation.
    "layer_swap_probability": 0.05,
    # Probability that two networks will undergo network-split crossover.
    "network_split_crossover_probability": 0.1,
    # Probability that two networks will undergo network-mix crossover.
    "network_mix_crossover_probability": 0.15
}

def gen_no_duplicates(count, maximum):
    '''Generates 'count' distinct indices between 0 and maximum, not including maximum.'''
    indices = []
    choices = [i for i in range(maximum)]
    for i in range(count):
        # Chooses an index from the list.
        index = random.choice(choices)
        # Removes the index from the list of choices.
        choices.remove(index)
        # Adds the index to the list of selected indices.
        indices.append(index)
    return indices

def gen_with_duplicates(count, maximum):
    '''Generates 'count' indices between 0 and maximum, not including maximum.'''
    indices = []
    for i in range(count):
        # Chooses an index.
        index = random.choice(range(maximum))
        # Adds the index to the list of selected indices.
        indices.append(index)
    return indices

class Layer():
    def __init__(self, this_layer_length, previous_layer_length):
        self.len1 = this_layer_length
        self.len0 = previous_layer_length
        self.W = np.random.default_rng().normal(0, 1, (self.len1, self.len0))
        self.b = np.zeros((self.len1, 1))
    # TODO: Merge shrink and grow into one function.
    def shrink(self, new_len1, new_len0):
        '''Deletes random rows and columns to achieve the required dimensions.'''
        if new_len1 < self.len1:
            # Calculates the number of rows to remove.
            count_rows_to_remove = self.len1 - new_len1
            # Chooses random indices of rows to remove.
            row_indices = gen_no_duplicates(count_rows_to_remove, self.len1)
            ##row_indices = range(0, self.len1 - new_len1)
            self.W = np.delete(self.W, row_indices, 0)
            # Chooses random biases to remove.
            bias_indices = gen_no_duplicates(count_rows_to_remove, self.len1)
            ##bias_indices = range(0, self.len1 - new_len1)
            self.b = np.delete(self.b, bias_indices, 0)
        if new_len0 < self.len0:
            # Calculates the number of cols to remove.
            count_cols_to_remove = self.len0 - new_len0
            # Chooses random indices of cols to remove.
            col_indices = gen_no_duplicates(count_cols_to_remove, self.len0)
            ##col_indices = range(0, self.len0 - new_len0)
            self.W = np.delete(self.W, col_indices, 1)
        # Redefines the dimensions of the layer.
        (self.len1, self.len0) = self.W.shape
    def grow(self, new_len1, new_len0):
        '''Inserts rows and columns at random to achieve the required dimensions.'''
        if new_len1 > self.len1:
            # Calculates the number of rows to add.
            count_rows_to_add = new_len1 - self.len1
            # Chooses random indices to insert rows at.
            row_indices = gen_with_duplicates(count_rows_to_add, self.len1)
            ##row_indices = range(0, new_len1 - self.len1)
            self.W = np.insert(self.W, row_indices, 0, 0)
            # Chooses random indices to insert biases at.
            bias_indices = gen_with_duplicates(count_rows_to_add, self.len1)
            ##bias_indices = range(0, new_len1 - self.len1)
            self.b = np.insert(self.b, bias_indices, 0, 0)
        if new_len0 > self.len0:
            # Calculates the number of columns to add.
            count_cols_to_add = new_len0 - self.len0
            # Chooses random indices to insert cols at.
            col_indices = gen_with_duplicates(count_cols_to_add, self.len0)
            ##col_indices = range(0, new_len0 - self.len0)
            self.W = np.insert(self.W, col_indices, 0, 1)
        # Redefines the dimensions of the layer.
        (self.len1, self.len0) = self.W.shape
    def rescale(self, new_len1, new_len0):
        self.shrink(new_len1, new_len0)
        self.grow(new_len1, new_len0)
    def scale_to_match(self, other):
        '''Scales self to match the dimensions of the other layer,
        or scales the other layer to match this one.'''
        if random.random() < 0.5:
            self.rescale(other.len1, other.len0)
        else:
            other.rescale(self.len1, self.len0)
    def mutate_gaussian(self):
        '''Mutates each weight and bias using a Gaussian distribution.'''
        weight_sigma = global_parameters["gaussian_mutation_weight_sigma"]
        bias_sigma = global_parameters["gaussian_mutation_bias_sigma"]
        # Generates the offsets to add to the weights and biases.
        weight_offsets = np.random.default_rng().normal(0, weight_sigma, self.W.shape)
        bias_offsets = np.random.default_rng().normal(0, bias_sigma, self.b.shape)
        # Adds the offsets to the weights and biases.
        self.W = self.W + weight_offsets
        self.b = self.b + bias_offsets
    def mutate_rescale(self):
        '''Adds or removes nodes from the layer. Requires repairing the next layer.'''
        if random.random() < global_parameters["rescale_mutation_probability"]:
            sigma = global_parameters["rescale_mutation_sigma"]
            # Generates a new number of nodes in the layer.
            new_len1 = round(np.random.default_rng().normal(self.len1, sigma))
            # Keeps the size of the layer within bounds.
            if new_len1 < global_parameters["min_layer_size"]:
                new_len1 = global_parameters["min_layer_size"]
            self.rescale(new_len1, self.len0)
    def _variable_blend(self, value0, value1):
        '''Performs blend crossover on a single variable with two specified values.'''
        if random.random() < 0.5:
            alpha = global_parameters["blend_crossover_alpha"]
            swap = False
            # If value1 is smaller than value0, swap them.
            if value1 < value0:
                value0, value1 = value1, value0
                swap = True
            difference = alpha * (value1 - value0)
            center = (value0 + value1) / 2
            lower = value0 - difference
            # Generates a child value closer to value0.
            child_value0 = np.random.default_rng().uniform(lower, center)
            # Calculates a child value closer to value1.
            child_deviation = center - child_value0
            child_value1 = center + child_deviation
            # If the parents were swapped, the children need to be swapped
            # to preserve the original order.
            if swap:
                child_value0, child_value1 = child_value1, child_value0
            return (child_value0, child_value1)
        else:
            return (value0, value1)
    def _variable_sbx(self, value0, value1):
        '''Performs simulated binary crossover on a single variable with two specified values.'''
        if random.random() < 0.5:
            eta = global_parameters["simulated_binary_crossover_eta"]
            uniform = np.random.default_rng().uniform(0, 1)
            if uniform <= 0.5:
                beta = (2 * uniform) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - uniform))) ** (1 / (eta + 1))
            child_value0 = 0.5 * ((1 + beta) * value0 + (1 - beta) * value1)
            child_value1 = 0.5 * ((1 - beta) * value0 + (1 + beta) * value1)
            return (child_value0, child_value1)
        else:
            return (value0, value1)
    def crossover_blend(self, other):
        ''' Performs blend crossover on two layers on different networks, after rescaling them to be the same shape.
        Requires repairing either the layer before or after the one that was rescaled.'''
        self.scale_to_match(other)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                # Performs BLX on the weight variables.
                (child0,child1) = self._variable_blend(self.W[i,j], other.W[i,j])
                # Assigns the new values.
                self.W[i,j] = child0
                other.W[i,j] = child1
        for i in range(self.b.shape[0]):
            # Performs BLX on the bias variables.
            (child0,child1) = self._variable_blend(self.b[i,0], other.b[i,0])
            # Assigns the new values.
            self.b[i,0] = child0
            other.b[i,0] = child1
    def crossover_simulated_binary(self, other):
        '''Performs simulated binary crossover on two layers on different networks, after rescaling them to be the same shape.
        Requires repairing either the layer before or after the one that was rescaled.'''
        self.scale_to_match(other)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                # Performs SBX on the weight variables.
                (child0,child1) = self._variable_sbx(self.W[i,j], other.W[i,j])
                # Assigns the new values.
                self.W[i,j] = child0
                other.W[i,j] = child1
        for i in range(self.b.shape[0]):
            # Performs SBX on the bias variables.
            (child0,child1) = self._variable_sbx(self.b[i,0], other.b[i,0])
            # Assigns the new values.
            self.b[i,0] = child0
            other.b[i,0] = child1
    def crossover(self, other):
        '''Performs crossover on two layers on different networks, after rescaling them to be the same shape.
        Requires repairing either the layer before or after the one that was rescaled.'''
        crossover_type = global_parameters["layer_level_crossover_type"]
        if crossover_type == "blx":
            self.crossover_blend(other)
        elif crossover_type == "sbx":
            self.crossover_simulated_binary(other)
        else:
            raise ValueError("Invalid layer-level crossover type.")

class Network:
    def __init__(self, layer_count, input_size, output_size):
        if layer_count < 1:
            raise ValueError(F"Cannot create {layer_count}-layer NN.")
        self.layer_count = layer_count
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        # Generates the size of each layer, including the input layer.
        # The input and output layers will very likely have incorrect sizes.
        mean = global_parameters["initial_layer_size_mean"]
        sigma = global_parameters["initial_layer_size_sigma"]
        layer_sizes = np.random.default_rng().normal(mean, sigma, (layer_count + 1,))
        # Fixes any layers that are too small.
        minimum_sizes = np.ones((layer_count + 1,)) * global_parameters["min_layer_size"]
        layer_sizes = np.maximum(layer_sizes, minimum_sizes)
        # Fixes the sizes of the input and output layers.
        layer_sizes[0] = input_size
        layer_sizes[-1] = output_size
        # Iterates over the layer indices, starting from the first non-input layer.
        for layer_index in range(1, layer_count + 1):
            this_layer_length = round(layer_sizes[layer_index])
            previous_layer_length = round(layer_sizes[layer_index - 1])
            self.layers.append(Layer(this_layer_length, previous_layer_length))
    def weight_view(self):
        return [x.W.shape for x in self.layers]
    def bias_view(self):
        return [x.b.shape for x in self.layers]
    def repair_at(self, layer_index):
        '''Fixes consecutive layers starting at layer_index so that they have compatible dimensions.'''
        # There is no layer after the last layer, so layer_index cannot be the last index.
        if layer_index >= self.layer_count - 1:
            raise ValueError("Cannot use repair_at at the last layer.")
        first_layer = self.layers[layer_index]
        second_layer = self.layers[layer_index + 1]
        # Returns if the layers are already compatible.
        if first_layer.len1 == second_layer.len0:
            return
        # Picks one of the layers to repair. The other is left unchanged.
        repair_first_layer = random.random() < 0.5
        if repair_first_layer:
            first_layer.rescale(second_layer.len0, first_layer.len0)
        else:
            second_layer.rescale(second_layer.len1, first_layer.len1)
    def repair_all(self):
        '''Fixes all layers that have invalid dimensions.'''
        # Fixes the first layer, if it has an incorrect input size.
        if self.layers[0].len0 != self.input_size:
            self.layers[0].rescale(self.layers[0].len1, self.input_size)
        # Fixes the last layer, if it has an incorrect output size.
        if self.layers[-1].len1 != self.output_size:
            self.layers[-1].rescale(self.output_size, self.layers[-1].len0)
        # Iterates all layers except the last, repairing consecutive layers.
        for layer_index in range(len(self.layers) - 1):
            self.repair_at(layer_index)
    def _mutate_insert_remove_layer(self):
        '''Inserts or removes a layer with a certain probability. Requires repair.'''
        if random.random() < global_parameters["layer_insertion_removal_probability"]:
            # Determines whether to add or remove a layer.
            # A layer is always inserted if there is only one layer.
            insert = self.layer_count == 1 or random.randint() < 0.5
            if insert:
                mean = global_parameters["initial_layer_size_mean"]
                sigma = global_parameters["initial_layer_size_sigma"]
                min_layer_size = global_parameters["min_layer_size"]
                # Generates the size of a new layer.
                layer_size = np.random.default_rng().normal(mean, sigma)
                # Fixes the size if it is too small.
                layer_size = np.maximum(layer_size, min_layer_size)
                # Picks a random spot to insert the layer.
                insertion_index = random.randint(0, self.layer_count - 1)
                # Generates the weights and biases of the new layer.
                new_layer = Layer(layer_size, self.layers[insertion_index].len1)
                # Inserts the layer.
                self.layers = self.layers[:insertion_index] + [new_layer] + self.layers[insertion_index:]
                self.layer_count += 1
            else:
                # Picks a random layer to remove.
                removal_index = random.randint(0, self.layer_count - 1)
                # Removes the layer.
                self.layers = self.layers[:removal_index] + self.layers[removal_index + 1:]
                self.layer_count -= 1
    def _mutate_layer_swap(self):
        '''Swaps two random layers with a certain probability. Requires repair.'''
        if random.random() < global_parameters["layer_swap_probability"]:
            if self.layer_count > 1:
                # Picks the first layer to swap.
                layer1_index = random.randint(0, self.layer_count - 1)
                remaining_indices = [i for i in range(self.layer_count) if i != layer1_index]
                # Picks the second layer to swap.
                layer2_index = random.choice(remaining_indices)
                layer1 = self.layers[layer1_index]
                layer2 = self.layers[layer2_index]
                # Swaps the layers.
                self.layers[layer1_index] = layer2
                self.layers[layer2_index] = layer1        
    def mutate(self):
        '''Performs mutation and repair on the network.
        Repair is done once after all mutations are performed.'''
        # Iterates over the layers and mutates them individually.
        for layer in self.layers:
            layer.mutate_gaussian()
            layer.mutate_rescale()
        # Mutates the entire network.
        self._mutate_insert_remove_layer()
        self._mutate_layer_swap()
        # Repairs the network.
        self.repair_all()
    def _network_split_crossover(self, other):
        '''Performs network-split crossover with another network, with a certain probability. Requires repair.'''
        if random.random() < global_parameters["network_split_crossover_probability"]:
            # If either network consists of a single layer, network-split crossover is not possible.
            if self.layer_count == 1 or other.layer_count == 1:
                return
            # Picks a crossover point on this network.
            crossover_point_self = random.randint(1, self.layer_count - 1)
            # Picks a crossover point on the other network.
            crossover_point_other = random.randint(1, other.layer_count - 1)
            # Splits the parents.
            self_segment1 = self.layers[:crossover_point_self]
            self_segment2 = self.layers[crossover_point_self:]
            other_segment1 = other.layers[:crossover_point_other]
            other_segment2 = other.layers[crossover_point_other:]
            # Crosses the parents.
            self.layers = self_segment1 + other_segment2
            other.layers = other_segment1 + self_segment2
            # Corrects the recorded lengths of the networks.
            self.layer_count = len(self.layers)
            other.layer_count = len(other.layers)
    def _network_mix_crossover(self, other):
        '''Performs network-mix crossover with another network, with a certain probability. Requires repair.'''
        if random.random() < global_parameters["network_mix_crossover_probability"]:
            # The mixable length is the length of the region over which it is possible
            # to mix the networks.
            mixable_length = min(self.layer_count, other.layer_count)
            unmixable_segment = []
            # If one of the parents is longer than the other, the mixable segments
            # are the parts of the parents until the last index of the shorter parent.
            # The unmixable segment is whatever is leftover on the longer parent.
            if len(self.layers) > len(other.layers):
                unmixable_segment = self.layers[mixable_length:]
            elif len(other.layers) > len(self.layers):
                unmixable_segment = other.layers[mixable_length:]
            # Truncates the parents so that the unmixable segment is removed.
            self.layers = self.layers[:mixable_length]
            other.layers = other.layers[:mixable_length]
            # Performs crossover on the indivdual layers of the networks.
            for i in range(mixable_length):
                # Performs crossover on the layer.
                self.layers[i].crossover(other.layers[i])
            # Picks one of the children to give the unmixable segment to.
            if random.random() < 0.5:
                self.layers += unmixable_segment
            else:
                other.layers += unmixable_segment
            # Corrects the recorded lengths of the networks.
            self.layer_count = len(self.layers)
            other.layer_count = len(other.layers)
    def crossover(self, other):
        '''Performs crossover with another network and repairs both.
        Repair is done once on both networks after crossover is performed.'''
        self._network_split_crossover(other)
        self._network_mix_crossover(other)
        self.repair_all()
        other.repair_all()