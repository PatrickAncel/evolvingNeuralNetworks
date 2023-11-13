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
    # The minimum acceptable number of nodes a layer can have.
    "min_layer_size": 5,
    # Standard deviation of variable-wise Gaussian mutation of weights.
    "gaussian_mutation_weight_sigma": 0.05,
    # Standard deviation of variable-wise Gaussian mutation of biases.
    "gaussian_mutation_bias_sigma": 0.05,
    # Probability that a network layer will be resized during mutation.
    "rescale_mutation_probability": 0.20,
    # The standard deviation of the number of nodes added/removed during rescale mutation.
    "rescale_mutation_sigma": 100,
    "blend_crossover_alpha": 0.5,
    "simulated_binary_crossover_eta": 2.0,
    # Valid values are "blx" and "sbx"
    "crossover_type": "sbx"
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