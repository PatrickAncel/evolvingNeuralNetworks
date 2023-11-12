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
    def grow(self, new_len1, new_len0):
        '''Inserts rows and columns at random to achieve the required dimensions.'''
        if new_len1 > self.len1:
            # TODO: Chooses random indices to insert rows at.
            # row_indices = gen_with_duplicates
            row_indices = range(0, new_len1 - self.len1)
            self.W = np.insert(self.W, row_indices, 0, 0)
            # TODO: Chooses random indices to insert biases at.
            bias_indices = range(0, new_len1 - self.len1)
            self.b = np.insert(self.b, bias_indices, 0, 0)
        if new_len0 > self.len0:
            # TODO: Chooses random indices to insert cols at.
            col_indices = range(0, new_len0 - self.len0)
            self.W = np.insert(self.W, col_indices, 0, 1)
    def rescale(self, new_len1, new_len0):
        self.shrink(new_len1, new_len0)
        self.grow(new_len1, new_len0)
    def scale_to_match_other(self, other):
        '''Scales self to match the dimensions of the other layer.'''
    def scale_to_match(self, other):
        '''Scales self to match the dimensions of the other layer,
        or scales the other layer to match this one.'''
        if random.random() < 0.5:
            self.scale_to_match_other(other)
        else:
            other.scale_to_match_other(self)

