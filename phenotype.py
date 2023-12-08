import torch

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
                self.layers.append(torch.nn.Sigmoid())
    
    def forward(self, input):
        output = input.view(-1, self.input_size)
        for layer in self.layers:
            output = layer(output)
        return output