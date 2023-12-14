'''
This module exists to train a neural network using gradient descent.
'''
import torch
import torchvision
import tqdm
import time

data_transform = torchvision.transforms.ToTensor()

dataset_train = torchvision.datasets.CIFAR10("./data", download = True, train = True, transform = data_transform)
dataset_test = torchvision.datasets.CIFAR10("./data", download = True, train = False, transform = data_transform)

dataset_train = torch.utils.data.Subset(dataset_train, range(30_000))

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()

gd_parameters = {
    "layer_count": 4,
    "layer_size": 3072,
    "input_size": 3072,
    "output_size": 10,
    "epoch_count": 25
}

class NetworkStandard(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer_count = gd_parameters["layer_count"]
        layer_size = gd_parameters["layer_size"]
        self.input_size = gd_parameters["input_size"]
        layers = []
        for i in range(layer_count):
            in_features = layer_size
            out_features = layer_size
            if i == 0:
                in_features = self.input_size
            if i == layer_count - 1:
                out_features = gd_parameters["output_size"]
            layers.append(torch.nn.Linear(in_features, out_features))
            if i < layer_count - 1:
                layers.append(torch.nn.ReLU())
        self.layer = torch.nn.Sequential(*layers)
    def forward(self, input):
        output = input.view(-1, self.input_size)
        output = self.layer(output)
        return output

def evaluate_solution(network):
    '''Evaluates the accuracy of a network on the test dataset.'''
    print("Evaluating Solution")
    score = 0
    network.eval()
    with torch.no_grad():
        for batch, label in tqdm.tqdm(dataloader_test):
            output = network(batch)
            choices = torch.argmax(output,dim=1)
            score += (choices==label).sum().item()
    accuracy = score / len(dataloader_test.dataset)
    print(F"Accuracy on Test Set: {accuracy}")
    return accuracy

def save_results(training_losses, accuracy, time_elapsed):
    filename = F"train results/{time.time()}.txt"
    f = open(filename, "w")
    f.write(str(gd_parameters))
    f.write("\n\n")
    f.write(str(training_losses))
    f.write(F"\n\nAccuracy: {accuracy}\n\nTime Elapsed: {time_elapsed}")
    f.close()

def train():
    start_time = time.time()
    epoch_count = gd_parameters["epoch_count"]
    training_losses = [] # Tracks the training loss at each epoch.
    network = NetworkStandard()
    optimizer = torch.optim.Adam(network.parameters())
    print("Beginning Training")
    network.train()
    for i in range(epoch_count):
        print(F"Epoch: {i}")
        total_loss = 0
        for batch, label in tqdm.tqdm(dataloader_train):
            optimizer.zero_grad()
            output = network(batch)
            loss = criterion(output, label)
            total_loss += loss
            loss.backward()
            optimizer.step()
        training_losses.append(total_loss.item())
    accuracy = evaluate_solution(network)
    time_elapsed = time.time() - start_time
    save_results(training_losses, accuracy, time_elapsed)
    print("Done Training")

if __name__ == "__main__":
    train()