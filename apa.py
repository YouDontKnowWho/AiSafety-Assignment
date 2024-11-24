# Import necessary libraries
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from datasets import load_dataset
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr.client import NumPyClient
from flwr.common import (
    Context,
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    EvaluateRes,
    FitRes,
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedProx, DifferentialPrivacyClientSideFixedClipping
from flwr.simulation import run_simulation
from collections import OrderedDict
from typing import Dict, Tuple, List, Optional
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import INFO

from flwr.client.mod import fixedclipping_mod

# Additional imports for plotting and data saving
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Define constants
NUM_CLIENTS = 5
NUM_ROUNDS = 10
BATCH_SIZE = 32

NOISE_MULTIPLIER = 0.5  # Noise multiplier for differential privacy
CLIPPING_NORM = 5.0     # Clipping norm for client-side clipping

# Global variables (will be updated per scenario)
ATTACKER_IDS = []  # List of attacker client IDs
USE_IID = True     # Whether to use IID partitioning

# Function to run simulation for a given scenario
def run_simulation_scenario(ATTACKER_IDS, USE_IID):
    # Define partitioner based on IID or non-IID
    if USE_IID:
        # IID Partitioning
        partitioner = IidPartitioner(num_partitions=NUM_CLIENTS)
    else:
        # Non-IID Partitioning using Dirichlet distribution
        alpha = 1.0  # Adjust alpha for desired heterogeneity
        partitioner = DirichletPartitioner(
            num_partitions=NUM_CLIENTS, alpha=alpha, partition_by="label"
        )

    # Load the CIFAR-10 dataset and partition it
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})

    def get_cifar10_dataloaders(cifar10_dataset, batch_size: int):
        """
        Function to handle CIFAR-10 data loaders.
        Applies appropriate transformations for CIFAR-10 images.
        """
        # CIFAR-10 normalization parameters
        pytorch_transforms = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Prepare transformation functions
        def apply_transforms(batch):
            """Apply transforms to the partition from FederatedDataset."""
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        cifar10_dataset = cifar10_dataset.with_transform(apply_transforms)

        # Construct PyTorch dataloader
        dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    # Define the neural network model suitable for CIFAR-10
    class Net(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)  # Input channels = 3 for CIFAR-10
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjusted for 32x32 images
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Functions to set and get model parameters
    def set_params(model, parameters):
        """Replace model parameters with those passed as parameters."""
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def get_params(model):
        """Extract model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Training function with label flipping for attackers and proximal term
    def train(net, trainloader, optimizer, device="cpu", is_attacker=False, global_params=None, mu=0.1):
        """Train the network on the training set with optional proximal term."""
        criterion = torch.nn.CrossEntropyLoss()
        net.to(device)
        net.train()
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            if is_attacker:
                # Flip labels for attackers (simple label flipping)
                labels = (labels + 1) % 10
            outputs = net(images)
            loss = criterion(outputs, labels)
            if global_params is not None:
                # Add proximal term
                proximal_loss = 0.0
                for param, global_param in zip(net.parameters(), global_params.values()):
                    proximal_loss += torch.norm(param - global_param.to(device))**2
                loss += (mu / 2) * proximal_loss
            loss.backward()
            optimizer.step()

    # Testing function with metric calculations
    def test(net, testloader, device):
        """Validate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        net.to(device)
        net.eval()
        correct, loss = 0, 0.0
        all_preds = []
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch["img"].to(device), batch["label"].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        accuracy = correct / len(testloader.dataset)

        # Compute metrics
        kappa = cohen_kappa_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        # Check if there are at least two classes in y_true
        unique_classes = np.unique(all_labels)
        if len(unique_classes) > 1:
            try:
                all_labels_bin = label_binarize(all_labels, classes=list(range(10)))
                all_outputs_array = np.array(all_outputs)
                roc_auc = roc_auc_score(all_labels_bin, all_outputs_array, average='macro', multi_class='ovr')
            except ValueError:
                roc_auc = float('nan')  # Assign NaN if ROC AUC cannot be computed
        else:
            roc_auc = float('nan')  # Assign NaN if only one class is present

        metrics = {
            "accuracy": accuracy,
            "kappa": kappa,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }
        return loss, accuracy, metrics

    # Define the FlowerClient class
    class FlowerClient(NumPyClient):
        def __init__(self, trainloader, valloader, is_attacker=False) -> None:
            super().__init__()
            self.trainloader = trainloader
            self.valloader = valloader
            self.model = Net(num_classes=10)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.is_attacker = is_attacker

        def fit(self, parameters, config):
            """Train the model locally with FedProx proximal term."""
            # Set model parameters
            set_params(self.model, parameters)

            # Extract global parameters from config
            global_params = OrderedDict(
                (k, torch.tensor(v)) for k, v in zip(self.model.state_dict().keys(), parameters)
            )

            # Get proximal_mu from config
            mu = config.get("proximal_mu", 0.1)  # Default to 0.1 if not set

            # Define the optimizer
            optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

            # Train the model with proximal term
            train(
                self.model,
                self.trainloader,
                optim,
                self.device,
                is_attacker=self.is_attacker,
                global_params=global_params,
                mu=mu  # Proximal term coefficient
            )

            # Return updated parameters without client-side clipping
            return get_params(self.model), len(self.trainloader.dataset), {}

        def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
            """Evaluate the model locally."""
            set_params(self.model, parameters)
            loss, accuracy, metrics = test(self.model, self.valloader, self.device)
            return float(loss), len(self.valloader.dataset), metrics

    # Define the client function
    def client_fn(context: Context):
        """Create a Flower client representing a participant in the federated learning."""
        partition_id = int(context.node_config["partition-id"])
        partition = fds.load_partition(partition_id, "train")
        # Partition into train/validation
        partition_train_val = partition.train_test_split(test_size=0.1, seed=42)
        # Get dataloaders
        trainloader = get_cifar10_dataloaders(partition_train_val["train"], batch_size=BATCH_SIZE)
        valloader = get_cifar10_dataloaders(partition_train_val["test"], batch_size=BATCH_SIZE)
        # Determine if the client is an attacker
        is_attacker = partition_id in ATTACKER_IDS
        # Instantiate the client
        client = FlowerClient(trainloader=trainloader, valloader=valloader, is_attacker=is_attacker)
        # Return the client without passing mods here
        return client

    # Define function to provide fit config with proximal_mu
    def fit_config(rnd: int):
        """Return training configuration dict for each round."""
        return {"proximal_mu": 0.1}  # Set proximal_mu value

    # Define custom strategy to log metrics
    class CustomFedProx(FedProx):
        def __init__(self, *args, **kwargs):
            self.rounds_list = kwargs.pop('rounds_list')
            self.accuracy_list = kwargs.pop('accuracy_list')
            self.kappa_list = kwargs.pop('kappa_list')
            self.f1_list = kwargs.pop('f1_list')
            self.roc_auc_list = kwargs.pop('roc_auc_list')
            proximal_mu = kwargs.pop('proximal_mu', 0.1)  # Extract proximal_mu
            super().__init__(*args, proximal_mu=proximal_mu, **kwargs)  # Pass proximal_mu to FedProx

        def aggregate_evaluate(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[BaseException],
        ):
            """Aggregate evaluation results using weighted average and log metrics per round."""
            if not results:
                return None, {}

            # Use weighted average to aggregate metrics
            num_examples_total = sum([r[1].num_examples for r in results])

            # Initialize sums
            accuracy_sum = 0.0
            kappa_sum = 0.0
            f1_sum = 0.0
            roc_auc_sum = 0.0
            roc_auc_weight = 0

            for _, evaluate_res in results:
                accuracy_sum += evaluate_res.metrics.get("accuracy", 0.0) * evaluate_res.num_examples
                kappa_sum += evaluate_res.metrics.get("kappa", 0.0) * evaluate_res.num_examples
                f1_sum += evaluate_res.metrics.get("f1_score", 0.0) * evaluate_res.num_examples
                roc_auc = evaluate_res.metrics.get("roc_auc", float('nan'))
                if not np.isnan(roc_auc):
                    roc_auc_sum += roc_auc * evaluate_res.num_examples
                    roc_auc_weight += evaluate_res.num_examples

            # Compute weighted averages
            accuracy = accuracy_sum / num_examples_total
            kappa = kappa_sum / num_examples_total
            f1 = f1_sum / num_examples_total
            roc_auc = roc_auc_sum / roc_auc_weight if roc_auc_weight > 0 else float('nan')

            # Log metrics
            log(INFO, f"Round {rnd} evaluation metrics:")
            log(INFO, f"Accuracy: {accuracy:.4f}")
            log(INFO, f"Kappa: {kappa:.4f}")
            log(INFO, f"F1 Score: {f1:.4f}")
            if not np.isnan(roc_auc):
                log(INFO, f"ROC AUC: {roc_auc:.4f}")
            else:
                log(INFO, f"ROC AUC: Undefined (only one class present in y_true)")

            # Store metrics
            self.rounds_list.append(rnd)
            self.accuracy_list.append(accuracy)
            self.kappa_list.append(kappa)
            self.f1_list.append(f1)
            self.roc_auc_list.append(roc_auc if not np.isnan(roc_auc) else 0.0)

            # Return aggregated loss and metrics
            return super().aggregate_evaluate(rnd, results, failures)

    # Define the server function
    def server_fn(context: Context, rounds_list, accuracy_list, kappa_list, f1_list, roc_auc_list):
        """Define the server strategy with differential privacy and metrics collection."""
        # Instantiate the model
        model = Net(num_classes=10)
        ndarrays = get_params(model)
        # Convert model parameters to flwr.common.Parameters
        global_model_init = ndarrays_to_parameters(ndarrays)

        # Define the base strategy (FedProx)
        base_strategy = FedProx(
            fraction_fit=1.0,        # All clients participate in training
            fraction_evaluate=1.0,   # All clients participate in evaluation
            initial_parameters=global_model_init,  # Initialized global model
            proximal_mu=0.1,         # Proximal term coefficient for FedProx
            on_fit_config_fn=fit_config  # Function to provide fit config
        )

        # Wrap the base strategy with DifferentialPrivacyClientSideFixedClipping
        dp_strategy = DifferentialPrivacyClientSideFixedClipping(
            strategy=base_strategy,
            noise_multiplier=NOISE_MULTIPLIER,
            clipping_norm=CLIPPING_NORM,
            num_sampled_clients=NUM_CLIENTS
        )

        # Initialize the custom strategy with metrics lists
        custom_strategy = CustomFedProx(
            strategy=dp_strategy,
            rounds_list=rounds_list,
            accuracy_list=accuracy_list,
            kappa_list=kappa_list,
            f1_list=f1_list,
            roc_auc_list=roc_auc_list,
            proximal_mu=0.1
        )

        # Construct ServerConfig
        config = ServerConfig(num_rounds=NUM_ROUNDS)

        # Wrap everything into a ServerAppComponents object
        return ServerAppComponents(strategy=custom_strategy, config=config)

    # Define your ServerApp and ClientApp
    def create_server_app(rounds_list, accuracy_list, kappa_list, f1_list, roc_auc_list):
        """Create a ServerApp with the custom strategy for metrics collection."""
        return ServerApp(
            server_fn=lambda context: server_fn(
                context,
                rounds_list,
                accuracy_list,
                kappa_list,
                f1_list,
                roc_auc_list
            )
        )

    # Run simulation based on defense
    def run_simulation_with_defense():
        """Run the federated learning simulation with differential privacy."""
        # Initialize metrics lists
        rounds_list = []
        accuracy_list = []
        kappa_list = []
        f1_list = []
        roc_auc_list = []

        # Create server and client apps
        server_app = create_server_app(
            rounds_list,
            accuracy_list,
            kappa_list,
            f1_list,
            roc_auc_list
        )
        client_app = ServerApp(
            server_fn=lambda context: server_fn(
                context,
                rounds_list,
                accuracy_list,
                kappa_list,
                f1_list,
                roc_auc_list
            )
        )

        # Note: The client mods are passed during ClientApp creation
        from flwr.client import ClientApp
        client_app = ClientApp(client_fn=client_fn, mods=[fixedclipping_mod])

        # Run the simulation
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=NUM_CLIENTS,
            backend_name="ray",
            verbose_logging=True,
        )

        # After the simulation, collect the metrics
        metrics = {
            "rounds": rounds_list,
            "accuracy": accuracy_list,
            "kappa": kappa_list,
            "f1_score": f1_list,
            "roc_auc": roc_auc_list,
        }

        return metrics

    # Function to run all scenarios
    def run_all_scenarios():
        """Run all defined federated learning scenarios and collect metrics."""
        scenarios = [
            # Uncomment and modify as needed
            # {"name": "Baseline_IID", "ATTACKER_IDS": [], "USE_IID": True},
            # {"name": "One_Attacker_IID", "ATTACKER_IDS": [0], "USE_IID": True},
            # {"name": "Two_Attackers_IID", "ATTACKER_IDS": [0, 1], "USE_IID": True},
            # {"name": "Baseline_NonIID", "ATTACKER_IDS": [], "USE_IID": False},
            # {"name": "One_Attacker_NonIID", "ATTACKER_IDS": [0], "USE_IID": False},
            {"name": "Two_Attackers_NonIID", "ATTACKER_IDS": [0, 1], "USE_IID": False},
        ]

        results = {}

        for scenario in scenarios:
            print(f"\nRunning Scenario: {scenario['name']}")
            # Update global ATTACKER_IDS and USE_IID
            global ATTACKER_IDS
            ATTACKER_IDS = scenario["ATTACKER_IDS"]
            global USE_IID
            USE_IID = scenario["USE_IID"]
            # Run simulation with defense (always using DP in this case)
            metrics = run_simulation_with_defense()
            results[scenario["name"]] = metrics

            # Plot the metrics
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.plot(metrics["rounds"], metrics["accuracy"], marker='o', color='blue')
            plt.title(f"Accuracy Over Rounds ({scenario['name']})")
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.plot(metrics["rounds"], metrics["kappa"], marker='o', color='green')
            plt.title(f"Kappa Over Rounds ({scenario['name']})")
            plt.xlabel('Round')
            plt.ylabel('Kappa')
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.plot(metrics["rounds"], metrics["f1_score"], marker='o', color='red')
            plt.title(f"F1 Score Over Rounds ({scenario['name']})")
            plt.xlabel('Round')
            plt.ylabel("F1 Score")
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(metrics["rounds"], metrics["roc_auc"], marker='o', color='purple')
            plt.title(f"ROC AUC Over Rounds ({scenario['name']})")
            plt.xlabel('Round')
            plt.ylabel("ROC AUC")
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            # Save metrics to CSV
            df = pd.DataFrame(metrics)
            df.to_csv(f"{scenario['name']}_metrics.csv", index=False)

        return results

    # Run all scenarios
    if __name__ == "__main__":
        results = run_all_scenarios()
