from torch.nn import Linear, Softmax, Sequential
from torch.nn import Module

from net.training import run_training_session as train


class MyelomaMLP(Module):
    """Multi-Layer Perceptron for myeloma survival analysis.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layers.
        output_size (int): Size of the output layer.
        n_hidden_layers (int): Number of hidden layers. Default: 1.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 n_hidden_layers: int = 1
                 ):
        super(MyelomaMLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc_input = Linear(input_size, hidden_size)
        self.fc_hidden = Sequential(
            *([Linear(hidden_size, hidden_size) for _ in range(n_hidden_layers - 1)]
              + [Linear(hidden_size, output_size)])
        )
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            Tensor: Output tensor after softmax activation.
        """
        x = self.fc_input(x)
        x = self.fc_hidden(x)
        x = self.softmax(x)
        return x


def train_k_models():
    from data.datamodules.datamodule import CoMMpassDataModule
    from figures import plot_all_net_predictions

    n_classes = 3
    v_nums = [train(
        model=MyelomaMLP,
        datamodule=CoMMpassDataModule,
        model_params={
            "input_size": 10,
            "hidden_size": 32,
            "output_size": n_classes,
            "n_hidden_layers": 3
        },
        training_params={
            "learning_rate": 0.001,
            "weight_decay": 1,
            "batch_size": 32,
            "k": i,
            "class_imbalance_penalty": .1
        },
        max_epochs=20,
        seed=34523,
    ).get('v_num') for i in range(1, 6)]
    plot_all_net_predictions(v_nums=v_nums, n_classes=n_classes, dm_cls=CoMMpassDataModule,
                             time_scale="days", ylabel="Overall Survival [a.u.]", partition="val_dataset", )


def main():
    train_k_models()
