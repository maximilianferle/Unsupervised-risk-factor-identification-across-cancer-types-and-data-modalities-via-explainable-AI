from torch.nn import Linear, Softmax
from torch.nn import Module

from data.datamodules.synthetic_datamodule import SyntheticDataModule
from net.training import run_training_session as train


class MultiLayerPerceptron(Module):
    """Multi-Layer Perceptron for survival clustering on synthetic data.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        output_size (int): Size of the output layer.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 ):
        super(MultiLayerPerceptron, self).__init__()
        self.hidden_size = hidden_size
        self.fc_input = Linear(input_size, hidden_size)
        self.fc_hidden = Linear(hidden_size, output_size)
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


def train_model():
    from figures import plot_single_net_prediction
    results = train(datamodule=SyntheticDataModule,
                    model=MultiLayerPerceptron,
                    model_params={
                        "input_size": 3,
                        "hidden_size": 16,
                        "output_size": 3,
                    },
                    training_params={
                        "learning_rate": 0.01,
                        "weight_decay": 0.01,
                        "batch_size": 32,
                        "k": 1,
                        "class_imbalance_penalty": 1,
                    },
                    max_epochs=50,
                    seed=17)
    plot_single_net_prediction(v_num=results['v_num'], model=results["model"].model, dm=results["dm"])


def main():
    train_model()
