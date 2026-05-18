from torch.nn import Conv2d, MaxPool2d, Linear, Softmax
from torch.nn import Module

from data.datamodules.mnist_surv import MNISTSurvivalDataModule
from net.training import run_training_session as train


class MNISTConvNetwork(Module):
    """Convolutional Neural Network for MNIST survival analysis.
    """
    def __init__(self):
        super(MNISTConvNetwork, self).__init__()

        self.conv1 = Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding="same")
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same")
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = Conv2d(in_channels=32, out_channels=8, kernel_size=2, padding="same")
        self.maxpool3 = MaxPool2d(kernel_size=2, stride=2)

        self.linear = Linear(in_features=8, out_features=3)
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor after softmax activation.
        """
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.linear(x.squeeze())
        x = self.softmax(x)
        return x


def train_model():
    from figures import plot_single_net_prediction
    results = train(datamodule=MNISTSurvivalDataModule,
                    model=MNISTConvNetwork,
                    model_params={},
                    training_params={
                        "learning_rate": 0.001,
                        "weight_decay": 0.01,
                        "batch_size": 64,
                        "k": 1,
                        "class_imbalance_penalty": 0.1,
                    },
                    max_epochs=20,
                    seed=21)
    plot_single_net_prediction(v_num=results['v_num'], model=results["model"].model, dm=results["dm"])


def main():
    train_model()
