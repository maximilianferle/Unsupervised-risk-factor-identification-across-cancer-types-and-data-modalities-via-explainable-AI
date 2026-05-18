from torch.nn import Conv2d, MaxPool2d, Linear, Softmax
from torch.nn import Module

from data.datamodules.lung_surv import LUNGSurvivalDataModule
from net.training import run_training_session as train


class CudaLungConvNetwork(Module):
    """Convolutional Neural Network for NSCLC survival clustering.
    """
    def __init__(self):
        super(CudaLungConvNetwork, self).__init__()

        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=16, padding="same")
        self.maxpool1 = MaxPool2d(kernel_size=4, stride=4)

        self.conv2 = Conv2d(in_channels=32, out_channels=8, kernel_size=8, padding="same")
        self.maxpool2 = MaxPool2d(kernel_size=4, stride=4)

        self.conv3 = Conv2d(in_channels=8, out_channels=8, kernel_size=8, padding="same")
        self.maxpool3 = MaxPool2d(kernel_size=4, stride=4)

        self.conv4 = Conv2d(in_channels=8, out_channels=8, kernel_size=8, padding="same")
        self.maxpool4 = MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding="same")
        self.maxpool5 = MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = Conv2d(in_channels=4, out_channels=4, kernel_size=2, padding="same")
        self.maxpool6 = MaxPool2d(kernel_size=2, stride=2)

        self.linear = Linear(in_features=4, out_features=2)
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """Forward hook of the network.

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
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.conv6(x)
        x = self.maxpool6(x)
        x = self.linear(x.squeeze())
        x = self.softmax(x)
        return x


def train_k_models():
    from figures.vis_net_prediction import plot_all_net_predictions

    n_classes = 2
    v_nums = [train(
        datamodule=LUNGSurvivalDataModule,
        model=CudaLungConvNetwork,
        model_params={},
        training_params={
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "batch_size": 32,
            "k": k,
            "class_imbalance_penalty": .1,
        },
        max_epochs=50,
        seed=26,
        use_cached_dm=False,
    ).get('v_num') for k in range(1, 6)]
    plot_all_net_predictions(v_nums=v_nums, n_classes=n_classes, dm_cls=LUNGSurvivalDataModule, calc_weights=True,
                             time_scale="days", ylabel="Overall Survival [a.u.]", partition="val_dataset", )


def main():
    train_k_models()


if __name__ == '__main__':
    main()
