from lightning import LightningModule
from torch.distributions import Categorical
from torch.optim import AdamW

from loss import PartialMultivariateLogRankLoss


class TrainableNetwork(LightningModule):
    """PyTorch Lightning wrapper for training survival analysis models.

    Args:
        model (nn.Module): PyTorch model class to wrap.
        model_params (dict): Parameters for initializing the model.
        learning_rate (float): Learning rate for the optimizer. Default: 0.0001.
        weight_decay (float): Weight decay for the optimizer. Default: 1.
        class_imbalance_penalty (float): Weight for the penalty term in the loss function.
            Default: 0.1.
    """
    def __init__(self,
                 model,
                 model_params: dict,
                 learning_rate=0.0001,
                 weight_decay=1,
                 class_imbalance_penalty=.1,
                 ):
        super(TrainableNetwork, self).__init__()
        self.model = model(**model_params)
        self.loss = PartialMultivariateLogRankLoss(penalty_weight=class_imbalance_penalty)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """Training step for the model.

        Args:
            batch (tuple): Batch of data containing features and survival targets.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Loss value for backpropagation.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, *y)
        self.log(name="train_loss", value=loss, prog_bar=True)
        return -loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model.

        Args:
            batch (tuple): Batch of data containing features and survival targets.
            batch_idx (int): Index of the batch.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, *y)
        entropy = Categorical(probs=y_hat).entropy().mean()
        self.log(name="val_loss", value=loss, prog_bar=True)
        self.log(name="val_entropy", value=entropy, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        """Prediction step for the model.

        Args:
            batch (tuple): Batch of data containing features and survival targets.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Model predictions.
        """
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        """Configures the optimizer for training.

        Returns:
            Optimizer: AdamW optimizer with specified learning rate and weight decay.
        """
        return AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
