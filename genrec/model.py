import torch.nn as nn

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class AbstractModel(nn.Module):
    """
    Abstract base class for recommendation models.
    Must fullfill these contracts with trainer/evaluator:
    
    1) Implement the forward() method to return an object with a .loss attribute for training:
       `outputs = model(batch)`
       - subclass implements `forward(batch)` (inherited from nn.Module)
       - returned object must expose `.loss` scalar tensor

    2) Implement a generate() method for evaluation/inference:
       `preds = model.generate(batch, n_return_sequences=k)`
       - must return ranked predictions consumable by evaluator
       - typical shape: `(B, K)` where:
           B = batch size
           K = number of returned candidates
    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(AbstractModel, self).__init__()

        self.config = config
        self.dataset = dataset
        self.tokenizer = tokenizer

    @property
    def n_parameters(self):
        """
        Number of trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'Total number of trainable parameters: {total_params}'

    def calculate_loss(self, batch):
        """
        Optional legacy extension point.

        Most models should place their training loss in `forward(batch)` output
        as `.loss`, but this method is kept for compatibility with older designs.
        """
        raise NotImplementedError('calculate_loss method must be implemented.')

    def generate(self, batch, n_return_sequences=1):
        """
        Generate top-k predictions for evaluation/inference.

        Args:
            batch (dict): Tokenized mini-batch.
            n_return_sequences (int): Number of predictions per example.

        Returns:
            Tensor-like predictions, usually shaped `(B, K, L)`.
        """
        raise NotImplementedError('predict method must be implemented.')
