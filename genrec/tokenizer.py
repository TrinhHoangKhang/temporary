from logging import getLogger
from genrec.dataset import AbstractDataset

'''
Tokenize the splited dataset
INPUT (result after dataset.split()):
{
    'train': HF Dataset([
        {'user': 'A2OKNI5Z', 'item_seq': ['B001A3E5A4', 'B002BVQY1C', 'B003K2WJVQ']},
        {'user': 'A3K2L9X1', 'item_seq': ['C001M2P5X7', 'C002N3Q6Y8']},
        ...
    ]),
    'val': HF Dataset([...]),
    'test': HF Dataset([...]),
}
OUTPUT (after tokenizer.tokenize()):
{
    'train': HF Dataset([
        {
            'input_ids': [[1, 5, 20, 0, 0, ...], ...],      # For each sample: Sequence of item IDs (NOT SEMANTIC IDs)
            'attention_mask': [[1, 1, 1, 0, 0, ...], ...],  # For each sample: show which item are real vs padding
            'labels': [[10, 15, 0, -100, -100, ...], ...],  # For each sample: target item ID
            'seq_lens': [3, 2, ...],                        # For each sample: sequence length
            ... and more depend on the model
        },
        ...
    ]),
    'val': HF Dataset([...]),   
    'test': HF Dataset([...]),
}

Note: The tokenize() output does not concern semantic IDs. The tokenizer save the item2sem mapping and then the model will look up the semantic IDs during training 
'''
class AbstractTokenizer:

    def __init__(self, config: dict, dataset: AbstractDataset):
        self.config = config
        self.logger = getLogger()
        # Must be set by subclass. Evaluator uses this for optional EOS trimming.
        self.eos_token = None
        self.collate_fn = {'train': None, 'val': None, 'test': None}

    def _init_tokenizer(self):
        """
        Optional setup hook for tokenizer-specific preprocessing.

        Typical uses:
            - build/load semantic ID mappings
            - load cache files
            - prepare vocab tables

        Not called automatically by the framework; call it in subclass `__init__`
        when needed.
        """
        raise NotImplementedError('Tokenizer initialization not implemented.')

    def tokenize(self, datasets):
        """
        Convert raw split datasets into model-ready tokenized datasets.

        Args:
            datasets (dict): Dict with keys `train`, `val`, `test`, each value
                a Hugging Face Dataset containing at least `user` and `item_seq`.

        Returns:
            dict: Same split keys with tokenized columns consumed by the model.

        Implementation guidance:
            - Keep split keys unchanged.
            - Ensure output fields match model `forward()/generate()` needs.
            - Add evaluation helpers (e.g., `idx`) if downstream expects them.
        """
        raise NotImplementedError('Tokenization not implemented.')

    @property
    def vocab_size(self):
        """
        Size of vocabulary used by model embedding layers.
        """
        raise NotImplementedError('Vocabulary size not implemented.')

    @property
    def padding_token(self):
        """
        Padding token ID. Framework convention is 0.
        """
        return 0

    @property
    def max_token_seq_len(self):
        """
        Maximum token sequence length produced by this tokenizer.
        """
        raise NotImplementedError('Maximum token sequence length not implemented.')

    def log(self, message, level='info'):
        """
        Log helper that respects distributed training setup.
        """
        from genrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)
