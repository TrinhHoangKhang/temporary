from logging import getLogger

from genrec.dataset import AbstractDataset


class AbstractTokenizer:
    """
    Abstract base class for all tokenizers in this framework.

    ─────────────────────────────────────────────────────────────────────────
    WHAT THIS CLASS DOES
    ─────────────────────────────────────────────────────────────────────────
    A tokenizer bridges:
        dataset split output (raw IDs)
            {'user': str, 'item_seq': list[str]}
    and
        model-ready tensors (tokenized features)
            {'input_ids': ..., 'attention_mask': ..., 'labels': ..., ...}
            
    ─────────────────────────────────────────────────────────────────────────
    INPUT CONTRACT
    ─────────────────────────────────────────────────────────────────────────
    `tokenize(datasets)` receives a dict with split keys:
        {
            'train': HF Dataset(columns=['user', 'item_seq']),
            'val':   HF Dataset(columns=['user', 'item_seq']),
            'test':  HF Dataset(columns=['user', 'item_seq']),
        }

    Each row contains raw string IDs (not integer token IDs).

    ─────────────────────────────────────────────────────────────────────────
    OUTPUT CONTRACT
    ─────────────────────────────────────────────────────────────────────────
    `tokenize(datasets)` must return:
        {
            'train': HF Dataset(...tokenized columns...),
            'val':   HF Dataset(...tokenized columns...),
            'test':  HF Dataset(...tokenized columns...),
        }

    De facto required fields:
        - `labels` for all splits
        - `idx` for val/test (will be used by evaluator)
        - `input_ids`, `attention_mask`, 'seq_len': needed by model.forward()/generate()

    """

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
