from logging import getLogger
from datasets import Dataset


class AbstractDataset:
    """
    Abstract base class for all datasets in this framework.

    ─────────────────────────────────────────────────────────────────────────
    WHAT THIS CLASS DOES
    ─────────────────────────────────────────────────────────────────────────
    This class defines the shared skeleton that every dataset must follow.
    It handles splitting logic and exposes standard statistics. The only thing
    a subclass needs to do is implement `_download_and_process_raw()`, which
    must populate two attributes:
    
        self.all_item_seqs  (dict)  
        self.id_mapping     (dict) 
        self.item2meta      (dict)  
        
    ─────────────────────────────────────────────────────────────────────────
    ABOUT `_download_and_process_raw()`
    ─────────────────────────────────────────────────────────────────────────
    After `_download_and_process_raw()` is called (inside `__init__`), the
    following attributes must be fully populated:

    self.all_item_seqs : dict[raw_user_id (str) -> list[raw_item_id (str)]]
        The complete chronologically-ordered interaction sequence for each user.
        Uses raw string IDs (not integer IDs).
        Example:
            {
                'user_A': ['item_1', 'item_3', 'item_7'],
                'user_B': ['item_2', 'item_5'],
            }

    self.id_mapping : dict with four keys:
        'user2id' : dict[raw_user_id (str) -> int]   # integer IDs start at 1; 0 is reserved for [PAD]
        'item2id' : dict[raw_item_id (str) -> int]   # same convention
        'id2user' : list[str]                        # index = integer ID, so id2user[0] == '[PAD]'
        'id2item' : list[str]                        # same convention
        Example:
            {
                'user2id': {'[PAD]': 0, 'user_A': 1, 'user_B': 2},
                'item2id': {'[PAD]': 0, 'item_1': 1, 'item_2': 2, ...},
                'id2user': ['[PAD]', 'user_A', 'user_B'],
                'id2item': ['[PAD]', 'item_1', 'item_2', ...],
            }

    self.item2meta : dict[raw_item_id (str) -> any] | None
        Optional. Maps each item to its metadata (e.g. a text description string,
        or a dict of features).

    ─────────────────────────────────────────────────────────────────────────
    OUTPUT OF split() — what the tokenizer receives
    ─────────────────────────────────────────────────────────────────────────
    `split()` returns a dict of three Hugging Face Dataset objects:
        {
            'train': Dataset(columns=['user', 'item_seq']),
            'val':   Dataset(columns=['user', 'item_seq']),
            'test':  Dataset(columns=['user', 'item_seq']),
        }

    Each row has:
        'user'     : str              -- raw user string ID
        'item_seq' : list[str]        -- ordered list of raw item string IDs

    The split strategy is leave-one-out by default:
        test  → full sequence  [item_1, ..., item_N]
        val   → all but last   [item_1, ..., item_{N-1}]
        train → all but last 2 [item_1, ..., item_{N-2}]

    The tokenizer is responsible for converting raw string IDs to integer token
    IDs and producing model-specific tensors (input_ids, labels, etc.).
    """

    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config['accelerator']
        self.logger = getLogger()

        # Filled by dataset-specific raw processing.
        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = None

        # Cache for split() output so splitting runs only once.
        self.split_data = None

    def __str__(self) -> str:
        return f'[Dataset] {self.__class__.__name__}\n' \
                f'\tNumber of users: {self.n_users}\n' \
                f'\tNumber of items: {self.n_items}\n' \
                f'\tNumber of interactions: {self.n_interactions}\n' \
                f'\tAverage item sequence length: {self.avg_item_seq_len}'

    @property
    def n_users(self):
        """
        Number of users including `[PAD]`.
        """
        return len(self.user2id)

    @property
    def n_items(self):
        """
        Number of items including `[PAD]`.
        """
        return len(self.item2id)

    @property
    def n_interactions(self):
        """
        Total number of user-item interactions across all user sequences.
        """
        n_inters = 0
        for user in self.all_item_seqs:
            n_inters += len(self.all_item_seqs[user])
        return n_inters

    @property
    def avg_item_seq_len(self):
        """
        Average sequence length per user.
        """
        return self.n_interactions / self.n_users

    @property
    def user2id(self):
        """
        Shortcut for `self.id_mapping['user2id']`.
        """
        return self.id_mapping['user2id']

    @property
    def item2id(self):
        """
        Shortcut for `self.id_mapping['item2id']`.
        """
        return self.id_mapping['item2id']

    def _download_and_process_raw(self):
        """
        [MUST OVERRIDE IN SUBCLASS]

        Download/load/process raw data, then populate at least:
            - self.all_item_seqs
            - self.id_mapping
            - self.item2meta (optional)
        """
        raise NotImplementedError('This method should be implemented in the subclass')

    def _leave_one_out(self):
        """
        Split sequences using leave-one-out policy.

        For a full user sequence `[i1, i2, ..., iN]`:
            - `test` uses `[i1, ..., iN]`
            - `val` uses `[i1, ..., i(N-1)]` if `N > 1`
            - `train` uses `[i1, ..., i(N-2)]` if `N > 2`

        Returned datasets contain raw string IDs (not token IDs).

        Returns:
            dict: {
                'train': Dataset(columns=['user', 'item_seq']),
                'val': Dataset(columns=['user', 'item_seq']),
                'test': Dataset(columns=['user', 'item_seq']),
            }
        """
        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}
        for user in self.all_item_seqs:
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(self.all_item_seqs[user])
            if len(self.all_item_seqs[user]) > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(self.all_item_seqs[user][:-1])
            if len(self.all_item_seqs[user]) > 2:
                datasets['train']['user'].append(user)
                datasets['train']['item_seq'].append(self.all_item_seqs[user][:-2])
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def split(self):
        """
        Build and cache split datasets according to `config['split']`.

        Supported split strategies:
            - 'leave_one_out'
            - 'last_out' (alias of leave-one-out behavior here)

        Returns:
            dict: split datasets for `train`, `val`, `test`.
        """
        if self.split_data is not None:
            return self.split_data

        split_strategy = self.config['split']
        if split_strategy in ['leave_one_out', 'last_out']:
            datasets = self._leave_one_out()
        else:
            raise NotImplementedError(f'Split strategy [{split_strategy}] not implemented.')

        self.split_data = datasets
        return self.split_data

    def log(self, message, level='info'):
        from genrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)
