import torch


class Evaluator:
    """
    Batch-level ranking metric evaluator.

    Supported metrics:
        - Recall@K
        - NDCG@K

    Input contract:
        preds  : torch.LongTensor with shape (B, K, L)
        labels : torch.LongTensor with shape (B, L)

    Matching rule:
        A prediction is correct only when predicted token sequence exactly
        equals the ground-truth sequence (after optional EOS trimming on label).

    Special case:
        Some models may return `(preds, n_visited_items)`; this evaluator accepts
        both plain tensor predictions and that tuple form.
    """

    def __init__(self, config, tokenizer):
        """
        Args:
            config (dict): Must contain `metrics` and `topk`.
            tokenizer: Must expose `eos_token` (and optionally `item2tokens`).
        """
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        # EOS is used to trim padded label suffix for sequence-level comparison.
        self.eos_token = self.tokenizer.eos_token
        # Largest requested K; incoming preds must contain exactly this many ranks.
        self.maxk = max(config['topk'])

    def calculate_pos_index(self, preds, labels):
        """
        Build a boolean hit matrix over ranked predictions.

        Returns:
            torch.BoolTensor of shape (B, maxK), where True marks the first rank
            whose predicted sequence exactly matches the ground-truth sequence.
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            # Ignore everything after EOS in labels (common when labels are padded).
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                # Exact sequence equality; partial token overlap does not count.
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        """Per-example Recall@K (0 or 1 under single-target assumption)."""
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        """
        Per-example NDCG@K under single relevant target assumption.

        With one relevant target, NDCG reduces to discounted hit score at hit rank.
        """
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def calculate_metrics(self, preds, labels):
        """
        Compute all configured metrics for a batch.

        Args:
            preds: Either
                - Tensor with shape (B, K, L), or
                - Tuple `(preds_tensor, n_visited_items)` where `n_visited_items`
                  is a per-example tensor used for efficiency reporting.
            labels: Tensor with shape (B, L).

        Returns:
            dict[str, torch.Tensor]:
                metric tensors keyed as `recall@k`, `ndcg@k`, plus
                `n_visited_items`.
        """
        # Some decoding implementations return extra search-efficiency metadata.
        if isinstance(preds, tuple):
            preds, n_visited_items = preds
        else:
            # Fallback: assume full-catalog scoring when metadata is unavailable.
            n_visited_items = torch.FloatTensor([len(self.tokenizer.item2tokens)] * preds.shape[0])
        results = {}
        # Compute the hit matrix once and reuse for all ranking metrics.
        pos_index = self.calculate_pos_index(preds, labels)
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        # Keep this separate from ranking metrics for easy reporting.
        results['n_visited_items'] = n_visited_items
        return results
