"""
Evaluation test script: Load a trained model checkpoint and evaluate it using the Evaluator.

This script demonstrates:
1. Loading a trained model from a checkpoint
2. Preparing test data
3. Getting model predictions
4. Running the evaluator to compute metrics
5. Visualizing what the model predicts for individual examples

Usage:
    python test-script/03_evaluation.py \
        --model_ckpt ckpt/RPG_Beauty_2024.pth \
        --category Beauty \
        --cache_dir ./cache

Or with defaults (requires a trained model first):
    python test-script/03_evaluation.py
"""

import argparse
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014
from genrec.models.RPG.tokenizer import RPGTokenizer
from genrec.models.RPG.model import RPG
from genrec.evaluator import Evaluator


def setup_logging():
    """Setup basic logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )


def print_separator(title):
    """Print a separator with title."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(ckpt_dir):
        return None
    
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
    return os.path.join(ckpt_dir, files[-1]) if files else None


def visualize_predictions(batch, preds, labels, dataset, tokenizer, model, num_examples=3):
    """Visualize what the model predicted for individual examples."""
    print_separator("MODEL PREDICTION EXAMPLES")
    
    batch_size = preds.shape[0]
    num_to_show = min(num_examples, batch_size)
    
    # Build reverse mapping: item_id -> item_str
    id2item = {v: k for k, v in dataset.item2id.items()}
    
    for idx in range(num_to_show):
        print(f"\n--- Example {idx + 1} ---")
        
        # Get ground truth label (the target item to predict)
        label_token_seq = labels[idx].cpu().tolist()
        # Remove EOS and -100 padding
        label_seq = [x for x in label_token_seq if x != -100 and x != tokenizer.eos_token]
        
        print(f"Ground truth target item tokens: {label_seq}")
        if label_seq and len(label_seq) > 0:
            target_item_id = label_seq[0]  # For RPG, target is a single item
            if target_item_id > 0 and target_item_id < len(id2item):
                target_item_str = id2item[target_item_id]
                print(f"Ground truth target item (string): {target_item_str}")
        
        # Get top-K predictions
        top_k_preds = preds[idx].cpu().tolist()[:5]  # Show top 5
        print(f"\nTop-5 predicted items (item IDs):")
        for rank, pred_item_id in enumerate(top_k_preds, 1):
            pred_item_id = int(pred_item_id[0]) if isinstance(pred_item_id, list) else int(pred_item_id)
            if pred_item_id > 0 and pred_item_id < len(id2item):
                pred_item_str = id2item[pred_item_id]
                is_correct = "✓ CORRECT!" if pred_item_id == target_item_id else ""
                print(f"  Rank {rank}: Item {pred_item_id} ({pred_item_str}) {is_correct}")
            else:
                print(f"  Rank {rank}: Invalid item ID {pred_item_id}")
        
        # Show input sequence
        input_ids = batch['input_ids'][idx].cpu().tolist()
        print(f"\nInput sequence (item IDs): {input_ids}")
        print(f"Input sequence (items):")
        for pos, item_id in enumerate(input_ids):
            if item_id == 0:  # Padding
                print(f"  Position {pos}: [PAD]")
            elif item_id > 0 and item_id < len(id2item):
                print(f"  Position {pos}: Item {item_id} ({id2item[item_id]})")


def inspect_model_internals(model, batch, device):
    """Inspect intermediate outputs from the model."""
    print_separator("MODEL INTERNAL OUTPUTS")
    
    print(f"\nInput batch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  seq_lens: {batch['seq_lens'].shape}")
    
    # Move to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Forward pass (get internal states)
    with torch.no_grad():
        outputs = model.forward(batch, return_loss=False)
    
    print(f"\nModel forward pass outputs:")
    print(f"  final_states shape: {outputs.final_states.shape}")
    print(f"    - batch_size: {outputs.final_states.shape[0]}")
    print(f"    - n_pred_heads: {outputs.final_states.shape[2]}")
    print(f"    - embedding_dim: {outputs.final_states.shape[3]}")
    
    # Get the state at the last position
    seq_lens = batch['seq_lens']
    print(f"\nSequence lengths: {seq_lens.cpu().tolist()}")
    
    # Generate predictions
    with torch.no_grad():
        preds = model.generate(batch, n_return_sequences=10)
    
    print(f"\nPrediction output shape: {preds.shape}")
    print(f"  - batch_size: {preds.shape[0]}")
    print(f"  - num_candidates: {preds.shape[1]}")
    print(f"  - token_seq_length: {preds.shape[2] if len(preds.shape) > 2 else 1}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model using the Evaluator'
    )
    parser.add_argument(
        '--model_ckpt',
        type=str,
        default=None,
        help='Path to trained model checkpoint (.pth file). If None, finds latest in ckpt_dir'
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='ckpt',
        help='Directory containing checkpoints (used if model_ckpt not provided)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='Beauty',
        help='Amazon category'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./cache',
        help='Directory to cache dataset'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Evaluation batch size'
    )
    parser.add_argument(
        '--num_candidates',
        type=int,
        default=50,
        help='Number of candidates to generate per example'
    )
    parser.add_argument(
        '--num_examples_to_show',
        type=int,
        default=3,
        help='Number of prediction examples to visualize'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print_separator("EVALUATION TEST SCRIPT")
    logger.info(f'Device: {args.device}')
    logger.info(f'Category: {args.category}')
    logger.info(f'Cache dir: {args.cache_dir}')
    
    # Find checkpoint
    if args.model_ckpt is None:
        args.model_ckpt = find_latest_checkpoint(args.ckpt_dir)
    
    if args.model_ckpt is None:
        logger.error(f'No checkpoint found in {args.ckpt_dir}')
        logger.error('Please train a model first or provide --model_ckpt')
        return 1
    
    if not os.path.exists(args.model_ckpt):
        logger.error(f'Checkpoint not found: {args.model_ckpt}')
        return 1
    
    logger.info(f'Using checkpoint: {args.model_ckpt}')
    
    try:
        # Step 1: Create dataset
        print_separator("STEP 1: Loading Dataset")
        config = {
            'category': args.category,
            'cache_dir': args.cache_dir,
            'metadata': 'sentence',
            'split': 'leave_one_out',
        }
        
        dataset = AmazonReviews2014(config)
        logger.info(f'Dataset loaded: {dataset.n_users} users, {dataset.n_items} items')
        
        # Step 2: Split dataset
        print_separator("STEP 2: Splitting Dataset")
        split_datasets = dataset.split()
        logger.info(f"Test split: {len(split_datasets['test'])} samples")
        
        # Step 3: Create and configure tokenizer
        print_separator("STEP 3: Creating Tokenizer")
        tokenizer_config = {
            **config,
            'max_item_seq_len': 50,
            'num_proc': 1,
            'n_codebook': 32,
            'codebook_size': 256,
            'sent_emb_dim': 768,
            'sent_emb_model': 'sentence-transformers/sentence-t5-base',
            'sent_emb_batch_size': 512,
            'sent_emb_pca': 128,
            'device': args.device,
            'opq_use_gpu': False,
            'opq_gpu_id': 0,
            'faiss_omp_num_threads': 32,
            'accelerator': None,
        }
        
        tokenizer = RPGTokenizer(tokenizer_config, dataset)
        logger.info(f'Tokenizer created: vocab_size={tokenizer.vocab_size}')
        
        # Step 4: Tokenize test split
        print_separator("STEP 4: Tokenizing Test Split")
        tokenized_datasets = tokenizer.tokenize(split_datasets)
        logger.info(f'Test split tokenized: {len(tokenized_datasets["test"])} samples')
        
        # Step 5: Create test dataloader
        print_separator("STEP 5: Creating DataLoader")
        test_dataloader = DataLoader(
            tokenized_datasets['test'],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=tokenizer.collate_fn['test']
        )
        logger.info(f'DataLoader created: {len(test_dataloader)} batches')
        
        # Step 6: Create model and load checkpoint
        print_separator("STEP 6: Loading Model")
        model = RPG(tokenizer_config, dataset, tokenizer)
        model.load_state_dict(torch.load(args.model_ckpt, map_location=args.device))
        model = model.to(args.device)
        model.eval()
        logger.info(f'Model loaded from checkpoint')
        logger.info(model.n_parameters)
        
        # Step 7: Create evaluator
        print_separator("STEP 7: Creating Evaluator")
        evaluator_config = {
            'metrics': ['recall', 'ndcg'],
            'topk': [5, 10],
        }
        evaluator = Evaluator(evaluator_config, tokenizer)
        logger.info(f'Evaluator created: metrics={evaluator_config["metrics"]}, topk={evaluator_config["topk"]}')
        
        # Step 8: Run evaluation on first batch
        print_separator("STEP 8: Running Evaluation on First Batch")
        
        # Get first batch
        first_batch = next(iter(test_dataloader))
        batch_size = first_batch['input_ids'].shape[0]
        logger.info(f'Batch size: {batch_size}')
        
        # Inspect model internals for this batch
        inspect_model_internals(model, first_batch, args.device)
        
        # Generate predictions
        print_separator("STEP 9: Generating Predictions")
        first_batch_device = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in first_batch.items()}
        with torch.no_grad():
            preds = model.generate(first_batch_device, n_return_sequences=args.num_candidates)
        
        logger.info(f'Predictions shape: {preds.shape}')
        logger.info(f'Predictions dtype: {preds.dtype}')
        logger.info(f'First prediction: {preds[0][:3]}')  # Show first 3 predictions for first example
        
        # Run evaluator
        print_separator("STEP 10: Computing Metrics")
        labels = first_batch_device['labels']
        metrics = evaluator.calculate_metrics(preds, labels)
        
        logger.info(f'Metrics computed successfully!')
        for key, value in metrics.items():
            if key != 'n_visited_items':
                logger.info(f'  {key}: {value.mean().item():.4f}')
            else:
                logger.info(f'  {key}: {value.mean().item():.0f}')
        
        # Visualize predictions
        visualize_predictions(first_batch, preds, labels, dataset, tokenizer, model, 
                             num_examples=args.num_examples_to_show)
        
        # Step 11: Full evaluation on all test data
        print_separator("STEP 11: Full Test Set Evaluation")
        logger.info(f'Evaluating on all {len(test_dataloader)} batches...')
        
        all_results = OrderedDict()
        all_results['recall@5'] = []
        all_results['recall@10'] = []
        all_results['ndcg@5'] = []
        all_results['ndcg@10'] = []
        all_results['n_visited_items'] = []
        
        num_batches_processed = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                preds = model.generate(batch, n_return_sequences=args.num_candidates)
                metrics = evaluator.calculate_metrics(preds, batch['labels'])
                
                for key, value in metrics.items():
                    all_results[key].append(value)
                
                num_batches_processed += 1
                if (batch_idx + 1) % 5 == 0:
                    logger.info(f'  Processed {batch_idx + 1}/{len(test_dataloader)} batches')
        
        # Aggregate results
        print_separator("FINAL TEST RESULTS")
        final_results = OrderedDict()
        for metric in ['recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']:
            if all_results[metric]:
                values = torch.cat(all_results[metric])
                final_results[metric] = values.mean().item()
                logger.info(f'{metric}: {final_results[metric]:.4f}')
        
        if all_results['n_visited_items']:
            values = torch.cat(all_results['n_visited_items'])
            final_results['n_visited_items'] = values.mean().item()
            logger.info(f'n_visited_items: {final_results["n_visited_items"]:.0f}')
        
        logger.info('\n✓ Evaluation complete!')
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning('Evaluation interrupted by user.')
        return 1
    except Exception as e:
        logger.error(f'Error during evaluation: {e}', exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
