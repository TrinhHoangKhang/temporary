"""
Simple test script to see the actual output of tokenize() method.

Usage:
    python test-script/02_tokenizing.py --category Beauty --cache_dir ./cache
"""

import argparse
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014
from genrec.models.RPG.tokenizer import RPGTokenizer


def setup_logging():
    """Setup basic logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )


def print_separator(title):
    """Print a separator with title."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def inspect_tokenized_output(split_name, tokenized_dataset):
    """Inspect and print the tokenized output."""
    print_separator(f"TOKENIZED {split_name.upper()} SPLIT")
    
    print(f"\nDataset shape: {len(tokenized_dataset)} samples")
    print(f"Columns: {tokenized_dataset.column_names}")
    print(f"Data type: {tokenized_dataset.data_type}")
    
    # Show first sample in detail
    if len(tokenized_dataset) > 0:
        sample = tokenized_dataset[0]
        
        print("\n--- FIRST SAMPLE ---")
        print(f"Number of examples in this sample: {len(sample['input_ids'])}")
        
        # Show each example
        for idx, (input_ids, attn_mask, labels, seq_len) in enumerate(
            zip(sample['input_ids'], sample['attention_mask'], sample['labels'], sample['seq_lens'])
        ):
            print(f"\n  Example {idx + 1}:")
            print(f"    input_ids:      {list(input_ids)}")
            print(f"    attention_mask: {list(attn_mask)}")
            print(f"    labels:         {list(labels)}")
            print(f"    seq_lens:       {seq_len}")
            
            # Show positions with actual labels (not -100)
            active_labels = [i for i, label in enumerate(labels) if label != -100]
            if active_labels:
                print(f"    ✓ Positions with actual labels: {active_labels}")
                print(f"    ✓ Target items to predict: {[labels[i] for i in active_labels]}")


def main():
    parser = argparse.ArgumentParser(
        description='Test tokenizer output'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='Beauty',
        help='Amazon category (e.g., Books, Electronics, Movies_and_TV)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./cache',
        help='Directory to cache dataset'
    )
    parser.add_argument(
        '--metadata_mode',
        type=str,
        default='sentence',
        choices=['none', 'raw', 'sentence'],
        help='How to process metadata'
    )
    parser.add_argument(
        '--max_item_seq_len',
        type=int,
        default=5,
        help='Maximum item sequence length'
    )
    parser.add_argument(
        '--num_proc',
        type=int,
        default=1,
        help='Number of processes for tokenization'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print_separator("TOKENIZER TEST SCRIPT")
    logger.info(f'Category: {args.category}')
    logger.info(f'Cache dir: {args.cache_dir}')
    logger.info(f'Metadata mode: {args.metadata_mode}')
    logger.info(f'Max item seq len: {args.max_item_seq_len}')
    
    try:
        # Step 1: Create dataset
        print_separator("STEP 1: Creating Dataset")
        config = {
            'category': args.category,
            'cache_dir': args.cache_dir,
            'metadata': args.metadata_mode,
            'split': 'leave_one_out',
        }
        
        dataset = AmazonReviews2014(config)
        logger.info(f'Dataset created: {dataset.n_users} users, {dataset.n_items} items')
        logger.info(dataset)
        
        # Step 2: Split dataset
        print_separator("STEP 2: Splitting Dataset")
        split_datasets = dataset.split()
        
        logger.info(f"Train: {len(split_datasets['train'])} samples")
        logger.info(f"Val: {len(split_datasets['val'])} samples")
        logger.info(f"Test: {len(split_datasets['test'])} samples")
        
        # Show example raw data
        print("\n--- EXAMPLE RAW SAMPLE (before tokenization) ---")
        raw_sample = split_datasets['train'][0]
        print(f"user: {raw_sample['user']}")
        print(f"item_seq: {raw_sample['item_seq']}")
        print(f"item_seq length: {len(raw_sample['item_seq'])}")
        
        # Step 3: Create tokenizer
        print_separator("STEP 3: Creating Tokenizer")
        tokenizer_config = {
            **config,
            'max_item_seq_len': args.max_item_seq_len,
            'num_proc': args.num_proc,
            'codebook_size': 256,
            'n_codebook': 4,
            'sent_emb_dim': 384,  # sentence-transformers default
            'sent_emb_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'sent_emb_batch_size': 32,
            'sent_emb_pca': 0,
            'device': 'cpu',
            'opq_use_gpu': False,
            'opq_gpu_id': 0,
            'faiss_omp_num_threads': 16,
            'accelerator': None,  # No distributed training for testing
        }
        
        tokenizer = RPGTokenizer(tokenizer_config, dataset)
        logger.info(f'Tokenizer created')
        logger.info(f'  - Vocab size: {tokenizer.vocab_size}')
        logger.info(f'  - Max token seq len: {tokenizer.max_token_seq_len}')
        logger.info(f'  - Number of items: {len(tokenizer.item2id)}')
        
        # Step 4: Tokenize
        print_separator("STEP 4: Tokenizing Datasets")
        logger.info('Starting tokenization...')
        tokenized_datasets = tokenizer.tokenize(split_datasets)
        
        # Step 5: Inspect outputs
        for split in ['train', 'val', 'test']:
            inspect_tokenized_output(split, tokenized_datasets[split])
        
        # Step 6: Show statistics
        print_separator("TOKENIZATION STATISTICS")
        for split in ['train', 'val', 'test']:
            dataset = tokenized_datasets[split]
            print(f"\n{split.upper()} split:")
            print(f"  Total samples: {len(dataset)}")
            
            # Count total examples (some samples may have multiple examples)
            total_examples = sum(len(sample['input_ids']) for sample in dataset)
            print(f"  Total examples (with multiple per sample): {total_examples}")
            
            # Show sequence length statistics
            all_seq_lens = []
            for sample in dataset:
                all_seq_lens.extend(sample['seq_lens'])
            
            print(f"  Sequence length stats:")
            print(f"    Min: {min(all_seq_lens)}, Max: {max(all_seq_lens)}, Mean: {sum(all_seq_lens)/len(all_seq_lens):.1f}")
        
        print_separator("TOKENIZER TEST COMPLETE")
        logger.info("✓ Test completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning('Test interrupted by user.')
        return 1
    except Exception as e:
        logger.error(f'Error during test: {e}', exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
