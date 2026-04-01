"""
Simple test script to see the actual output of tokenize() method.

Usage:
    python test-script/02_tokenizing.py \
  --category Beauty \
  --max_users 100 \
  --max_items 500 \
  --max_item_seq_len 10 \
  --sent_emb_model sentence-transformers/sentence-t5-base
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
    parser.add_argument(
        '--sent_emb_model',
        type=str,
        default='text-embedding-3-large',
        choices=['text-embedding-3-large', 'sentence-transformers/sentence-t5-base'],
        help='Sentence embedding model to use'
    )
    parser.add_argument(
        '--openai_api_key',
        type=str,
        default=None,
        help='OpenAI API key (required if using text-embedding-3-large)'
    )
    parser.add_argument(
        '--max_users',
        type=int,
        default=None,
        help='Limit dataset to N users (for faster testing). None = use all users'
    )
    parser.add_argument(
        '--max_items',
        type=int,
        default=None,
        help='Limit dataset to N items (for faster testing). None = use all items'
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
    logger.info(f'Sent emb model: {args.sent_emb_model}')
    
    # Validate OpenAI API key if using OpenAI model
    if 'text-embedding-3' in args.sent_emb_model and args.openai_api_key is None:
        import os
        args.openai_api_key = os.environ.get('OPENAI_API_KEY')
        if args.openai_api_key is None:
            logger.warning('OpenAI API key not provided and OPENAI_API_KEY env var not set')
            logger.warning('You can provide it with --openai_api_key or set OPENAI_API_KEY env var')
    
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
        
        # Filter dataset if requested (for faster testing)
        if args.max_users or args.max_items:
            print_separator("FILTERING DATASET FOR FASTER TESTING")
            original_users = dataset.n_users
            original_items = dataset.n_items
            
            if args.max_users:
                # Filter to max_users
                user_list = list(dataset.id_mapping['id2user'])[1:args.max_users+1]  # Skip [PAD] at index 0
                
                # Update id_mapping
                dataset.id_mapping['user2id'] = {'[PAD]': 0}
                dataset.id_mapping['id2user'] = ['[PAD]']
                for i, user in enumerate(user_list, 1):
                    dataset.id_mapping['user2id'][user] = i
                    dataset.id_mapping['id2user'].append(user)
                
                # Filter split data
                dataset.all_item_seqs = {
                    user: seqs for user, seqs in dataset.all_item_seqs.items()
                    if user in user_list
                }
                
                # Recreate split_data with filtered users
                dataset.split_data['train'] = [
                    item for item in dataset.split_data['train']
                    if item['user'] in user_list
                ]
                dataset.split_data['val'] = [
                    item for item in dataset.split_data['val']
                    if item['user'] in user_list
                ]
                dataset.split_data['test'] = [
                    item for item in dataset.split_data['test']
                    if item['user'] in user_list
                ]
                logger.info(f'Filtered users: {original_users} → {len(user_list) + 1} (including [PAD])')
            
            if args.max_items:
                # Filter to max_items
                item_list = list(dataset.id_mapping['id2item'])[1:args.max_items+1]  # Skip [PAD] at index 0
                
                # Update id_mapping
                dataset.id_mapping['item2id'] = {'[PAD]': 0}
                dataset.id_mapping['id2item'] = ['[PAD]']
                for i, item in enumerate(item_list, 1):
                    dataset.id_mapping['item2id'][item] = i
                    dataset.id_mapping['id2item'].append(item)
                
                # Filter sequences to only include these items
                def filter_item_seq(seq):
                    return [item for item in seq if item in item_list]
                
                # Update all_item_seqs
                new_all_item_seqs = {}
                for user, seqs in dataset.all_item_seqs.items():
                    filtered_seq = filter_item_seq(seqs)
                    if len(filtered_seq) > 1:
                        new_all_item_seqs[user] = filtered_seq
                dataset.all_item_seqs = new_all_item_seqs
                
                # Update split data
                dataset.split_data['train'] = [
                    {**item, 'item_seq': filter_item_seq(item['item_seq'])}
                    for item in dataset.split_data['train']
                    if len(filter_item_seq(item['item_seq'])) > 1
                ]
                dataset.split_data['val'] = [
                    {**item, 'item_seq': filter_item_seq(item['item_seq'])}
                    for item in dataset.split_data['val']
                    if len(filter_item_seq(item['item_seq'])) > 1
                ]
                dataset.split_data['test'] = [
                    {**item, 'item_seq': filter_item_seq(item['item_seq'])}
                    for item in dataset.split_data['test']
                    if len(filter_item_seq(item['item_seq'])) > 1
                ]
                logger.info(f'Filtered items: {original_items} → {len(item_list) + 1} (including [PAD])')
            
            logger.info(f'After filtering:')
            logger.info(f'  Users: {dataset.n_users}, Items: {dataset.n_items}')
            logger.info(f'  Train samples: {len(dataset.split_data["train"])}')
            logger.info(f'  Val samples: {len(dataset.split_data["val"])}')
            logger.info(f'  Test samples: {len(dataset.split_data["test"])}')
        
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
        
        # Determine embedding model config based on choice
        if 'text-embedding-3-large' in args.sent_emb_model:
            sent_emb_dim = 3072
            sent_emb_pca = 512
            openai_api_key = args.openai_api_key
        else:  # sentence-t5
            sent_emb_dim = 768
            sent_emb_pca = 128
            openai_api_key = None
        
        tokenizer_config = {
            **config,
            # Backbone model config
            'max_item_seq_len': args.max_item_seq_len,
            'num_proc': args.num_proc,
            
            # Semantic ID (OPQ) config - SAME AS OFFICIAL PIPELINE
            'n_codebook': 32,
            'codebook_size': 256,
            'opq_use_gpu': False,
            'opq_gpu_id': 0,
            'faiss_omp_num_threads': 32,
            
            # Sentence embedding config - FROM OFFICIAL PIPELINE
            'sent_emb_model': args.sent_emb_model,
            'sent_emb_dim': sent_emb_dim,
            'sent_emb_pca': sent_emb_pca,
            'sent_emb_batch_size': 512,
            'device': 'cpu',
            'openai_api_key': openai_api_key,
            
            # Required for tokenizer
            'accelerator': None,  # No distributed training for testing
        }
        
        tokenizer = RPGTokenizer(tokenizer_config, dataset)
        logger.info(f'Tokenizer created')
        logger.info(f'  - Vocab size: {tokenizer.vocab_size}')
        logger.info(f'  - Max token seq len: {tokenizer.max_token_seq_len}')
        logger.info(f'  - Number of items: {len(tokenizer.item2id)}')
        logger.info(f'  - Sent emb model: {args.sent_emb_model}')
        logger.info(f'  - Sent emb dim: {sent_emb_dim}')
        logger.info(f'  - Sent emb PCA: {sent_emb_pca}')
        
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
