"""
Standalone script to download and preprocess the dataset independently.
Mostly for debugging

Usage:
    python test-script/01_download_and_preprocess.py --category Beauty --cache_dir ./cache
"""

import argparse
import logging
from math import log
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014


def setup_logging():
    """Setup basic logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Download and preprocess dataset independently'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='Books',
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
        '--split',
        type=str,
        default='leave_one_out',
        help='Data split strategy (default: leave_one_out)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info('='*60)
    logger.info('Dataset Download & Preprocessing')
    logger.info('='*60)
    logger.info(f'Category: {args.category}')
    logger.info(f'Cache dir: {args.cache_dir}')
    logger.info(f'Metadata mode: {args.metadata_mode}')
    logger.info('='*60)
    
    # Create minimal config
    config = {
        'category': args.category,
        'cache_dir': args.cache_dir,
        'metadata': args.metadata_mode,
        'split': args.split,
        # accelerator is NOT included - dataset will work without it
    }
    
    try:
        # Download and preprocess
        logger.info('Creating dataset...')
        logger.info('')
        
        import time
        start_time = time.time()
    
        dataset = AmazonReviews2014(config)
        
        
        
        elapsed = time.time() - start_time
        
        logger.info('')
        logger.info(f'Dataset created in {elapsed:.1f}s')
        logger.info('')
        
        # Print dataset statistics
        logger.info(dataset)
        
        # Show processed file locations
        processed_dir = os.path.join(
            config['cache_dir'],
            'AmazonReviews2014',
            args.category,
            'processed'
        )
        logger.info('')
        logger.info(f'Processed data saved to: {processed_dir}')
        logger.info(f'  - all_item_seqs.json')
        logger.info(f'  - id_mapping.json')
        if args.metadata_mode != 'none':
            logger.info(f'  - metadata.{args.metadata_mode}.json')
        
        logger.info('')
        logger.info('Dataset download and preprocessing complete!')
        
        # Try showing the result of split() 
        
        logger.info('Splitting dataset...')
        split_datasets = dataset.split()
        
        logger.info("Train split:")
        logger.info(split_datasets['train'])
        logger.info("Validation split:")
        logger.info(split_datasets['val'])
        logger.info("Test split:")
        logger.info(split_datasets['test'])
        
        # 2. Check number of samples
        logger.info(f"Train: {len(split_datasets['train'])} samples")
        logger.info(f"Val: {len(split_datasets['val'])} samples")
        logger.info(f"Test: {len(split_datasets['test'])} samples")

        # Look at one sample in detail
        logger.info('Example train sample:')
        logger.info(split_datasets['train'][0])
        
    except KeyboardInterrupt:
        logger.warning('Dataset download interrupted by user.')
        return 1
    except Exception as e:
        logger.error(f'Error during download/preprocessing: {e}', exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
