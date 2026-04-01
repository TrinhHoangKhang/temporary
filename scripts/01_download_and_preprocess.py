"""
Standalone script to download and preprocess the dataset independently.

Usage:
    python scripts/01_download_and_preprocess.py --category Books --cache_dir ./cache

This script can be run without the full pipeline, making it easy to:
- Download data once and reuse it
- Test different datasets
- Preprocess data on a different machine
"""

import argparse
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014


def setup_logging():
    """Setup basic logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_minimal_config(category, cache_dir, metadata_mode):
    """
    Create a minimal config dict needed for dataset download/preprocessing.
    
    Args:
        category: Amazon category (e.g., 'Books')
        cache_dir: Where to cache downloaded data
        metadata_mode: 'none', 'raw', or 'sentence'
    
    Returns:
        Minimal config dict
    """
    return {
        'category': category,
        'cache_dir': cache_dir,
        'metadata': metadata_mode,
        'split': 'leave_one_out',
        # accelerator is NOT included - dataset will work without it
    }


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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info('='*60)
    logger.info('Dataset Download & Preprocessing (Standalone Mode)')
    logger.info('='*60)
    logger.info(f'Category: {args.category}')
    logger.info(f'Cache dir: {args.cache_dir}')
    logger.info(f'Metadata mode: {args.metadata_mode}')
    logger.info('='*60)
    
    # Create minimal config
    config = create_minimal_config(
        category=args.category,
        cache_dir=args.cache_dir,
        metadata_mode=args.metadata_mode
    )
    
    try:
        # Download and preprocess
        logger.info('Creating dataset...')
        dataset = AmazonReviews2014(config)
        
        # Print dataset statistics
        logger.info('')
        logger.info(dataset)
        
        # Show processed file locations
        processed_dir = os.path.join(
            config['cache_dir'],
            'AmazonReviews2014',
            args.category,
            'processed'
        )
        logger.info('')
        logger.info(f'✓ Processed data saved to: {processed_dir}')
        logger.info(f'  - all_item_seqs.json')
        logger.info(f'  - id_mapping.json')
        if args.metadata_mode != 'none':
            logger.info(f'  - metadata.{args.metadata_mode}.json')
        
        logger.info('')
        logger.info('✓ Dataset download and preprocessing complete!')
        
    except Exception as e:
        logger.error(f'Error during download/preprocessing: {e}', exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
