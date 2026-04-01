"""
DEBUG/VERBOSE version of dataset download script.
Use this if you want to see detailed progress and debug output.

Usage:
    python scripts/01_download_and_preprocess_verbose.py --category Books --cache_dir ./cache
"""

import argparse
import logging
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014


def setup_verbose_logging():
    """Setup detailed logging to see all progress."""
    # Root logger - show everything
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Console handler with detailed format
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    
    # Also setup main logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_minimal_config(category, cache_dir, metadata_mode):
    """Create minimal config for dataset."""
    return {
        'category': category,
        'cache_dir': cache_dir,
        'metadata': metadata_mode,
        'split': 'leave_one_out',
    }


def main():
    parser = argparse.ArgumentParser(
        description='Download and preprocess dataset (VERBOSE MODE)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='Books',
        help='Amazon category'
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
    
    # Setup verbose logging
    setup_verbose_logging()
    logger = logging.getLogger(__name__)
    
    logger.info('='*70)
    logger.info('VERBOSE: Dataset Download & Preprocessing')
    logger.info('='*70)
    logger.info(f'Category: {args.category}')
    logger.info(f'Cache dir: {args.cache_dir}')
    logger.info(f'Metadata mode: {args.metadata_mode}')
    logger.info('')
    logger.info('NOTE: This is VERBOSE mode - you will see detailed progress')
    logger.info('WARNING: First download can take 10-30+ minutes depending on network speed')
    logger.info('         The dataset is 2-5GB in size')
    logger.info('='*70)
    
    config = create_minimal_config(
        category=args.category,
        cache_dir=args.cache_dir,
        metadata_mode=args.metadata_mode
    )
    
    try:
        import time
        start_time = time.time()
        
        logger.info('')
        logger.info('[STEP 1] Initializing dataset...')
        dataset = AmazonReviews2014(config)
        
        elapsed = time.time() - start_time
        logger.info(f'✓ Dataset created in {elapsed:.1f}s')
        logger.info('')
        
        logger.info('[STEP 2] Dataset Statistics:')
        logger.info(dataset)
        logger.info('')
        
        processed_dir = os.path.join(
            config['cache_dir'],
            'AmazonReviews2014',
            args.category,
            'processed'
        )
        
        logger.info('[STEP 3] Output Files:')
        logger.info(f'Location: {processed_dir}')
        
        # Check what files actually exist
        if os.path.exists(processed_dir):
            files = os.listdir(processed_dir)
            for f in sorted(files):
                filepath = os.path.join(processed_dir, f)
                size_mb = os.path.getsize(filepath) / (1024*1024)
                logger.info(f'  ✓ {f} ({size_mb:.1f} MB)')
        
        logger.info('')
        logger.info('='*70)
        logger.info('✓ SUCCESS: Dataset download and preprocessing complete!')
        logger.info(f'Total time: {elapsed:.1f}s')
        logger.info('='*70)
        
    except KeyboardInterrupt:
        logger.warning('Download interrupted by user (Ctrl+C)')
        return 1
    except Exception as e:
        logger.error(f'ERROR: {e}', exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
