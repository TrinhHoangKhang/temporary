"""
Simple test to verify the refactored dataset code works independently.

This test demonstrates that the dataset can be instantiated without
an Accelerator object, proving modularity improvements.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dataset_without_accelerator():
    """Test that dataset can be created without Accelerator."""
    
    from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014
    
    logger.info("="*60)
    logger.info("TEST 1: Create dataset WITHOUT Accelerator (Standalone)")
    logger.info("="*60)
    
    # Minimal config - NO accelerator!
    config = {
        'category': 'Books',
        'cache_dir': './test_cache',
        'metadata': 'none',  # Skip metadata for faster test
        'split': 'leave_one_out',
        # Note: NO 'accelerator' key!
    }
    
    try:
        logger.info("Creating AmazonReviews2014 dataset...")
        dataset = AmazonReviews2014(config)
        
        logger.info("✓ SUCCESS: Dataset created without Accelerator!")
        logger.info(f"  - Users: {dataset.n_users}")
        logger.info(f"  - Items: {dataset.n_items}")
        logger.info(f"  - Interactions: {dataset.n_interactions}")
        logger.info(f"  - Accelerator: {dataset.accelerator}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ FAILED: {e}", exc_info=True)
        return False


def test_dataset_with_accelerator():
    """Test that dataset still works WITH Accelerator (backward compat)."""
    
    from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014
    from accelerate import Accelerator
    
    logger.info("")
    logger.info("="*60)
    logger.info("TEST 2: Create dataset WITH Accelerator (Backward Compat)")
    logger.info("="*60)
    
    try:
        # Create a minimal accelerator
        accelerator = Accelerator()
        
        config = {
            'category': 'Books',
            'cache_dir': './test_cache',
            'metadata': 'none',
            'split': 'leave_one_out',
            'accelerator': accelerator,  # WITH accelerator
        }
        
        logger.info("Creating AmazonReviews2014 dataset with Accelerator...")
        dataset = AmazonReviews2014(config)
        
        logger.info("✓ SUCCESS: Dataset created with Accelerator!")
        logger.info(f"  - Users: {dataset.n_users}")
        logger.info(f"  - Items: {dataset.n_items}")
        logger.info(f"  - Accelerator: {type(dataset.accelerator).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ FAILED: {e}", exc_info=True)
        return False


def test_logging_without_accelerator():
    """Test that logging works without Accelerator."""
    
    from genrec.dataset import AbstractDataset
    
    logger.info("")
    logger.info("="*60)
    logger.info("TEST 3: Logging WITHOUT Accelerator")
    logger.info("="*60)
    
    try:
        config = {
            'split': 'leave_one_out',
        }
        
        class TestDataset(AbstractDataset):
            def _download_and_process_raw(self):
                pass
        
        dataset = TestDataset(config)
        
        logger.info("Testing dataset.log() method without accelerator...")
        dataset.log("This is a test log message")
        
        logger.info("✓ SUCCESS: Logging works without Accelerator!")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ FAILED: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    
    logger.info("\n")
    logger.info("╔" + "="*58 + "╗")
    logger.info("║" + " "*58 + "║")
    logger.info("║" + "  Dataset Modularity Tests (Backward Compatibility)".center(58) + "║")
    logger.info("║" + " "*58 + "║")
    logger.info("╚" + "="*58 + "╝")
    
    results = []
    
    # Run tests
    results.append(("Test 1: No Accelerator", test_dataset_without_accelerator()))
    results.append(("Test 2: With Accelerator", test_dataset_with_accelerator()))
    results.append(("Test 3: Logging", test_logging_without_accelerator()))
    
    # Summary
    logger.info("\n")
    logger.info("="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("="*60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("="*60)
    
    if passed == total:
        logger.info("✓ All tests passed! Dataset modularity is working correctly.")
        return 0
    else:
        logger.error(f"✗ {total - passed} test(s) failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
