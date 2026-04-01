from logging import getLogger
from typing import Union
import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import get_config, init_seed, init_logger, init_device, \
    get_dataset, get_tokenizer, get_model, get_trainer, log


class Pipeline:
    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        checkpoint_path: str = None,
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        """
        Build all runtime components for one experiment.

        Args:
            model_name: Model key string (registered in utils) or model class.
            dataset_name: Dataset key string (registered in utils) or dataset class.
            checkpoint_path: Optional checkpoint to load before evaluation/training.
            tokenizer: Optional tokenizer class override.
            trainer: Optional trainer instance override.
            config_dict: Runtime config overrides.
            config_file: Optional yaml config path.
        """
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )
        # Resolve runtime device and whether distributed training is enabled.
        self.config['device'], self.config['use_ddp'] = init_device() 
        self.checkpoint_path = checkpoint_path

        # Configure experiment log directory and accelerator tracker backend.
        self.project_dir = os.path.join(
            self.config['tensorboard_log_dir'],
            self.config["dataset"],
            self.config["model"]
        )
        self.accelerator = Accelerator(log_with='tensorboard', project_dir=self.project_dir)
        self.config['accelerator'] = self.accelerator

        # Initialize seed and shared logger after accelerator is ready.
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')

        # Build raw dataset and split into train/val/test in raw-ID space.
        self.raw_dataset = get_dataset(dataset_name)(self.config)
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()

        # Build tokenizer (custom override or model-registered default), then tokenize splits.
        if tokenizer is not None:
            self.tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)
        
        try:
            self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)
        except Exception as e:
            self.log(f'ERROR during tokenization: {e}', level='error')
            raise

        # Build model on main process first to avoid duplicated expensive initialization.
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)
            if checkpoint_path is not None:
                # Restore user-provided checkpoint for fine-tuning or direct evaluation.
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.config['device']))
                self.log(f'Loaded model checkpoint from {checkpoint_path}')
        self.log(self.model)
        self.log(self.model.n_parameters)

        # Build trainer (override allowed for custom training loops).
        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer)

    def run(self):
        """
        Execute the full experiment lifecycle.

        Returns:
            dict with:
                - `best_epoch`
                - `best_val_score`
                - `test_results`
        """
        # Build split-specific dataloaders; tokenizer can provide custom collate functions.
        train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            collate_fn=self.tokenizer.collate_fn['train']
        )
        val_dataloader = DataLoader(
            self.tokenized_datasets['val'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['val']
        )
        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )

        # Train with periodic validation; trainer owns early-stopping/checkpoint policy.
        best_epoch, best_val_score = self.trainer.fit(train_dataloader, val_dataloader)

        # Before test-time evaluation, sync workers and restore the best checkpoint.
        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)
        if self.checkpoint_path is None:
            # If we trained in this run, evaluate the best saved model instead of last-step weights.
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))

        # Prepare model + test dataloader for distributed-safe inference.
        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process and self.checkpoint_path is None:
            self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        
        # If the model supports decoding graph construction, enable it for evaluation efficiency reporting.
        if hasattr(self.trainer.model, 'generate_w_decoding_graph'):
            self.trainer.model.generate_w_decoding_graph = True
        test_results = self.trainer.evaluate(test_dataloader)

        # Emit test metrics to tracker from main process only.
        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {test_results}')

        # Ensure tracker resources are closed cleanly.
        self.trainer.end()
        return {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'test_results': test_results,
        }

    def log(self, message, level='info'):
        """Framework logging helper that respects distributed setup."""
        return log(message, self.config['accelerator'], self.logger, level=level)
