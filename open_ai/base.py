# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import os
from abc import abstractmethod
from time import sleep
from typing import Optional, Union

import openai
import pandas as pd
from datasets import Dataset, DatasetDict
from geniusrise import BatchInput, BatchOutput, Bolt, State
from openai.cli import FineTune
from tqdm import tqdm


class OpenAIFineTuner(Bolt):
    r"""
    An abstract base class for writing bolts for fine-tuning OpenAI models.

    This base class is intended to be subclassed for fine-tuning OpenAI models.
    The chief objective of its subclasses is to load and preprocess the dataset,
    though of course, other methods, including fine-tuning, can be overridden for customization.

    This bolt uses the OpenAI API to fine-tune a pre-trained model.

    Each subclass can be invoked using the `genius` cli or yaml.

    ## Using genius cli

    ```bash
    genius <bolt_name> rise \
        batch \
            --input_s3_bucket my-input-bucket \
            --input_s3_folder my-input-folder \
        batch \
            --output_s3_bucket my-output-bucket \
            --output_s3_folder my-output-folder \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise \
            --postgres_table task_state \
        fine_tune \
        --args
            model=gpt-3.5-turbo \
            n_epochs=2 \
            batch_size=64 \
            learning_rate_multiplier=0.5 \
            prompt_loss_weight=1 \
            wait=True
    ```

    This will load and preprocess data from input s3 location, and upload it to openai for fine tuning, and wait.

    ## Using YAML

    Bolts can be invoked using the `genius` cli on a yaml file.

    Create a yaml file with the following content (looks very similar to cli):

    ```yaml
    version: 1

    bolts:
        my_fine_tuner:
            name: OpenAIClassificationFineTuner
            method: fine_tune
            args:
                model: gpt-3.5-turbo
                n_epochs: 2
                batch_size: 64
                learning_rate_multiplier: 0.5
                prompt_loss_weight: 1
                wait: True
            input:
                type: batch
                bucket: my-input-bucket
                folder: my-input-folder
            output:
                type: batch
                bucket: my-output-bucket
                folder: my-output-folder
            state:
                type: postgres
                host: 127.0.0.1
                port: 5432
                user: postgres
                password: postgres
                database: geniusrise
                table: state
    ```

    ```bash
    genius rise
    ```

    Gotchas:

    1. Extra command line arguments can be passed to the load_dataset method via fine_tune method by appending `data_` to the param name.

    e.g.

    ```yaml
            args:
                model: gpt-3.5-turbo
                ...
                data_some_arbitrary_key: passed_to_load_dataset_method
    ```
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
    ) -> None:
        """
        Initialize the bolt.

        Args:
            input (BatchInput): The batch input data.
            output (BatchOutput): The output data.
            state (State): The state manager.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = logging.getLogger(self.__class__.__name__)
        self.train_file: Optional[str] = None
        self.eval_file: Optional[str] = None

    @abstractmethod
    def load_dataset(self, dataset_path: str, **kwargs) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        """
        Load a dataset from a file.

        Args:
            dataset_path (str): The path to the dataset file.
            **kwargs: Additional keyword arguments to pass to the `load_dataset` method.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def prepare_fine_tuning_data(self, data: Union[Dataset, DatasetDict, Optional[Dataset]], data_type: str) -> None:
        """
        Prepare the given data for fine-tuning.

        Args:
            data: The dataset to prepare.
            data_type: Either 'train' or 'eval' to specify the type of data.

        Raises:
            ValueError: If data_type is not 'train' or 'eval'.
        """
        if data_type not in ["train", "eval"]:
            raise ValueError("data_type must be either 'train' or 'eval'.")

        # Convert the data to a pandas DataFrame
        df = pd.DataFrame.from_records(data=data)

        # Save the processed data into a file in JSONL format
        file_path = os.path.join(self.input.get(), f"{data_type}.jsonl")
        df.to_json(file_path, orient="records", lines=True)

        if data_type == "train":
            self.train_file = file_path
        else:
            self.eval_file = file_path

    def preprocess_data(self, **kwargs) -> None:
        """
        Load and preprocess the dataset.

        Raises:
            Exception: If any step in the preprocessing fails.
        """
        try:
            self.input.copy_from_remote()
            train_dataset_path = os.path.join(self.input.get(), "train")
            eval_dataset_path = os.path.join(self.input.get(), "eval")
            train_dataset = self.load_dataset(train_dataset_path, **kwargs)
            eval_dataset = self.load_dataset(eval_dataset_path, **kwargs)
            self.prepare_fine_tuning_data(train_dataset, "train")
            self.prepare_fine_tuning_data(eval_dataset, "eval")
        except Exception as e:
            self.log.exception(f"Failed to preprocess data: {e}")
            raise

    def fine_tune(
        self,
        model: str,
        n_epochs: int,
        batch_size: int,
        learning_rate_multiplier: int,
        prompt_loss_weight: int,
        suffix: Optional[str] = None,
        wait: bool = False,
        **kwargs,
    ) -> openai.FineTune:
        """
        Fine-tune the model.

        Args:
            model (str): The pre-trained model name.
            suffix (str): The suffix to append to the model name.
            n_epochs (int): Total number of training epochs to perform.
            batch_size (int): Batch size during training.
            learning_rate_multiplier (int): Learning rate multiplier.
            prompt_loss_weight (int): Prompt loss weight.
            wait (bool, optional): Whether to wait for the fine-tuning to complete. Defaults to False.
            **kwargs: Additional keyword arguments for training and data loading.

        Raises:
            Exception: If any step in the fine-tuning process fails.
        """
        try:
            # Preprocess data
            dataset_kwargs = {k.replace("data_", ""): v for k, v in kwargs.items() if "data_" in k}
            self.preprocess_data(**dataset_kwargs)

            # Upload the training and validation files to OpenAI's servers
            tf = FineTune._get_or_upload(self.train_file, check_if_file_exists=False)
            vf = FineTune._get_or_upload(self.eval_file, check_if_file_exists=False)

            # Prepare the parameters for the fine-tuning request
            fine_tune_params = {
                "model": model,
                "suffix": suffix,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate_multiplier,
                "prompt_loss_weight": prompt_loss_weight,
                "training_file": tf,
                "validation_file": vf,
            }

            # Remove None values from the parameters
            fine_tune_params = {k: v for k, v in fine_tune_params.items() if v is not None}

            # Make the fine-tuning request
            fine_tune_job = openai.FineTune.create(**fine_tune_params)

            # Log the job ID
            self.log.info(f"🚀 Started fine-tuning job with ID {fine_tune_job.id}")

            if wait:
                self.wait_for_fine_tuning(fine_tune_job.id)

        except Exception as e:
            self.log.exception(f"Failed to fine-tune model: {e}")
            self.state.set_state(self.id, {"success": False, "exception": str(e)})
            raise

        self.state.set_state(self.id, {"success": True})
        return fine_tune_job

    @staticmethod
    def get_fine_tuning_job(job_id: str) -> openai.FineTune:
        """
        Get the status of a fine-tuning job.
        """
        return openai.FineTune.retrieve(job_id)

    def wait_for_fine_tuning(self, job_id: str, check_interval: int = 60) -> Optional[openai.FineTune]:
        """Wait for a fine-tuning job to complete, checking the status every `check_interval` seconds."""
        while True:
            job = self.get_fine_tuning_job(job_id)
            if job.status == "succeeded":
                self.log.info(f"🎉 Fine-tuning job {job_id} succeeded.")
                return job
            elif job.status == "failed":
                self.log.info(f"😭 Fine-tuning job {job_id} failed.")
                return job
            else:
                for _ in tqdm(
                    range(check_interval),
                    desc="Waiting for fine-tuning to complete",
                    ncols=100,
                ):
                    sleep(1)

    @staticmethod
    def delete_fine_tuned_model(model_id: str) -> openai.FineTune:
        """Delete a fine-tuned model."""
        return openai.FineTune.delete(model_id)
