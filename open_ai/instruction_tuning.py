import glob
import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Optional, Union

import pandas as pd
import pyarrow.parquet as pq
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_from_disk
from pyarrow import feather

from .base import OpenAIFineTuner


class OpenAIInstructionFineTuner(OpenAIFineTuner):
    r"""
    A bolt for fine-tuning OpenAI models on instruction following tasks.

    This bolt uses the OpenAI API to fine-tune a pre-trained model for instruction following tasks.

    Args:
        input (BatchInput): The batch input data.
        output (BatchOutput): The output data.
        state (State): The state manager.

    ## Using geniusrise to invoke via command line
    ```bash
    genius OpenAIInstructionFineTuner rise \
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

    ## Using geniusrise to invoke via YAML file
    ```yaml
    version: 1

    bolts:
        my_fine_tuner:
            name: OpenAIInstructionFineTuner
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

    """

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        r"""
        Load an instruction following dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### Hugging Face Dataset
        Dataset files saved by the Hugging Face datasets library.

        ### JSONL
        Each line is a JSON object representing an example.

        ### CSV
        Should contain 'instruction' and 'output' columns.

        ### Parquet
        Should contain 'instruction' and 'output' columns.

        ### JSON
        An array of dictionaries with 'instruction' and 'output' keys.

        ### XML
        Each 'record' element should contain 'instruction' and 'output' child elements.

        ### YAML
        Each document should be a dictionary with 'instruction' and 'output' keys.

        ### TSV
        Should contain 'instruction' and 'output' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'instruction' and 'output' columns.

        ### SQLite (.db)
        Should contain a table with 'instruction' and 'output' columns.

        ### Feather
        Should contain 'instruction' and 'output' columns.
        """
        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            data = []
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                return load_from_disk(dataset_path)
            else:
                for filename in glob.glob(f"{dataset_path}/*"):
                    filepath = os.path.join(dataset_path, filename)
                    if filename.endswith(".jsonl"):
                        with open(filepath, "r") as f:
                            for line in f:
                                example = json.loads(line)
                                data.append(example)
                    elif filename.endswith(".csv"):
                        df = pd.read_csv(filepath)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".parquet"):
                        df = pq.read_table(filepath).to_pandas()
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".json"):
                        with open(filepath, "r") as f:
                            data.extend(json.load(f))
                    elif filename.endswith(".xml"):
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        for record in root.findall("record"):
                            example = {
                                "instruction": record.find("instruction").text,  # type: ignore
                                "output": record.find("output").text,  # type: ignore
                            }
                            data.append(example)
                    elif filename.endswith((".yaml", ".yml")):
                        with open(filepath, "r") as f:
                            data.extend(yaml.safe_load(f))
                    elif filename.endswith(".tsv"):
                        df = pd.read_csv(filepath, sep="\t")
                        data.extend(df.to_dict("records"))
                    elif filename.endswith((".xls", ".xlsx")):
                        df = pd.read_excel(filepath)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".db"):
                        conn = sqlite3.connect(filepath)
                        query = "SELECT instruction, output FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                return Dataset.from_pandas(pd.DataFrame(data))

        except Exception as e:
            self.log.exception(f"Failed to load dataset: {e}")
            raise

    def prepare_fine_tuning_data(self, data: Union[Dataset, DatasetDict, Optional[Dataset]], data_type: str) -> None:
        r"""
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

        # For instruction tuning tasks, we need to convert the data into the format expected by OpenAI
        df["prompt"] = df["instruction"]
        df["completion"] = df["output"].apply(str)
        df = df[["prompt", "completion"]]

        # Save the processed data into a file in JSONL format
        file_path = os.path.join(self.input.get(), f"{data_type}.jsonl")
        df.to_json(file_path, orient="records", lines=True)

        if data_type == "train":
            self.train_file = file_path
        else:
            self.eval_file = file_path
