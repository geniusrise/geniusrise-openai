import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq
import yaml
from datasets import Dataset, DatasetDict, load_from_disk

from .base import OpenAIFineTuner


class NamedEntityRecognitionFineTuner(OpenAIFineTuner):
    r"""
    A bolt for fine-tuning OpenAI models on named entity recognition tasks.

    This bolt extends the OpenAIFineTuner to handle the specifics of named entity recognition tasks,
    such as the specific format of the datasets and the specific metrics for evaluation.

    Args:
        input (BatchInput): The batch input data.
        output (BatchOutput): The output data.
        state (State): The state manager.

    ## Using geniusrise to invoke via command line
    ```bash
    genius NamedEntityRecognitionFineTuner rise \
        batch \
            --input_bucket my_bucket \
            --input_folder my_folder \
        batch \
            --output_bucket my_output_bucket \
            --output_folder my_output_folder \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise \
            --postgres_table state \
        load_dataset \
            --args dataset_path=my_dataset_path
    ```

    ## Using geniusrise to invoke via YAML file
    ```yaml
    version: "1"
    bolts:
        my_ner_bolt:
            name: "NamedEntityRecognitionFineTuner"
            method: "load_dataset"
            args:
                dataset_path: "my_dataset_path"
            input:
                type: "batch"
                args:
                    bucket: "my_bucket"
                    folder: "my_folder"
            output:
                type: "batch"
                args:
                    bucket: "my_output_bucket"
                    folder: "my_output_folder"
            state:
                type: "postgres"
                args:
                    postgres_host: "127.0.0.1"
                    postgres_port: 5432
                    postgres_user: "postgres"
                    postgres_password: "postgres"
                    postgres_database: "geniusrise"
                    postgres_table: "state"
            deploy:
                type: "k8s"
                args:
                    name: "my_ner_bolt"
                    namespace: "default"
                    image: "my_ner_bolt_image"
                    replicas: 1
    ```
    """

    def load_dataset(
        self, dataset_path: str, **kwargs: Any
    ) -> Union[Dataset, DatasetDict, None]:
        r"""
        Load a named entity recognition dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.

        Returns:
            DatasetDict: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### Hugging Face Dataset
        Dataset files saved by the Hugging Face datasets library.

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"tokens": ["token1", "token2", ...], "ner_tags": [0, 1, ...]}
        ```

        ### CSV
        Should contain 'tokens' and 'ner_tags' columns.
        ```csv
        tokens,ner_tags
        "['token1', 'token2', ...]", "[0, 1, ...]"
        ```

        ### Parquet
        Should contain 'tokens' and 'ner_tags' columns.

        ### JSON
        An array of dictionaries with 'tokens' and 'ner_tags' keys.
        ```json
        [{"tokens": ["token1", "token2", ...], "ner_tags": [0, 1, ...]}]
        ```

        ### XML
        Each 'record' element should contain 'tokens' and 'ner_tags' child elements.
        ```xml
        <record>
            <tokens>token1 token2 ...</tokens>
            <ner_tags>0 1 ...</ner_tags>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'tokens' and 'ner_tags' keys.
        ```yaml
        - tokens: ["token1", "token2", ...]
          ner_tags: [0, 1, ...]
        ```

        ### TSV
        Should contain 'tokens' and 'ner_tags' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'tokens' and 'ner_tags' columns.

        ### SQLite (.db)
        Should contain a table with 'tokens' and 'ner_tags' columns.

        ### Feather
        Should contain 'tokens' and 'ner_tags' columns.
        """

        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                dataset = load_from_disk(dataset_path)
            else:
                data = []
                for filename in os.listdir(dataset_path):
                    filepath = os.path.join(dataset_path, filename)
                    if filename.endswith(".jsonl"):
                        with open(filepath, "r") as f:
                            for line in f:
                                example = json.loads(line)
                                data.append(example)
                    # Additional file types support
                    elif filename.endswith(".csv"):
                        df = pd.read_csv(filepath)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".parquet"):
                        df = pq.read_table(filepath).to_pandas()
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".json"):
                        with open(filepath, "r") as f:
                            json_data = json.load(f)
                            data.extend(json_data)
                    elif filename.endswith(".xml"):
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        for record in root.findall("record"):
                            tokens = record.find("tokens").text.split()  # type: ignore
                            ner_tags = list(map(int, record.find("ner_tags").text.split()))  # type: ignore
                            data.append({"tokens": tokens, "ner_tags": ner_tags})
                    elif filename.endswith(".yaml") or filename.endswith(".yml"):
                        with open(filepath, "r") as f:
                            yaml_data = yaml.safe_load(f)
                            data.extend(yaml_data)
                    elif filename.endswith(".tsv"):
                        df = pd.read_csv(filepath, sep="\t")
                        data.extend(df.to_dict("records"))
                    elif filename.endswith((".xls", ".xlsx")):
                        df = pd.read_excel(filepath)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".db"):
                        conn = sqlite3.connect(filepath)
                        query = "SELECT tokens, ner_tags FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                dataset = Dataset.from_pandas(pd.DataFrame(data))

            return dataset
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def prepare_fine_tuning_data(self, data: Union[Dataset, DatasetDict, Optional[Dataset]], data_type: str) -> None:
        """
        Prepare the given data for fine-tuning.

        Args:
            data: The dataset to prepare.
            data_type: Either 'train' or 'eval' to specify the type of data.

        Raises:
            ValueError: If data_type is not 'train' or 'eval'.
        """
        if data_type not in ['train', 'eval']:
            raise ValueError("data_type must be either 'train' or 'eval'.")

        # Convert the data to a pandas DataFrame
        df = pd.DataFrame.from_records(data=data)

        # For NER tasks, we need to convert the data into the format expected by OpenAI
        df["prompt"] = df["tokens"]
        df["completion"] = df["ner_tags"]
        df = df[["prompt", "completion"]]

        # Save the processed data into a file in JSONL format
        file_path = os.path.join(self.input.get(), f"{data_type}.jsonl")
        df.to_json(file_path, orient="records", lines=True)

        if data_type == 'train':
            self.train_file = file_path
        else:
            self.eval_file = file_path
