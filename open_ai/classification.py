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

import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Optional, Union

import pandas as pd
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_from_disk
from pyarrow import feather
from pyarrow import parquet as pq

from .base import OpenAIFineTuner


class OpenAIClassificationFineTuner(OpenAIFineTuner):
    r"""
    A bolt for fine-tuning OpenAI models for text classification tasks.

    Args:
        input (BatchInput): The batch input data.
        output (BatchOutput): The output data.
        state (State): The state manager.

    ## Using geniusrise to invoke via command line

    ```bash
    genius OpenAIClassificationFineTuner rise \
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
    """

    def load_dataset(self, dataset_path: str, **kwargs) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        r"""
        Load a classification dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"text": "The text content", "label": "The label"}
        ```

        ### CSV
        Should contain 'text' and 'label' columns.
        ```csv
        text,label
        "The text content","The label"
        ```

        ### Parquet
        Should contain 'text' and 'label' columns.

        ### JSON
        An array of dictionaries with 'text' and 'label' keys.
        ```json
        [{"text": "The text content", "label": "The label"}]
        ```

        ### XML
        Each 'record' element should contain 'text' and 'label' child elements.
        ```xml
        <record>
            <text>The text content</text>
            <label>The label</label>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'text' and 'label' keys.
        ```yaml
        - text: "The text content"
        label: "The label"
        ```

        ### TSV
        Should contain 'text' and 'label' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'text' and 'label' columns.

        ### SQLite (.db)
        Should contain a table with 'text' and 'label' columns.

        ### Feather
        Should contain 'text' and 'label' columns.
        """
        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                # Load dataset saved by Hugging Face datasets library
                dataset = load_from_disk(dataset_path)
                dataset = dataset.rename_column("text", "prompt")
                dataset = dataset.rename_column("label", "completion")
                return dataset
            else:
                data = []
                for filename in os.listdir(dataset_path):
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
                            json_data = json.load(f)
                            data.extend(json_data)
                    elif filename.endswith(".xml"):
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        for record in root.findall("record"):
                            text = record.find("text").text  # type: ignore
                            label = record.find("label").text  # type: ignore
                            data.append({"text": text, "label": label})
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
                        query = "SELECT text, label FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                dataset = Dataset.from_pandas(pd.DataFrame(data))
                dataset = dataset.rename_column("text", "prompt")
                dataset = dataset.rename_column("label", "completion")
                return dataset
        except Exception as e:
            self.log.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise
