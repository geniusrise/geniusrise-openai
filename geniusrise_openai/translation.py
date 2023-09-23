# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

import json
import sqlite3
import xml.etree.ElementTree as ET
import ast
import pandas as pd
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_from_disk
from pyarrow import feather
from pyarrow import parquet as pq

from openai.validators import (
    apply_necessary_remediation,
    apply_optional_remediation,
    get_validators,
)

from geniusrise_openai.base import OpenAIFineTuner


class OpenAITranslationFineTuner(OpenAIFineTuner):
    r"""
    A bolt for fine-tuning OpenAI models for translation tasks.

    This bolt uses the OpenAI API to fine-tune a pre-trained model for translation.

    Args:
        input (BatchInput): The batch input data.
        output (BatchOutput): The output data.
        state (State): The state manager.

    CLI Usage:

    ```bash
        genius HuggingFaceCommonsenseReasoningFineTuner rise \
            batch \
                --input_s3_bucket geniusrise-test \
                --input_s3_folder train \
            batch \
                --output_s3_bucket geniusrise-test \
                --output_s3_folder model \
            fine_tune \
                --args model_name=my_model tokenizer_name=my_tokenizer num_train_epochs=3 per_device_train_batch_size=8
    ```

    YAML Configuration:

    ```yaml
        version: "1"
        bolts:
            my_fine_tuner:
                name: "HuggingFaceCommonsenseReasoningFineTuner"
                method: "fine_tune"
                args:
                    model_name: "my_model"
                    tokenizer_name: "my_tokenizer"
                    num_train_epochs: 3
                    per_device_train_batch_size: 8
                    data_max_length: 512
                input:
                    type: "batch"
                    args:
                        bucket: "my_bucket"
                        folder: "my_dataset"
                output:
                    type: "batch"
                    args:
                        bucket: "my_bucket"
                        folder: "my_model"
                deploy:
                    type: k8s
                    args:
                        kind: deployment
                        name: my_fine_tuner
                        context_name: arn:aws:eks:us-east-1:genius-dev:cluster/geniusrise-dev
                        namespace: geniusrise
                        image: geniusrise/geniusrise
                        kube_config_path: ~/.kube/config
    ```

    Supported Data Formats:
        - JSONL
        - CSV
        - Parquet
        - JSON
        - XML
        - YAML
        - TSV
        - Excel (.xls, .xlsx)
        - SQLite (.db)
        - Feather
    """

    def load_dataset(
        self, dataset_path: str, origin: str = "en", target: str = "fr", **kwargs
    ) -> Dataset | DatasetDict | Optional[Dataset]:
        r"""
        Load a dataset from a directory.

        ## Supported Data Formats and Structures for Translation Tasks:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {
            "translation": {
                "en": "English text",
                "fr": "French text"
            }
        }
        ```

        ### CSV
        Should contain 'en' and 'fr' columns.
        ```csv
        en,fr
        "English text","French text"
        ```

        ### Parquet
        Should contain 'en' and 'fr' columns.

        ### JSON
        An array of dictionaries with 'en' and 'fr' keys.
        ```json
        [
            {
                "en": "English text",
                "fr": "French text"
            }
        ]
        ```

        ### XML
        Each 'record' element should contain 'en' and 'fr' child elements.
        ```xml
        <record>
            <en>English text</en>
            <fr>French text</fr>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'en' and 'fr' keys.
        ```yaml
        - en: "English text"
          fr: "French text"
        ```

        ### TSV
        Should contain 'en' and 'fr' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'en' and 'fr' columns.

        ### SQLite (.db)
        Should contain a table with 'en' and 'fr' columns.

        ### Feather
        Should contain 'en' and 'fr' columns.

        Args:
            dataset_path (str): The path to the directory containing the dataset files.
            max_length (int, optional): The maximum length for tokenization. Defaults to 512.
            origin (str, optional): The origin language. Defaults to 'en'.
            target (str, optional): The target language. Defaults to 'fr'.
            **kwargs: Additional keyword arguments.

        Returns:
            DatasetDict: The loaded dataset.
        """
        self.origin = origin
        self.target = target

        try:
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
                    elif filename.endswith(".csv"):
                        df = pd.read_csv(filepath)
                        df["translation"] = df["translation"].apply(ast.literal_eval)
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
                            en = record.find(self.origin).text  # type: ignore
                            fr = record.find(self.target).text  # type: ignore
                            data.append({"translation": {self.origin: en, self.target: fr}})
                    elif filename.endswith(".yaml") or filename.endswith(".yml"):
                        with open(filepath, "r") as f:
                            yaml_data = yaml.safe_load(f)
                            data.extend(yaml_data)
                    elif filename.endswith(".tsv"):
                        df = pd.read_csv(filepath, sep="\t")
                        df["translation"] = df["translation"].apply(ast.literal_eval)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith((".xls", ".xlsx")):
                        df = pd.read_excel(filepath)
                        df["translation"] = df["translation"].apply(ast.literal_eval)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".db"):
                        conn = sqlite3.connect(filepath)
                        query = f"SELECT {self.origin}, {self.target} FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                if self.data_extractor_lambda:
                    fn = eval(self.data_extractor_lambda)
                    data = [fn(d) for d in data]
                else:
                    data = data

                dataset = Dataset.from_pandas(pd.DataFrame(data))
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

        return dataset

    def prepare_fine_tuning_data(self, data: Dataset | DatasetDict | Optional[Dataset], data_type: str) -> None:
        """
        Prepare the given data for fine-tuning.

        Args:
            data: The dataset to prepare.
            data_type: Either 'train' or 'eval' to specify the type of data.

        Raises:
            ValueError: If data_type is not 'train' or 'eval'.
        """
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame.from_records(data=data)

        # For translation tasks, we need to convert the data into the format expected by OpenAI
        if "translation" in df:
            print(df["translation"].apply(lambda x: x[self.origin]))
            df["prompt"] = df["translation"].apply(lambda x: x[self.origin])
            df["completion"] = df["translation"].apply(lambda x: x[self.target])
            df = df[["prompt", "completion"]]
        else:
            df["prompt"] = df[self.origin]
            df["completion"] = df[self.target]
            df = df[["prompt", "completion"]]

        # Initialize a list to store optional remediations
        optional_remediations = []

        # Get OpenAI's validators
        validators = get_validators()  # type: ignore

        # Apply necessary remediations and store optional remediations
        for validator in validators:
            remediation = validator(df)
            if remediation is not None:
                optional_remediations.append(remediation)
                df = apply_necessary_remediation(df, remediation)  # type: ignore

        # Check if there are any optional or necessary remediations
        any_optional_or_necessary_remediations = any(
            [
                remediation
                for remediation in optional_remediations
                if remediation.optional_msg is not None or remediation.necessary_msg is not None
            ]
        )

        # Apply optional remediations if there are any
        if any_optional_or_necessary_remediations:
            self.log.info("Based on the analysis we will perform the following actions:")
            for remediation in optional_remediations:
                df, _ = apply_optional_remediation(df, remediation, auto_accept=True)  # type: ignore
        else:
            self.log.info("Validations passed, no remediations needed to be applied.")

        # Save the processed data into two files in JSONL format
        self.train_file = os.path.join(self.input.get(), "train.jsonl")  # type: ignore
        self.eval_file = os.path.join(self.input.get(), "eval.jsonl")  # type: ignore
        df.to_json(self.train_file, orient="records", lines=True)
        df.to_json(self.eval_file, orient="records", lines=True)
