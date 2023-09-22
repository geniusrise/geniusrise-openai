import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Optional

import pandas as pd
import yaml  # type: ignore
from pyarrow import feather
from pyarrow import parquet as pq
from datasets import Dataset, DatasetDict, load_from_disk

from openai.validators import (
    apply_necessary_remediation,
    apply_optional_remediation,
    get_validators,
)

from geniusrise_openai.base import OpenAIFineTuner


class OpenAISummarizationFineTuner(OpenAIFineTuner):
    r"""
    A bolt for fine-tuning OpenAI models for summarization tasks.

    This bolt uses the OpenAI API to fine-tune a pre-trained model for summarization.

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

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> Optional[DatasetDict]:
        r"""
        Load a dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset | DatasetDict: The loaded dataset.

        ## Supported Data Formats and Structures:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"text": "The text content", "summary": "The summary"}
        ```

        ### CSV
        Should contain 'text' and 'summary' columns.
        ```csv
        text,summary
        "The text content","The summary"
        ```

        ### Parquet
        Should contain 'text' and 'summary' columns.

        ### JSON
        An array of dictionaries with 'text' and 'summary' keys.
        ```json
        [{"text": "The text content", "summary": "The summary"}]
        ```

        ### XML
        Each 'record' element should contain 'text' and 'summary' child elements.
        ```xml
        <record>
            <text>The text content</text>
            <summary>The summary</summary>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'text' and 'summary' keys.
        ```yaml
        - text: "The text content"
          summary: "The summary"
        ```

        ### TSV
        Should contain 'text' and 'summary' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'text' and 'summary' columns.

        ### SQLite (.db)
        Should contain a table with 'text' and 'summary' columns.

        ### Feather
        Should contain 'text' and 'summary' columns.
        """

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
                            document = record.find("document").text  # type: ignore
                            summary = record.find("summary").text  # type: ignore
                            data.append({"document": document, "summary": summary})
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
                        query = "SELECT document, summary FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))
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

        # For summarization tasks, we need to convert the data into the format expected by OpenAI
        df["prompt"] = df["document"]
        df["completion"] = df["summary"]
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
