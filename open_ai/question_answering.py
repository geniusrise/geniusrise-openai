import ast
import json
import os
import sqlite3
import yaml  # type: ignore
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

from .base import OpenAIFineTuner


class OpenAIQuestionAnsweringFineTuner(OpenAIFineTuner):
    r"""
    A bolt for fine-tuning OpenAI models on question answering tasks.

    ## Using geniusrise to invoke via command line
    ```bash
    genius OpenAIQuestionAnsweringFineTuner rise \
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
    version: "1"
    bolts:
        my_openai_bolt:
            name: "OpenAIQuestionAnsweringFineTuner"
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
                    name: "my_openai_bolt"
                    namespace: "default"
                    image: "my_openai_bolt_image"
                    replicas: 1
    ```
    """

    def load_dataset(
        self,
        dataset_path: str,
        **kwargs: Dict[str, Any],
    ) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        r"""
        Load a dataset from a directory.

        ## Supported Data Formats and Structures:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"context": "The context content", "question": "The question", "answers": {"answer_start": [int], "context": [str]}}
        ```

        ### CSV
        Should contain 'context', 'question', and 'answers' columns.
        ```csv
        context,question,answers
        "The context content","The question","{'answer_start': [int], 'text': [str]}"
        ```

        ### Parquet
        Should contain 'context', 'question', and 'answers' columns.

        ### JSON
        An array of dictionaries with 'context', 'question', and 'answers' keys.
        ```json
        [{"context": "The context content", "question": "The question", "answers": {"answer_start": [int], "context": [str]}}]
        ```

        ### XML
        Each 'record' element should contain 'context', 'question', and 'answers' child elements.
        ```xml
        <record>
            <context>The context content</context>
            <question>The question</question>
            <answers answer_start="int" context="str"></answers>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'context', 'question', and 'answers' keys.
        ```yaml
        - context: "The context content"
          question: "The question"
          answers:
            answer_start: [int]
            context: [str]
        ```

        ### TSV
        Should contain 'context', 'question', and 'answers' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'context', 'question', and 'answers' columns.

        ### SQLite (.db)
        Should contain a table with 'context', 'question', and 'answers' columns.

        ### Feather
        Should contain 'context', 'question', and 'answers' columns.

        Args:
            dataset_path (str): The path to the dataset directory.
            pad_on_right (bool): Whether to pad on the right.
            max_length (int): The maximum length of the sequences.
            doc_stride (int): The document stride.
            evaluate_squadv2 (bool): Whether to evaluate using SQuAD v2 metrics.

        Returns:
            Dataset: The loaded dataset.
        """
        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                # Load dataset saved by Hugging Face datasets library
                return load_from_disk(dataset_path)
            else:
                # Load dataset from text files
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
                        df["answers"] = df["answers"].apply(ast.literal_eval)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".parquet"):
                        df = pd.read_parquet(filepath)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".json"):
                        with open(filepath, "r") as f:
                            json_data = json.load(f)
                            data.extend(json_data)
                    elif filename.endswith(".xml"):
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        for record in root.findall("record"):
                            context = record.find("context").text  # type: ignore
                            question = record.find("question").text  # type: ignore
                            answers = record.find("answers").text  # type: ignore
                            data.append(
                                {
                                    "context": context,
                                    "question": question,
                                    "answers": ast.literal_eval(answers),  # type: ignore
                                }
                            )
                    elif filename.endswith(".yaml") or filename.endswith(".yml"):
                        with open(filepath, "r") as f:
                            yaml_data = yaml.safe_load(f)
                            data.extend(yaml_data)
                    elif filename.endswith(".tsv"):
                        df = pd.read_csv(filepath, sep="\t")
                        df["answers"] = df["answers"].apply(ast.literal_eval)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith((".xls", ".xlsx")):
                        df = pd.read_excel(filepath)
                        df["answers"] = df["answers"].apply(ast.literal_eval)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".db"):
                        conn = sqlite3.connect(filepath)
                        query = "SELECT context, question, answers FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        df["answers"] = df["answers"].apply(ast.literal_eval)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = pd.read_feather(filepath)
                        data.extend(df.to_dict("records"))
                return Dataset.from_pandas(pd.DataFrame(data))
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
        if data_type not in ["train", "eval"]:
            raise ValueError("data_type must be either 'train' or 'eval'.")

        # Convert the data to a pandas DataFrame
        df = pd.DataFrame.from_records(data=data)

        # For NER tasks, we need to convert the data into the format expected by OpenAI
        df["prompt"] = df["context"] + "\n" + df["question"]
        df["completion"] = df["answers"].apply(lambda x: x[0]["text"][0])
        df = df[["prompt", "completion"]]

        # Save the processed data into a file in JSONL format
        file_path = os.path.join(self.input.get(), f"{data_type}.jsonl")
        df.to_json(file_path, orient="records", lines=True)

        if data_type == "train":
            self.train_file = file_path
        else:
            self.eval_file = file_path
