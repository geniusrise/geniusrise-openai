# import json
# import os
# import sqlite3
# import xml.etree.ElementTree as ET
# from typing import Any, Dict, Optional, Union

# import pandas as pd
# import pyarrow.parquet as pq
# import yaml
# from datasets import Dataset, DatasetDict, load_from_disk
# from pyarrow import feather

# from .base import OpenAIFineTuner


# class OpenAICommonsenseReasoningFineTuner(OpenAIFineTuner):
#     """
#     A bolt for fine-tuning OpenAI models for commonsense reasoning tasks.

#     This bolt uses the OpenAI API to fine-tune a pre-trained model for commonsense reasoning.

#     ## Using geniusrise to invoke via command line
#     ```bash
#     genius OpenAICommonsenseReasoningFineTuner rise \
#         # Add your command line arguments here
#     ```

#     ## Using geniusrise to invoke via YAML file
#     ```yaml
#     # Add your YAML configuration here
#     ```
#     """

#     def load_dataset(self, dataset_path: str, **kwargs: Any) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
#         """
#         Load a commonsense reasoning dataset from a directory.

#         Args:
#             dataset_path (str): The path to the dataset directory.
#             **kwargs: Additional keyword arguments.

#         Returns:
#             Dataset: The loaded dataset.

#         Raises:
#             Exception: If there was an error loading the dataset.

#         ## Supported Data Formats and Structures:

#         ### Hugging Face Dataset
#         Dataset files saved by the Hugging Face datasets library.

#         ### JSONL
#         Each line is a JSON object representing an example.

#         ### CSV
#         Should contain 'premise', 'hypothesis', and 'label' columns.

#         ### Parquet
#         Should contain 'premise', 'hypothesis', and 'label' columns.

#         ### JSON
#         An array of dictionaries with 'premise', 'hypothesis', and 'label' keys.

#         ### XML
#         Each 'record' element should contain 'premise', 'hypothesis', and 'label' child elements.

#         ### YAML
#         Each document should be a dictionary with 'premise', 'hypothesis', and 'label' keys.

#         ### TSV
#         Should contain 'premise', 'hypothesis', and 'label' columns separated by tabs.

#         ### Excel (.xls, .xlsx)
#         Should contain 'premise', 'hypothesis', and 'label' columns.

#         ### SQLite (.db)
#         Should contain a table with 'premise', 'hypothesis', and 'label' columns.

#         ### Feather
#         Should contain 'premise', 'hypothesis', and 'label' columns.
#         """
#         try:
#             data = []
#             if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
#                 return load_from_disk(dataset_path)
#             else:
#                 for filename in os.listdir(dataset_path):
#                     filepath = os.path.join(dataset_path, filename)
#                     if filename.endswith(".jsonl"):
#                         with open(filepath, "r") as f:
#                             for line in f:
#                                 example = json.loads(line)
#                                 data.append(example)
#                     elif filename.endswith(".csv"):
#                         df = pd.read_csv(filepath)
#                         data.extend(df.to_dict("records"))
#                     elif filename.endswith(".parquet"):
#                         df = pq.read_table(filepath).to_pandas()
#                         data.extend(df.to_dict("records"))
#                     elif filename.endswith(".json"):
#                         with open(filepath, "r") as f:
#                             data.extend(json.load(f))
#                     elif filename.endswith(".xml"):
#                         tree = ET.parse(filepath)
#                         root = tree.getroot()
#                         for record in root.findall("record"):
#                             example = {
#                                 "premise": record.find("premise").text,
#                                 "hypothesis": record.find("hypothesis").text,
#                                 "label": int(record.find("label").text),
#                             }
#                             data.append(example)
#                     elif filename.endswith((".yaml", ".yml")):
#                         with open(filepath, "r") as f:
#                             data.extend(yaml.safe_load(f))
#                     elif filename.endswith(".tsv"):
#                         df = pd.read_csv(filepath, sep="\t")
#                         data.extend(df.to_dict("records"))
#                     elif filename.endswith((".xls", ".xlsx")):
#                         df = pd.read_excel(filepath)
#                         data.extend(df.to_dict("records"))
#                     elif filename.endswith(".db"):
#                         conn = sqlite3.connect(filepath)
#                         query = "SELECT premise, hypothesis, label FROM dataset_table;"
#                         df = pd.read_sql_query(query, conn)
#                         data.extend(df.to_dict("records"))
#                     elif filename.endswith(".feather"):
#                         df = feather.read_feather(filepath)
#                         data.extend(df.to_dict("records"))

#                 return Dataset.from_pandas(pd.DataFrame(data))

#         except Exception as e:
#             self.log.exception(f"Failed to load dataset: {e}")
#             raise

#     def prepare_fine_tuning_data(self, data: Union[Dataset, DatasetDict, Optional[Dataset]], data_type: str) -> None:
#         """
#         Prepare the given data for fine-tuning.

#         Args:
#             data: The dataset to prepare.
#             data_type: Either 'train' or 'eval' to specify the type of data.

#         Raises:
#             ValueError: If data_type is not 'train' or 'eval'.
#         """
#         if data_type not in ["train", "eval"]:
#             raise ValueError("data_type must be either 'train' or 'eval'.")

#         # Convert the data to a pandas DataFrame
#         df = pd.DataFrame.from_records(data=data)

#         # For commonsense reasoning tasks, we need to convert the data into the format expected by OpenAI
#         df["prompt"] = df["premise"] + "\n" + df["hypothesis"]
#         df["completion"] = df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})
#         df = df[["prompt", "completion"]]

#         # Save the processed data into a file in JSONL format
#         file_path = os.path.join(self.input.get(), f"{data_type}.jsonl")
#         df.to_json(file_path, orient="records", lines=True)

#         if data_type == "train":
#             self.train_file = file_path
#         else:
#             self.eval_file = file_path

#     def preprocess_data(self, **kwargs) -> None:
#         """
#         Load and preprocess the dataset.

#         Raises:
#             Exception: If any step in the preprocessing fails.
#         """
#         try:
#             self.input.copy_from_remote()
#             train_dataset_path = os.path.join(self.input.get(), "train")
#             eval_dataset_path = os.path.join(self.input.get(), "eval")
#             train_dataset = self.load_dataset(train_dataset_path, **kwargs)
#             eval_dataset = self.load_dataset(eval_dataset_path, **kwargs)
#             self.prepare_fine_tuning_data(train_dataset, "train")
#             self.prepare_fine_tuning_data(eval_dataset, "eval")
#         except Exception as e:
#             self.log.exception(f"Failed to preprocess data: {e}")
#             raise
