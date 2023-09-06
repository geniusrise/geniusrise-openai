# import json
# import os
# import sqlite3
# import xml.etree.ElementTree as ET
# from typing import Optional, Union

# import pandas as pd
# from datasets import Dataset, DatasetDict, load_from_disk
# from pyarrow import feather
# from pyarrow import parquet as pq

# from openai.validators import (
#     apply_necessary_remediation,
#     apply_optional_remediation,
#     get_validators,
# )

# from .base import OpenAIFineTuner


# class OpenAIClassificationFineTuner(OpenAIFineTuner):
#     r"""
#     A bolt for fine-tuning OpenAI models for text classification tasks.

#     Args:
#         input (BatchInput): The batch input data.
#         output (BatchOutput): The output data.
#         state (State): The state manager.

#     ## Using geniusrise to invoke via command line

#     ```bash
#     genius OpenAIClassificationFineTuner rise \
#         batch \
#             --input_folder my_dataset \
#         streaming \
#             --output_kafka_topic my_topic \
#             --output_kafka_cluster_connection_string localhost:9094 \
#         postgres \
#             --postgres_host 127.0.0.1 \
#             --postgres_port 5432 \
#             --postgres_user postgres \
#             --postgres_password postgres \
#             --postgres_database geniusrise \
#             --postgres_table state \
#         load_dataset \
#             --args dataset_path=my_dataset
#     ```

#     ## Using geniusrise to invoke via YAML file
#     ```yaml
#     version: "1"
#     bolts:
#         my_fine_tuner:
#             name: "OpenAIClassificationFineTuner"
#             method: "load_dataset"
#             args:
#                 dataset_path: "my_dataset"
#             input:
#                 type: "batch"
#                 args:
#                     folder: "my_dataset"
#             output:
#                 type: "streaming"
#                 args:
#                     output_topic: "my_topic"
#                     kafka_servers: "localhost:9094"
#             state:
#                 type: "postgres"
#                 args:
#                     postgres_host: "127.0.0.1"
#                     postgres_port: 5432
#                     postgres_user: "postgres"
#                     postgres_password: "postgres"
#                     postgres_database: "geniusrise"
#                     postgres_table: "state"
#             deploy:
#                 type: "k8s"
#                 args:
#                     name: "my_fine_tuner"
#                     namespace: "default"
#                     image: "my_fine_tuner_image"
#                     replicas: 1
#     ```
#     """

#     def load_dataset(self, dataset_path: str, **kwargs) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
#         r"""
#         Load a classification dataset from a directory.

#         Args:
#             dataset_path (str): The path to the dataset directory.

#         Returns:
#             Dataset: The loaded dataset.

#         Raises:
#             Exception: If there was an error loading the dataset.

#         ## Supported Data Formats and Structures:

#         ### JSONL, CSV, Parquet, JSON, XML, YAML, TSV, Excel (.xls, .xlsx), SQLite (.db), Feather
#         """
#         try:
#             self.log.info(f"Loading dataset from {dataset_path}")
#             if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
#                 # Load dataset saved by Hugging Face datasets library
#                 return load_from_disk(dataset_path)
#             else:
#                 data = []
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
#                             json_data = json.load(f)
#                             data.extend(json_data)
#                     elif filename.endswith(".xml"):
#                         tree = ET.parse(filepath)
#                         root = tree.getroot()
#                         for record in root.findall("record"):
#                             text = record.find("text").text  # type: ignore
#                             label = record.find("label").text  # type: ignore
#                             data.append({"text": text, "label": label})
#                     elif filename.endswith(".yaml") or filename.endswith(".yml"):
#                         with open(filepath, "r") as f:
#                             yaml_data = yaml.safe_load(f)
#                             data.extend(yaml_data)
#                     elif filename.endswith(".tsv"):
#                         df = pd.read_csv(filepath, sep="\t")
#                         data.extend(df.to_dict("records"))
#                     elif filename.endswith((".xls", ".xlsx")):
#                         df = pd.read_excel(filepath)
#                         data.extend(df.to_dict("records"))
#                     elif filename.endswith(".db"):
#                         conn = sqlite3.connect(filepath)
#                         query = "SELECT text, label FROM dataset_table;"
#                         df = pd.read_sql_query(query, conn)
#                         data.extend(df.to_dict("records"))
#                     elif filename.endswith(".feather"):
#                         df = feather.read_feather(filepath)
#                         data.extend(df.to_dict("records"))

#                 return Dataset.from_pandas(pd.DataFrame(data))
#         except Exception as e:
#             self.log.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
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

#         # Save the processed data into a file in JSONL format
#         file_path = os.path.join(self.input.get(), f"{data_type}.jsonl")
#         df.to_json(file_path, orient="records", lines=True)

#         if data_type == "train":
#             self.train_file = file_path
#         else:
#             self.eval_file = file_path
