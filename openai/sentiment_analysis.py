# import json
# import os
# import sqlite3
# import xml.etree.ElementTree as ET
# from typing import Any, Dict, List, Optional, Union

# import pandas as pd
# from datasets import Dataset, DatasetDict, load_from_disk
# from geniusrise import BatchInput, BatchOutput, State
# from pyarrow import feather
# from pyarrow import parquet as pq

# from .base import OpenAIFineTuner


# class OpenAISentimentAnalysisFineTuner(OpenAIFineTuner):
#     r"""
#     A bolt for fine-tuning OpenAI models on sentiment analysis tasks.

#     ## Using Command Line
#     ```bash
#     genius OpenAISentimentAnalysisFineTuner rise \
#         batch \
#             --input_bucket my-bucket \
#             --input_folder my-folder \
#         batch \
#             --output_bucket my-bucket \
#             --output_folder my-folder \
#         postgres \
#             --postgres_host 127.0.0.1 \
#             --postgres_port 5432 \
#             --postgres_user postgres \
#             --postgres_password postgres \
#             --postgres_database geniusrise \
#             --postgres_table state \
#         listen \
#             --args various=30 arguments=40 that=50 this=70 bolt=63 may=lol have='{"lol": "lel"}'
#     ```

#     ## Using YAML File
#     ```yaml
#     version: "1"
#     bolts:
#         my_fine_tuner:
#             name: "OpenAISentimentAnalysisFineTuner"
#             method: "load_dataset"
#             args:
#                 dataset_path: "/path/to/dataset"
#             input:
#                 type: "batch"
#                 args:
#                     bucket: "my-bucket"
#                     folder: "my-folder"
#             output:
#                 type: "batch"
#                 args:
#                     bucket: "my-bucket"
#                     folder: "my-folder"
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

#     Args:
#         input (BatchInput): The batch input data.
#         output (BatchOutput): The output data.
#         state (State): The state manager.
#     """

#     def load_dataset(self, dataset_path: str, **kwargs: Any) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
#         r"""
#         Load a dataset from a directory.

#         Args:
#             dataset_path (str): The path to the dataset directory.
#             **kwargs: Additional keyword arguments.

#         Returns:
#             Dataset | DatasetDict: The loaded dataset.

#         ## Supported Data Formats and Structures:

#         ### JSONL
#         Each line is a JSON object representing an example.
#         ```json
#         {"text": "The text content", "label": "The label"}
#         ```

#         ### CSV
#         Should contain 'text' and 'label' columns.
#         ```csv
#         text,label
#         "The text content","The label"
#         ```

#         ### Parquet
#         Should contain 'text' and 'label' columns.

#         ### JSON
#         An array of dictionaries with 'text' and 'label' keys.
#         ```json
#         [{"text": "The text content", "label": "The label"}]
#         ```

#         ### XML
#         Each 'record' element should contain 'text' and 'label' child elements.
#         ```xml
#         <record>
#             <text>The text content</text>
#             <label>The label</label>
#         </record>
#         ```

#         ### YAML
#         Each document should be a dictionary with 'text' and 'label' keys.
#         ```yaml
#         - text: "The text content"
#           label: "The label"
#         ```

#         ### TSV
#         Should contain 'text' and 'label' columns separated by tabs.

#         ### Excel (.xls, .xlsx)
#         Should contain 'text' and 'label' columns.

#         ### SQLite (.db)
#         Should contain a table with 'text' and 'label' columns.

#         ### Feather
#         Should contain 'text' and 'label' columns.
#         """
#         data = []
#         try:
#             if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
#                 dataset = load_from_disk(dataset_path)
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
#                 dataset = Dataset.from_pandas(pd.DataFrame(data))
#         except Exception as e:
#             self.log.exception(f"Failed to load dataset: {e}")
#             raise
#         return dataset

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
