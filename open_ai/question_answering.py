# import ast
# import json
# import os
# import sqlite3
# import xml.etree.ElementTree as ET
# from typing import Any, Dict, Optional, Union

# import pandas as pd
# from datasets import Dataset, DatasetDict, load_from_disk
# from geniusrise import BatchInput, BatchOutput, State

# from .base import OpenAIFineTuner


# class OpenAIQuestionAnsweringFineTuner(OpenAIFineTuner):
#     r"""
#     A bolt for fine-tuning OpenAI models on question answering tasks.

#     ## Using geniusrise to invoke via command line
#     ```bash
#     genius OpenAIQuestionAnsweringFineTuner rise \
#         batch \
#             --input_bucket my_bucket \
#             --input_folder my_folder \
#         batch \
#             --output_bucket my_output_bucket \
#             --output_folder my_output_folder \
#         postgres \
#             --postgres_host 127.0.0.1 \
#             --postgres_port 5432 \
#             --postgres_user postgres \
#             --postgres_password postgres \
#             --postgres_database geniusrise \
#             --postgres_table state \
#         load_dataset \
#             --args dataset_path=my_dataset_path
#     ```

#     ## Using geniusrise to invoke via YAML file
#     ```yaml
#     version: "1"
#     bolts:
#         my_openai_bolt:
#             name: "OpenAIQuestionAnsweringFineTuner"
#             method: "load_dataset"
#             args:
#                 dataset_path: "my_dataset_path"
#             input:
#                 type: "batch"
#                 args:
#                     bucket: "my_bucket"
#                     folder: "my_folder"
#             output:
#                 type: "batch"
#                 args:
#                     bucket: "my_output_bucket"
#                     folder: "my_output_folder"
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
#                     name: "my_openai_bolt"
#                     namespace: "default"
#                     image: "my_openai_bolt_image"
#                     replicas: 1
#     ```
#     """

#     def __init__(
#         self,
#         input: BatchInput,
#         output: BatchOutput,
#         state: State,
#         **kwargs: Dict[str, Any],
#     ) -> None:
#         """
#         Initialize the bolt.

#         Args:
#             input (BatchInput): The batch input data.
#             output (BatchOutput): The output data.
#             state (State): The state manager.
#             **kwargs: Additional keyword arguments.
#         """
#         super().__init__(input=input, output=output, state=state)

#     def load_dataset(
#         self,
#         dataset_path: str,
#         **kwargs: Dict[str, Any],
#     ) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
#         """
#         Load a dataset from a directory.

#         Supported Data Formats and Structures:
#         - JSONL: Each line is a JSON object representing an example.
#         - CSV: Should contain 'text', 'question', and 'answer' columns.
#         - Parquet: Should contain 'text', 'question', and 'answer' columns.
#         - JSON: An array of dictionaries with 'text', 'question', and 'answer' keys.
#         - XML: Each 'record' element should contain 'text', 'question', and 'answer' child elements.
#         - YAML: Each document should be a dictionary with 'text', 'question', and 'answer' keys.
#         - TSV: Should contain 'text', 'question', and 'answer' columns separated by tabs.
#         - Excel (.xls, .xlsx): Should contain 'text', 'question', and 'answer' columns.
#         - SQLite (.db): Should contain a table with 'text', 'question', and 'answer' columns.
#         - Feather: Should contain 'text', 'question', and 'answer' columns.

#         Args:
#             dataset_path (str): The path to the dataset directory.

#         Returns:
#             Dataset: The loaded dataset.
#         """
#         try:
#             self.log.info(f"Loading dataset from {dataset_path}")
#             if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
#                 # Load dataset saved by Hugging Face datasets library
#                 return load_from_disk(dataset_path)
#             else:
#                 # Load dataset from text files
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
#                         df = pd.read_parquet(filepath)
#                         data.extend(df.to_dict("records"))
#                     elif filename.endswith(".json"):
#                         with open(filepath, "r") as f:
#                             json_data = json.load(f)
#                             data.extend(json_data)
#                     elif filename.endswith(".xml"):
#                         tree = ET.parse(filepath)
#                         root = tree.getroot()
#                         for record in root.findall("record"):
#                             text = record.find("text").text
#                             question = record.find("question").text
#                             answer = record.find("answer").text
#                             data.append({"text": text, "question": question, "answer": answer})
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
#                         query = "SELECT text, question, answer FROM dataset_table;"
#                         df = pd.read_sql_query(query, conn)
#                         data.extend(df.to_dict("records"))
#                     elif filename.endswith(".feather"):
#                         df = pd.read_feather(filepath)
#                         data.extend(df.to_dict("records"))
#                 return Dataset.from_pandas(pd.DataFrame(data))
#         except Exception as e:
#             self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
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

#         # For NER tasks, we need to convert the data into the format expected by OpenAI
#         df["prompt"] = df["text"] + "\n" + df["quesition"]
#         df["completion"] = df["ner_tags"]
#         df = df[["prompt", "completion"]]

#         # Save the processed data into a file in JSONL format
#         file_path = os.path.join(self.input.get(), f"{data_type}.jsonl")
#         df.to_json(file_path, orient="records", lines=True)

#         if data_type == "train":
#             self.train_file = file_path
#         else:
#             self.eval_file = file_path