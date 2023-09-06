# import os
# from typing import Optional

# import pandas as pd
# from datasets import Dataset, DatasetDict, load_from_disk

# from openai.validators import (
#     apply_necessary_remediation,
#     apply_optional_remediation,
#     get_validators,
# )

# from .base import OpenAIFineTuner


# class OpenAITranslationFineTuner(OpenAIFineTuner):
#     """
#     A bolt for fine-tuning OpenAI models for translation tasks.

#     This bolt uses the OpenAI API to fine-tune a pre-trained model for translation.

#     The dataset should be in the following format:
#     - Each example is a dictionary with the following keys:
#         - 'source': a string representing the source text to be translated.
#         - 'target': a string representing the translated text.
#     """

#     def load_dataset(self, dataset_path: str, **kwargs) -> Dataset | DatasetDict | Optional[Dataset]:
#         """
#         Load a translation dataset from a directory.

#         The directory can contain either:
#         - Dataset files saved by the Hugging Face datasets library, or
#         - Two text files named 'source.txt' and 'target.txt'. The 'source.txt' file should contain the source texts, one per line. The 'target.txt' file should contain the corresponding translated texts, also one per line.

#         Args:
#             dataset_path (str): The path to the dataset directory.

#         Returns:
#             Dataset: The loaded dataset.

#         Raises:
#             Exception: If there was an error loading the dataset.
#         """
#         try:
#             self.log.info(f"Loading dataset from {dataset_path}")
#             if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
#                 # Load dataset saved by Hugging Face datasets library
#                 return load_from_disk(dataset_path)
#             else:
#                 # Load dataset from text files
#                 data = []
#                 with open(os.path.join(dataset_path, "source.txt"), "r") as f:
#                     source_texts = f.read().strip().split("\n")
#                 with open(os.path.join(dataset_path, "target.txt"), "r") as f:
#                     target_texts = f.read().strip().split("\n")
#                 for source, target in zip(source_texts, target_texts):
#                     data.append({"source": source, "target": target})
#                 return Dataset.from_pandas(pd.DataFrame(data))
#         except Exception as e:
#             self.log.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
#             raise

#     def prepare_fine_tuning_data(
#         self,
#         data: Dataset | DatasetDict | Optional[Dataset],
#         apply_optional_remediations: bool = False,
#     ) -> None:
#         """
#         Prepare the given data for fine-tuning.

#         This method applies necessary and optional remediations to the data based on OpenAI's validators.
#         The remediations are logged and the processed data is saved into two files.

#         The data is converted into the format expected by OpenAI for fine-tuning:
#         - The 'prompt' field is created from the 'source' field.
#         - The 'completion' field is created from the 'target' field.

#         Args:
#             data (Dataset | DatasetDict | Optional[Dataset]): The data to prepare for fine-tuning. This should be a
#             Dataset or DatasetDict object, or a pandas DataFrame. If it's a DataFrame, it should have the following
#             columns: 'source', 'target'.
#             apply_optional_remediations (bool, optional): Whether to apply optional remediations. Defaults to False.

#         Raises:
#             Exception: If there was an error preparing the data.
#         """
#         # Convert the data to a pandas DataFrame
#         df = pd.DataFrame.from_records(data=data)

#         # For translation tasks, we need to convert the data into the format expected by OpenAI
#         df["prompt"] = df["source"]
#         df["completion"] = df["target"]
#         df = df[["prompt", "completion"]]

#         # Initialize a list to store optional remediations
#         optional_remediations = []

#         # Get OpenAI's validators
#         validators = get_validators()  # type: ignore

#         # Apply necessary remediations and store optional remediations
#         for validator in validators:
#             remediation = validator(df)
#             if remediation is not None:
#                 optional_remediations.append(remediation)
#                 df = apply_necessary_remediation(df, remediation)  # type: ignore

#         # Check if there are any optional or necessary remediations
#         any_optional_or_necessary_remediations = any(
#             [
#                 remediation
#                 for remediation in optional_remediations
#                 if remediation.optional_msg is not None or remediation.necessary_msg is not None
#             ]
#         )

#         # Apply optional remediations if there are any
#         if any_optional_or_necessary_remediations and apply_optional_remediations:
#             self.log.info("Based on the analysis we will perform the following actions:")
#             for remediation in optional_remediations:
#                 df, _ = apply_optional_remediation(df, remediation, auto_accept=True)  # type: ignore
#         else:
#             self.log.info("Validations passed, no remediations needed to be applied.")

#         # Save the processed data into two files in JSONL format
#         self.train_file = os.path.join(self.input.get(), "train.jsonl")  # type: ignore
#         self.eval_file = os.path.join(self.input.get(), "eval.jsonl")  # type: ignore
#         df.to_json(self.train_file, orient="records", lines=True)
#         df.to_json(self.eval_file, orient="records", lines=True)
