# import os
# import tempfile

# import pandas as pd
# import pytest
# from datasets import Dataset
# from geniusrise.bolts.openai.language_model import OpenAILanguageModelFineTuner
# from geniusrise.core import BatchInput, BatchOutput, InMemoryState

# # Retrieve environment variables
# api_key = os.getenv("OPENAI_API_KEY")
# api_type = os.getenv("OPENAI_API_TYPE")
# api_base_url = os.getenv("OPENAI_API_BASE_URL")
# api_version = os.getenv("OPENAI_API_VERSION")


# @pytest.fixture
# def bolt():
#     # Use temporary directories for input and output
#     input_dir = tempfile.mkdtemp()
#     output_dir = tempfile.mkdtemp()

#     # Create the expected directory structure for the train and eval datasets
#     train_dataset_path = os.path.join(input_dir, "train")
#     eval_dataset_path = os.path.join(input_dir, "eval")
#     os.makedirs(train_dataset_path)
#     os.makedirs(eval_dataset_path)

#     input = BatchInput(input_dir, "geniusrise-test-bucket", "test-openai-input")
#     output = BatchOutput(output_dir, "geniusrise-test-bucket", "test-openai-output")
#     state = InMemoryState()

#     return OpenAILanguageModelFineTuner(
#         input=input,
#         output=output,
#         state=state,
#         api_type=api_type,
#         api_key=api_key,
#         api_base=api_base_url,
#         api_version=api_version,
#         eval=False,
#     )


# def test_load_dataset(bolt):
#     # Create a temporary directory with a sample dataset
#     dataset_dir = tempfile.mkdtemp()
#     with open(os.path.join(dataset_dir, "example1.jsonl"), "w") as f:
#         f.write('{"text": "This is a sample text."}')

#     # Load the dataset
#     dataset = bolt.load_dataset(dataset_dir)
#     assert dataset is not None
#     assert len(dataset) == 1
#     assert dataset[0]["text"] == "This is a sample text."


# def test_prepare_fine_tuning_data(bolt):
#     # Create a sample dataset
#     data = [
#         {"text": "Hello, world!"},
#         {"text": "Goodbye, world!"},
#     ]
#     data_df = pd.DataFrame(data)

#     # Convert data_df to a Dataset
#     dataset = Dataset.from_pandas(data_df)

#     bolt.prepare_fine_tuning_data(dataset)

#     # Check that the train and eval files were created
#     assert os.path.isfile(bolt.train_file)
#     assert os.path.isfile(bolt.eval_file)

#     # Check the content of the train file
#     with open(bolt.train_file, "r") as f:
#         train_data = [line.strip() for line in f.readlines()]
#     assert train_data[0] == '{"prompt":"Hello, world!","completion":"Hello, world!"}'
#     assert train_data[1] == '{"prompt":"Goodbye, world!","completion":"Goodbye, world!"}'


# def test_fine_tune(bolt):
#     # Create a sample dataset
#     data = [
#         {"text": "Hello, world!"},
#         {"text": "Goodbye, world!"},
#     ]
#     data_df = pd.DataFrame(data)

#     # Convert data_df to a Dataset
#     dataset = Dataset.from_pandas(data_df)

#     # Prepare the fine-tuning data
#     bolt.prepare_fine_tuning_data(dataset)

#     fine_tune_job = bolt.fine_tune(
#         model="ada",
#         suffix="test",
#         n_epochs=1,
#         batch_size=1,
#         learning_rate_multiplier=0.5,
#         prompt_loss_weight=1,
#     )
#     assert "ft-" in fine_tune_job.id
