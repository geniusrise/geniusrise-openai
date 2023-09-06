# type: ignore
import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
from datasets import Dataset
from geniusrise import BatchInput, BatchOutput, InMemoryState

from open_ai import OpenAIFineTuner
import openai


class TestOpenAIFineTuner(OpenAIFineTuner):
    def load_dataset(self, dataset_path, **kwargs):
        # Load a simple dataset for testing
        data = [{"prompt": "Hello", "completion": "Bonjour"}]
        return Dataset.from_pandas(pd.DataFrame(data))


@pytest.fixture
def bolt():
    # Use temporary directories for input and output
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test-bucket", "test-openai-input")
    output = BatchOutput(output_dir, "geniusrise-test-bucket", "test-openai-output")
    state = InMemoryState()

    return TestOpenAIFineTuner(
        input=input,
        output=output,
        state=state,
    )


def test_bolt_init(bolt):
    assert bolt.input is not None
    assert bolt.output is not None
    assert bolt.state is not None


def test_load_dataset(bolt):
    dataset = bolt.load_dataset("fake_path")
    assert dataset is not None
    assert len(dataset) == 1


def test_prepare_fine_tuning_data(bolt):
    data = [
        {"prompt": "Hello", "completion": "Bonjour"},
        {"prompt": "Goodbye", "completion": "Au revoir"},
    ]
    data_df = pd.DataFrame(data)
    bolt.prepare_fine_tuning_data(data_df, "train")
    bolt.prepare_fine_tuning_data(data_df, "eval")
    assert os.path.isfile(bolt.train_file)
    assert os.path.isfile(bolt.eval_file)


# # The following tests would interact with the actual OpenAI services
# # Make sure you have the necessary permissions and are aware of the potential costs


def test_fine_tune(bolt):
    # Prepare the fine-tuning data first
    data = [
        {"prompt": "Hello", "completion": "Bonjour"},
        {"prompt": "Goodbye", "completion": "Au revoir"},
    ]
    data_df = pd.DataFrame(data)
    bolt.prepare_fine_tuning_data(data_df, "train")
    bolt.prepare_fine_tuning_data(data_df, "eval")

    fine_tune_job = bolt.fine_tune(
        model="ada",
        suffix="test",
        n_epochs=1,
        batch_size=1,
        learning_rate_multiplier=0.5,
        prompt_loss_weight=1,
    )
    assert "ft-" in fine_tune_job.id


def test_get_fine_tuning_job(bolt):
    job = bolt.get_fine_tuning_job("ft-YIvWPvrzt9Lrfvs7CsvFQcvM")
    assert job.status == "pending" or job.status == "succeeded"


# def test_delete_fine_tuned_model(mock_delete, bolt):
# result = bolt.delete_fine_tuned_model("ft-YIvWPvrzt9Lrfvs7CsvFQcvM")
# assert result.deleted
