import os
import tempfile
import pytest
import pandas as pd
import json
import sqlite3
import xml.etree.ElementTree as ET
import yaml  # type: ignore
from datasets import Dataset
from pyarrow import feather, parquet as pq

from geniusrise import BatchInput, BatchOutput, InMemoryState
from open_ai import OpenAIQuestionAnsweringFineTuner


# Helper function to create synthetic data in different formats
def create_dataset_in_format(directory, ext):
    os.makedirs(directory, exist_ok=True)
    data = [
        {
            "context": f"context_{i}",
            "question": f"question_{i}",
            "answers": [{"answer_start": [0], "text": [f"answer_{i}"]}],
        }
        for i in range(10)
    ]
    df = pd.DataFrame(data)

    if ext == "huggingface":
        dataset = Dataset.from_pandas(df)
        dataset.save_to_disk(directory)
    elif ext == "csv":
        df.to_csv(os.path.join(directory, "data.csv"), index=False)
    elif ext == "jsonl":
        with open(os.path.join(directory, "data.jsonl"), "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    elif ext == "parquet":
        pq.write_table(feather.Table.from_pandas(df), os.path.join(directory, "data.parquet"))
    elif ext == "json":
        with open(os.path.join(directory, "data.json"), "w") as f:
            json.dump(data, f)
    elif ext == "xml":
        root = ET.Element("root")
        for item in data:
            record = ET.SubElement(root, "record")
            ET.SubElement(record, "context").text = item["context"]
            ET.SubElement(record, "question").text = item["question"]
            ET.SubElement(record, "answers").text = str(item["answers"])
        tree = ET.ElementTree(root)
        tree.write(os.path.join(directory, "data.xml"))
    elif ext == "yaml":
        with open(os.path.join(directory, "data.yaml"), "w") as f:
            yaml.dump(data, f)
    elif ext == "tsv":
        df.to_csv(os.path.join(directory, "data.tsv"), index=False, sep="\t")
    elif ext == "xlsx":
        df.to_excel(os.path.join(directory, "data.xlsx"), index=False)
    elif ext == "db":
        conn = sqlite3.connect(os.path.join(directory, "data.db"))
        df["answers"] = df["answers"].apply(str)
        df.to_sql("dataset_table", conn, if_exists="replace", index=False)
        conn.close()
    elif ext == "feather":
        feather.write_feather(df, os.path.join(directory, "data.feather"))


# Fixtures for each file type
@pytest.fixture(
    params=[
        "db",
        "xml",
        "csv",
        "huggingface",
        "jsonl",
        "parquet",
        "json",
        "yaml",
        "tsv",
        "xlsx",
        "feather",
    ]
)
def dataset_file(request, tmpdir):
    ext = request.param
    create_dataset_in_format(tmpdir + "/train", ext)
    create_dataset_in_format(tmpdir + "/eval", ext)
    return tmpdir, ext


@pytest.fixture
def bolt():
    # Use temporary directories for input and output
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test-bucket", "test-openai-input")
    output = BatchOutput(output_dir, "geniusrise-test-bucket", "test-openai-output")
    state = InMemoryState()

    return OpenAIQuestionAnsweringFineTuner(
        input=input,
        output=output,
        state=state,
    )


def test_load_dataset(bolt, dataset_file):
    tmpdir, ext = dataset_file
    bolt.input.input_folder = tmpdir

    # Load the dataset
    dataset = bolt.load_dataset(tmpdir + "/train")
    assert dataset is not None
    assert len(dataset) == 10
    assert dataset[0]["context"] == "context_0"
    assert dataset[0]["question"] == "question_0"
    assert dataset[0]["answers"] == [{"answer_start": [0], "text": ["answer_0"]}]


def test_prepare_fine_tuning_data(bolt, dataset_file):
    tmpdir, ext = dataset_file
    bolt.input.input_folder = tmpdir
    bolt.prepare_fine_tuning_data(bolt.load_dataset(tmpdir + "/train"), "train")
    bolt.prepare_fine_tuning_data(bolt.load_dataset(tmpdir + "/eval"), "eval")

    # Check that the train and eval files were created
    assert os.path.isfile(bolt.train_file)
    assert os.path.isfile(bolt.eval_file)

    # Check the content of the train file
    with open(bolt.train_file, "r") as f:
        train_data = [line.strip() for line in f.readlines()]
    assert train_data[0] == '{"prompt":"context_0\\nquestion_0","completion":"answer_0"}'


def test_fine_tune(bolt, dataset_file):
    tmpdir, ext = dataset_file
    bolt.input.input_folder = tmpdir

    fine_tune_job = bolt.fine_tune(
        model="ada",
        suffix="test",
        n_epochs=1,
        batch_size=1,
        learning_rate_multiplier=0.5,
        prompt_loss_weight=1,
    )
    assert "ft-" in fine_tune_job.id
