import json
import os
import sqlite3
import tempfile
import xml.etree.ElementTree as ET

import pandas as pd
import pytest
import yaml  # type: ignore
from datasets import Dataset
from pyarrow import feather
from pyarrow import parquet as pq

from geniusrise import BatchInput, BatchOutput, InMemoryState
from open_ai import OpenAITranslationFineTuner


# Helper function to create synthetic data in different formats
def create_dataset_in_format(directory, ext):
    os.makedirs(directory, exist_ok=True)
    data = [
        {
            "translation": {
                "en": f"This is a synthetic text example {i}",
                "fr": f"C'est un exemple de texte synthétique {i}",
            }
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
            ET.SubElement(record, "en").text = str(item["translation"]["en"])
            ET.SubElement(record, "fr").text = str(item["translation"]["fr"])
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
        ens = [x["translation"]["en"] for x in data]
        frs = [x["translation"]["fr"] for x in data]
        pd.DataFrame({"en": ens, "fr": frs}).to_sql("dataset_table", conn, if_exists="replace", index=False)
        conn.close()
    elif ext == "feather":
        feather.write_feather(df, os.path.join(directory, "data.feather"))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# Fixtures for each file type
@pytest.fixture(
    params=[
        "huggingface",
        "csv",
        "json",
        "jsonl",
        "parquet",
        "xml",
        "yaml",
        "tsv",
        "xlsx",
        "db",
        "feather",
    ]
)
def dataset_file(request, tmpdir):
    ext = request.param
    create_dataset_in_format(tmpdir + "/train", ext)
    create_dataset_in_format(tmpdir + "/eval", ext)
    return tmpdir, ext


@pytest.fixture
def translation_bolt():
    # Use temporary directories for input and output
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test-bucket", "test-openai-input")
    output = BatchOutput(output_dir, "geniusrise-test-bucket", "test-openai-output")
    state = InMemoryState()

    return OpenAITranslationFineTuner(
        input=input,
        output=output,
        state=state,
    )


def test_load_dataset(translation_bolt, dataset_file):
    tmpdir, ext = dataset_file
    translation_bolt.input.input_folder = tmpdir

    # Load the dataset
    dataset = translation_bolt.load_dataset(tmpdir + "/train")
    assert dataset is not None
    assert len(dataset) == 10
    if "translation" in dataset[0]:
        assert dataset[0]["translation"]["en"] == "This is a synthetic text example 0"
        assert dataset[0]["translation"]["fr"] == "C'est un exemple de texte synthétique 0"
    else:
        assert dataset[0]["en"] == "This is a synthetic text example 0"
        assert dataset[0]["fr"] == "C'est un exemple de texte synthétique 0"


def test_prepare_fine_tuning_data(translation_bolt, dataset_file):
    tmpdir, ext = dataset_file
    translation_bolt.input.input_folder = tmpdir
    # translation_bolt.source = "en"
    # translation_bolt.target = "fr"
    translation_bolt.prepare_fine_tuning_data(
        translation_bolt.load_dataset(tmpdir + "/train", source="en", target="fr"), "train"
    )
    translation_bolt.prepare_fine_tuning_data(
        translation_bolt.load_dataset(tmpdir + "/eval", source="en", target="fr"), "eval"
    )

    # Check that the train and eval files were created
    assert os.path.isfile(translation_bolt.train_file)
    assert os.path.isfile(translation_bolt.eval_file)

    # Check the content of the train file
    with open(translation_bolt.train_file, "r") as f:
        train_data = [line.strip() for line in f.readlines()]
    assert '"prompt"' in train_data[0]


def test_fine_tune(translation_bolt, dataset_file):
    tmpdir, ext = dataset_file
    translation_bolt.input.input_folder = tmpdir

    fine_tune_job = translation_bolt.fine_tune(
        model="ada",
        suffix="test",
        n_epochs=1,
        batch_size=1,
        learning_rate_multiplier=0.5,
        prompt_loss_weight=1,
        data_origin="en",
        data_target="fr",
    )
    assert "ft-" in fine_tune_job.id
