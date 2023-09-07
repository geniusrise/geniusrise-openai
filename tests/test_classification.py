# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import json
import os
import sqlite3
import tempfile
import xml.etree.ElementTree as ET

import pandas as pd
import pytest
import yaml  # type: ignore
from datasets import Dataset
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from pyarrow import feather
from pyarrow import parquet as pq

from open_ai import OpenAIClassificationFineTuner


# Helper function to create synthetic data in different formats
def create_dataset_in_format(directory, ext):
    os.makedirs(directory, exist_ok=True)
    data = [{"text": f"text_{i}", "label": f"label_{i % 2}"} for i in range(10)]
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
            ET.SubElement(record, "text").text = item["text"]
            ET.SubElement(record, "label").text = item["label"]
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
        df.to_sql("dataset_table", conn, if_exists="replace", index=False)
        conn.close()
    elif ext == "feather":
        feather.write_feather(df, os.path.join(directory, "data.feather"))


# Fixtures for each file type
@pytest.fixture(
    params=[
        "huggingface",
        "csv",
        "jsonl",
        "parquet",
        "json",
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
def bolt():
    # Use temporary directories for input and output
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test-bucket", "test-openai-input")
    output = BatchOutput(output_dir, "geniusrise-test-bucket", "test-openai-output")
    state = InMemoryState()

    return OpenAIClassificationFineTuner(
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
    assert dataset[0]["prompt"] == "text_0"
    assert dataset[0]["completion"] == "label_0"


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
