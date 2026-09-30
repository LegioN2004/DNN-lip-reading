# Lip Reading with Deep Learning

This project recognizes spoken phrases from video using only the speaker's mouth movements. It follows a notebook-based workflow for loading aligned video clips, extracting mouth regions, training a sequence model, and decoding predictions with connectionist temporal classification (CTC).

## What it does

The pipeline in [`lip-read.ipynb`](lip-read.ipynb) performs the following steps:

1. Downloads the video and alignment data with `gdown`.
2. Converts each video to grayscale frames and crops the mouth region to `46 x 140` pixels.
3. Normalizes the frame sequence and converts alignment text into character labels.
4. Builds a TensorFlow model with three 3D convolution blocks followed by two bidirectional LSTM layers.
5. Trains the model with CTC loss and periodically prints decoded examples.
6. Loads a saved checkpoint and runs predictions on test batches or an individual video.

The model is designed for short, aligned clips with a fixed maximum sequence length of 75 frames. It is an experiment and teaching project rather than a production-ready speech-reading system.

## Project layout

```text
.
├── lip-read.ipynb       # End-to-end exploration, training, and prediction
├── main.py              # Minimal TensorFlow environment check
├── pyproject.toml       # Project metadata and dependencies
├── uv.lock              # Locked environment dependencies
└── data/
    ├── s1/              # Video clips (*.mpg), downloaded at runtime
    └── alignments/s1/   # Word-level alignment files (*.align)
```

The `data/` directory and generated model files are ignored by Git because the dataset and checkpoints are downloaded artifacts.

## Requirements

- Python 3.10 or newer
- A working TensorFlow installation
- Jupyter Notebook or JupyterLab
- Enough disk space for the video dataset and checkpoints

TensorFlow can use a CPU, but training this model is substantially faster with a compatible GPU setup.

## Setup

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
source .venv/bin/activate
```

Start Jupyter from the project root:

```bash
jupyter lab
```

Then open `lip-read.ipynb` and run the cells in order. The notebook also contains installation and download cells for environments where the dependencies or dataset are not already available.

To verify the Python environment without opening the notebook:

```bash
uv run python main.py
```

## Data and checkpoints

The notebook downloads the training data from Google Drive into `data.zip` and extracts it into `data/`. It also includes a separate cell that downloads pretrained weights from `checkpoints.zip` into `models/`.

There are two ways to use the model:

- **Train from scratch:** run the training section. This can take a long time and writes checkpoints under `models/`.
- **Run inference:** skip training, download the checkpoint archive, load `models/checkpoint`, and run the prediction cells.

The notebook expects this layout:

```text
data/
├── s1/*.mpg
└── alignments/s1/*.align
```

Each video and alignment file must share the same base name, for example `bbal6n.mpg` and `bbal6n.align`.

## Running inference on a video

After loading the model weights, update the sample path in the final notebook section:

```python
sample = load_data(tf.convert_to_tensor("./data/s1/bras9a.mpg"))
yhat = model.predict(tf.expand_dims(sample[0], axis=0))
decoded = tf.keras.backend.ctc_decode(
    yhat, input_length=[75], greedy=True
)[0][0].numpy()
```

The notebook prints both the aligned reference text and the decoded prediction.

## Platform note

Some original example cells use Windows-style paths such as `'.\\data\\s1\\bras9a.mpg'`, and `load_data` currently splits paths on `\\`. On macOS or Linux, use forward-slash paths such as `./data/s1/bras9a.mpg`. If paths are passed through TensorFlow on another platform, update `load_data` to use `pathlib.Path` or `os.path` for platform-independent filename handling.

## Model overview

```text
Video frames (75 x 46 x 140 x 1)
	|
3D convolution + spatial pooling
	|
3D convolution + spatial pooling
	|
3D convolution + spatial pooling
	|
TimeDistributed flatten
	|
Bidirectional LSTM x 2
	|
Character probabilities
	|
CTC decoding
```

The character vocabulary includes lowercase letters, spaces, punctuation, and digits. Because the model uses CTC, the output sequence does not need to be aligned frame-by-frame with the input video.

## Known limitations

- The data loader uses a fixed mouth crop and assumes the source videos have the expected framing.
- The notebook currently uses a fixed 75-frame input length in several training and decoding cells.
- Training configuration, paths, and download steps are embedded in the notebook rather than exposed as a command-line interface.
- Recognition quality depends on the available speaker, vocabulary, alignment quality, and pretrained checkpoint.

## License and dataset

This repository does not currently declare a software license. Check the terms of the underlying video dataset and the linked model artifacts before redistributing them.
