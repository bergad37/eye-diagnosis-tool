# eye-diagnosis-tool

A retinal disease classification utility built on top of a RETFound feature
extractor and a scikit-learn classifier.

## Setup

This project uses a Python virtual environment located in `venv/`. Before
running any of the scripts you should activate it and install requirements:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

You can also invoke scripts without activating by using the interpreter path:

```bash
./venv/bin/python scripts/predict.py
```

## Usage

The `predict.py` script loads pre‑trained models and processes images contained
in `data/aptos/test_images/`.

```bash
./venv/bin/python scripts/predict.py
```

If you run the script with the wrong interpreter you'll see a helpful error
message telling you to install `joblib` or activate the virtual environment.
