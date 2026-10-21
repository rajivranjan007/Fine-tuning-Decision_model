# Tool selection with a fine-tuned LLM (Unsloth + xLAM)

## Goal
Given a user request and a few available tools (APIs), teach a small LLM to **pick the right tool** and say **how confident it is**.

> "Can you list the music genres available?"  
> tools: `get_genres`, `hot_100_songs`, `songs_v2_get_details`, `fetch_track_details`  
> answer: **`get_genres`** (with a probability for each tool)

This is the "which tool / which API should the agent call?" step that sits in front of any AI agent or MCP setup.

## What you get
- A LoRA-fine-tuned **Qwen3.5-2B** that chooses 1 tool out of 3-5 candidates, with a **probability per tool**.
- A **before vs after** comparison: accuracy, loss and calibration (ECE, does 80% confidence really mean 80% right?).
- A chart of the improvement and a list of wrong predictions to inspect.
- A demo cell where you type your own request and your own tools.

Results from one run (5,000 training rows, 2 epochs, 500 test rows, RTX 5050 8 GB):

| Metric | Before | After |
|---|---|---|
| Accuracy | 50.6% | 100% |
| ECE (lower is better) | 0.220 | 0.002 |
| Loss (lower is better) | 1.280 | 0.002 |

Caveat: the test rows are a random split, so the same tools appear in train and test. The 100% shows the model learned
this data, not yet that it handles **unseen** tools. Testing on tools held out from training is the next step.

## Get the data
Dataset: **xLAM function-calling 60k** (60,000 requests with tools and the correct call).
- Download page: https://huggingface.co/datasets/lockon/xlam-function-calling-60k
  (ungated mirror of [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k))

You do not download it by hand. One command does it and converts it:
```bash
pip install datasets
python prepare_xlam_data.py
```
This keeps rows with **one correct tool and 3-5 candidate tools**, removes duplicate requests, shuffles the tool order
(so the answer is not always first) and writes `data/xlam_train.jsonl` (13,821 rows) and `data/xlam_test.jsonl` (1,000 rows).

## Steps
1. **Install** (needs an NVIDIA GPU; 8 GB is enough; RTX 50-series needs a PyTorch `cu128` build):
   ```bash
   pip install unsloth datasets matplotlib
   ```
2. **Prepare the data:** `python prepare_xlam_data.py`
3. **Open** `train_tool_selection_model.ipynb`, pick the Python environment that has unsloth, and **run all cells**.
4. **Read the results:** BEFORE/AFTER numbers, the before-vs-after chart, accuracy by number of options, wrong predictions.
5. **Try your own tools** in the last cell.

For a quick first run the notebook uses 5,000 training and 500 test rows (`TRAIN_N`, `TEST_N` in the first cell).
Set both to `None` to use all of the data.

## Files
| File | What it does |
|---|---|
| `prepare_xlam_data.py` | downloads xLAM and converts it to the decision format |
| `train_tool_selection_model.ipynb` | trains, evaluates and visualises |
| `data/` | generated data (not committed) |
| `xlam-tool-selector/` | saved fine-tuned adapter (not committed) |
