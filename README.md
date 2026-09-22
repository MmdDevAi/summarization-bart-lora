# Dialogue Summarization with BART-large + LoRA

[

![Model](https://img.shields.io/badge/🤗-Model-yellow)

](https://huggingface.co/Msoldier-ai/bart-large-samsum-lora)
[

![Demo](https://img.shields.io/badge/🤗-Live%20Demo-blue)

](https://huggingface.co/spaces/Msoldier-ai/dialogue-summarizer-bart-lora)

Fine-tuning facebook/bart-large with LoRA to summarize chat-style conversations (SAMSum), evaluated with ROUGE and deployed as a live Gradio demo.



![demo](assets/demo.png)



## Project scope

A learning project to get hands-on with LoRA fine-tuning, generation tuning and evaluation, built with the help of AI assistants. The final model is a single training run with standard hyperparameters and no hyperparameter search, so the results are a baseline rather than a tuned result.

## Experiments

| Experiment | Data | Approach | ROUGE-1 / 2 / L / Lsum |
|---|---|---|---|
| PEGASUS | CNN/DailyMail (5,000-sample subset) | Fine-tuning + generation-parameter search (num_beams, length_penalty, min/max_length) | 38.6 / 18.7 / 31.6 / 34.2 |
| BART-large + LoRA (final) | SAMSum | LoRA, standard hyperparameters, no search | 51.26 / 26.39 / 42.44 / 47.24 |

PEGASUS full fine-tuning was heavy on a free-tier GPU (long training times, memory limits) and did not reach the quality I expected. I switched to BART-large with LoRA on SAMSum, which trains only a small adapter (~4.7 MB). The two experiments use different datasets (news vs. dialogue), so their scores are not directly comparable.

## Training (final model)

LoRA r=8, alpha=32, dropout 0.05 on q_proj/v_proj · 3 epochs · lr 2e-4 · effective batch size 16 · fp16 · best checkpoint by validation loss · Google Colab (NVIDIA T4). Full code in train.ipynb.

## Usage

pip install -r requirements.txt
python inference.py
## Tech stack

Python · PyTorch · Hugging Face Transformers, PEFT, Datasets · ROUGE · Gradio · Google Colab

## Limitations

- English only, trained on casual chat dialogues.
- With several speakers it sometimes attributes actions to the wrong person. Example: in a support chat where the agent promises a replacement charger, the model wrote "She will send a replacement" (the customer instead of the agent).
- On very short dialogues it may add redundant sentences.

## Future work

- Tune LoRA hyperparameters (rank, target modules, learning rate) and compare against other summarization models.
- Improve factual consistency and speaker attribution.
- Add factuality-oriented evaluation beyond ROUGE.

## References

BART (arXiv:1910.13461) · SAMSum (arXiv:1911.12237) · LoRA (arXiv:2106.09685) · PEGASUS (arXiv:1912.08777)
