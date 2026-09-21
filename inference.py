import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ADAPTER = "Msoldier-ai/bart-large-samsum-lora"  # اسم دقیق ریپوی مدل
tok = AutoTokenizer.from_pretrained("facebook/bart-large")
base = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
model = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload().eval()

text = input("Paste a dialogue (use \\n between lines): ").replace("\\n", "\n")
x = tok(text, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
    out = model.generate(**x, num_beams=4, max_length=128, min_length=15,
                         length_penalty=2.0, no_repeat_ngram_size=3,
                         early_stopping=True)
print(tok.decode(out[0], skip_special_tokens=True))