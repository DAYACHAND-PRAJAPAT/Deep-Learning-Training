from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np

dataset = load_dataset("Abirate/english_quotes")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

model = get_peft_model(base_model, lora_config)

def tokenize(batch):
    tokens = tokenizer(
        batch['quote'],
        truncation=True,
        padding='max_length',
        max_length=64
    )
    tokens["labels"] = np.array(tokens["input_ids"]).copy()
    tokens["labels"][tokens["labels"] == tokenizer.pad_token_id] = -100
    return tokens

tokenized = dataset.map(tokenize, batched=True)

args = TrainingArguments(
    "gpt2-finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'].select(range(200))
)

trainer.train()
