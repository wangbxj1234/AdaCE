from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments , AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
import evaluate
import numpy as np
from datetime import datetime



dataset = load_dataset("wbxlala/sleep_edf_3he1")

train_dataset = dataset['train']
test_dataset = dataset['test']

model_name = "bert-base-uncased"


# Load the pre-trained language model with sequence classification head
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
# Disable Trainer's DataParallel for multigpu，
#model.is_parallelizable = True
#model.model_parallel = True


tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer.pad_token = tokenizer.eos_token
#model.config.pad_token_id = tokenizer.pad_token_id

def tokenize_text(examples):
    return tokenizer((examples['sample']),truncation=True,max_length=512, )


# Tokenize the dataset using datasets.map()
#val_dataset = val_dataset.map(tokenize_text,batched=True,num_proc=4, )
train_dataset = train_dataset.map(tokenize_text,batched=True,num_proc=4, )
test_dataset = test_dataset.map(tokenize_text,batched=True,num_proc=4, )



training_args = TrainingArguments(
    output_dir="bert",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    evaluation_strategy="epoch"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#from transformers import DataCollatorForTokenClassification
#data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    metric1 = evaluate.load('accuracy')
    metric2 = evaluate.load('f1')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    f1= metric2.compute(predictions=predictions, references=labels,average='macro')["f1"]
    accuracy= metric1.compute(predictions=predictions, references=labels)["accuracy"]
    return { "f1": f1, "accuracy": accuracy}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)  # doctest: +SKIP


start_time = datetime.now()
trainer.train()
print(f"Training time is : {datetime.now()-start_time}")


