from datetime import datetime
from transformers import DefaultDataCollator
from transformers import AutoImageProcessor , AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
import torch
from datasets import Dataset

trainset=torch.load('/mnt/beegfs/bwangbo/huggingface/dataverse_files/train.pt')
testset =torch.load('/mnt/beegfs/bwangbo/huggingface/dataverse_files/test.pt')


def AdaCT(dataset):
    dat_dict1 = dict()
    dat_dict1["image"] = dataset['samples'].repeat(1, 3, 1).view(dataset['samples'].size(0),3, 30, 100)
    dat_dict1["label"] = dataset['labels']
    reshaped_dataset = Dataset.from_dict(dat_dict1)

    new_dataset = []
    for image,label in zip(
        reshaped_dataset['image'],
        reshaped_dataset['label'],
    ):
        new_example = {
            'pixel_values': (F.interpolate(torch.tensor(image).unsqueeze(0), size=(192, 192), mode='bilinear', align_corners=False)).squeeze(0),
            'label': int(label),
        }
        new_dataset.append(new_example)
    return new_dataset

train_dataset = AdaCT(trainset)
test_dataset = AdaCT(testset)



data_collator = DefaultDataCollator()

checkpoint = "microsoft/swinv2-large-patch4-window12-192-22k"


image_processor = AutoImageProcessor.from_pretrained(checkpoint)
#image_processor = DeiTImageProcessor.from_pretrained(checkpoint)


model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=5,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)


accuracy = evaluate.load("accuracy")
def compute_metrics(eval_preds):
    metric1 = evaluate.load('accuracy')
    metric2 = evaluate.load('f1')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    f1= metric2.compute(predictions=predictions, references=labels,average='macro')["f1"]
    accuracy= metric1.compute(predictions=predictions, references=labels)["accuracy"]
    return { "f1": f1, "accuracy": accuracy}


# def convert_to_new_format_1(dataset):
#     new_dataset = []
#     for image,label in zip(
#         dataset['image'],
#         dataset['label'],
#     ):
#         new_example = {
#             'pixel_values': (F.interpolate(torch.tensor(image).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)).squeeze(0),
#             'label': int(label),
#         }
#         new_dataset.append(new_example)
#     return new_dataset

# train_dataset = convert_to_new_format_1(train_dataset)
# test_dataset = convert_to_new_format_1(test_dataset)
#val_dataset = convert_to_new_format_1(encoded_val_dataset)

# print(train_dataset[0]['pixel_values'].size())
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    #push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)
start_time = datetime.now()
trainer.train()
print(f"Training time is : {datetime.now()-start_time}")
