from transformers import (DataCollatorForTokenClassification,
                          AutoTokenizer, AutoModelForTokenClassification)
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

pretrained_model = "bert-base-multilingual-uncased"
model_dir = f"Model/"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
dataset = DatasetDict().load_from_disk("../_data/datasets/batched/bert-base-multilingual-uncased/")


class FourOClassifier(nn.Module):
    def __init__(self, clf_hidden_size, num_labels):
        super(FourOClassifier, self).__init__()
        self.dense = nn.Linear(clf_hidden_size, clf_hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batch_norm = nn.BatchNorm1d(clf_hidden_size)
        self.output_layer = nn.Linear(clf_hidden_size, num_labels)

    def forward(self, clf_input):
        x = self.dense(clf_input)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # BatchNorm1d expects (N, C, L)
        x = self.output_layer(x)
        return x


model = AutoModelForTokenClassification.from_pretrained(pretrained_model, num_labels=3)
hidden_size = model.config.hidden_size

num_labels = model.config.num_labels
fouro_classifier = FourOClassifier(hidden_size, num_labels)
model.classifier = fouro_classifier
print(model)

training_args = TrainingArguments(
    output_dir=model_dir,
    do_train=True,
    # do_eval=True,
    # do_predict=True,
    learning_rate=2e-5,
    num_train_epochs=1,
    auto_find_batch_size=True,
    # per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    # per_device_eval_batch_size=1,
    weight_decay=0.01,
    logging_dir=model_dir + '/logs',
    logging_strategy="steps",
    logging_steps=1000,
    save_strategy="steps",
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True,

)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
)
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)
print(model)
print(dataset)
print(training_args)
trainer.train()
trainer.save_model(model_dir + "model/")
