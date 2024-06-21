from datasets import DatasetDict
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments

from labeler import Evaluator
from corpus_processor import Type

pretrained_model="bert-base-multilingual-uncased"


print("_"*20)
print(" "*10, end='')
print(pretrained_model)
print("_"*20)

model_dir = f"./Model_01.01/{pretrained_model}/model/"
dataset_dir = f"./built_datasets/{pretrained_model}/all.01/"
output_dir = f"./Model_01.01/{pretrained_model}/results/"
model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=3)
dataset = DatasetDict().load_from_disk(dataset_dir)

trainer = Trainer(model=model,

                  args=TrainingArguments(output_dir=output_dir),
                  train_dataset=dataset['train'],
                  eval_dataset=dataset["test"])
predictions = trainer.predict(dataset["test"])
predicted_labels = predictions.predictions.argmax(axis=-1)
evaluator = Evaluator(labels=(0, 1, 2))
evaluator.evaluate(dataset['test']['labels'], predicted_labels, Type.sents_raw)
evaluator.show_metrics()


## FOR TWO CLASS EVALUATION
# c2_predictions = [[1 if val == 2 else val for val in row] for row in predicted_labels]
# c2_targets = [[1 if val == 2 else val for val in row] for row in dataset['test']['labels']]
#
# evaluator = Evaluator(labels=(0, 1))
# evaluator.evaluate(c2_targets, c2_predictions, Type.sents_raw)
# evaluator.show_metrics()
#
#
