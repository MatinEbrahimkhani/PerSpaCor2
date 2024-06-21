import argparse
from datasets import DatasetDict
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments

from labeler import Evaluator
from corpus_processor import Type


def main(model_dir, dataset_dir, output_dir):
    print("_" * 20)
    print(" " * 10, end='')
    print(model_dir)
    print("_" * 20)

    model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=3)
    dataset = DatasetDict().load_from_disk(dataset_dir)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=output_dir),
        train_dataset=dataset['train'],
        eval_dataset=dataset["test"]
    )

    predictions = trainer.predict(dataset["test"])
    predicted_labels = predictions.predictions.argmax(axis=-1)

    evaluator = Evaluator(labels=(0, 1, 2))
    evaluator.evaluate(dataset['test']['labels'], predicted_labels, Type.sents_raw)
    evaluator.show_metrics()

    # Uncomment for two-class evaluation
    # c2_predictions = [[1 if val == 2 else val for val in row] for row in predicted_labels]
    # c2_targets = [[1 if val == 2 else val for val in row] for row in dataset['test']['labels']]
    #
    # evaluator = Evaluator(labels=(0, 1))
    # evaluator.evaluate(c2_targets, c2_predictions, Type.sents_raw)
    # evaluator.show_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a token classification model.')
    parser.add_argument('--model_dir', type=str, default="./Model_01.01/bert-base-multilingual-uncased/model/",
                        help='Path to the model directory')
    parser.add_argument('--dataset_dir', type=str, default="./built_datasets/bert-base-multilingual-uncased/all.01/",
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default="./Model_01.01/bert-base-multilingual-uncased/results/",
                        help='Path to the output directory')

    args = parser.parse_args()
    main(model_dir=args.model_dir, dataset_dir=args.dataset_dir, output_dir=args.output_dir)
