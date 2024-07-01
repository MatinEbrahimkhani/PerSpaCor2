import argparse
from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
import torch
from labeler import Evaluator
from corpus_processor import Type


def main(model_dir, dataset_dir, injected_dir, alpha=0.5):
    print("_" * 20)
    print(" " * 10, end='')
    print(model_dir)
    print("_" * 20)

    model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=3)
    dataset = DatasetDict().load_from_disk(dataset_dir)
    injected_dataset = load_from_disk(injected_dir)
    from transformers import Trainer, TrainingArguments

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./results/"),
        train_dataset=dataset['train'],
        eval_dataset=dataset["test"]
    )

    predictions = trainer.predict(dataset["test"])

    model_logits = predictions.predictions

    user_labels_tensor = torch.tensor(injected_dataset['labels'])
    valid_mask = user_labels_tensor != -1
    filtered_labels = user_labels_tensor[valid_mask]

    # One-hot encode the filtered labels
    user_labels_one_hot = torch.nn.functional.one_hot(filtered_labels, num_classes=3).float()


    user_labels_one_hot = user_labels_one_hot.unsqueeze(0)

    # Expand dimensions to match logits shape
    user_labels_one_hot = user_labels_one_hot.unsqueeze(0)

    # Combine logits and user labels
    combined_logits = model_logits * (1 - alpha) + user_labels_one_hot * alpha
    combined_probs = torch.nn.functional.softmax(combined_logits, dim=-1)

    # Get the final predictions
    final_predictions = torch.argmax(combined_probs, dim=-1)

    evaluator = Evaluator(labels=(0, 1, 2))
    evaluator.evaluate(dataset['test']['labels'], final_predictions, Type.sents_raw)
    evaluator.show_metrics()

    # Uncomment for two-class evaluation
    # c2_predictions = [[1 if val == 2 else val for val in row] for row in predicted_labels]
    # c2_targets = [[1 if val == 2 else val for val in row] for row in dataset['test']['labels']]
    #
    # evaluator = Evaluator(labels=(0, 1))
    # evaluator.evaluate(c2_targets, c2_predictions, Type.sents_raw)
    # evaluator.show_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model with injected dataset")

    parser.add_argument('--model_dir', type=str, default="./model01-plain/Model/model/",
                        help='Path to the model directory')
    parser.add_argument('--dataset_dir', type=str, default="./_data/datasets/batched/bert-base-multilingual-uncased/",
                        help='Path to the dataset directory')
    parser.add_argument('--injected_dir', type=str,
                        default="./_data/datasets/batched/bert-base-multilingual-uncased/injected/",
                        help='Path to the injected dataset directory')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='coefficient to include user labels 1.0: just user labels 0.0: just model output')
    args = parser.parse_args()
    main(model_dir=args.model_dir, dataset_dir=args.dataset_dir, injected_dir=args.injected_dir, alpha=args.alpha)
