from enum import Enum

from corpus_processor import Type
from tabulate import tabulate


class Evaluator:
    def __init__(self, labels=(0, 1, 2)):
        # check if the lists have the same length

        # get the set of unique labels
        self.labels = labels
        self.label_count = len(labels)
        # store the lists as attributes
        self._true_labels = None
        self._pred_labels = None
        self._metrics = {}

    def _set_labels(self, true_labels, pred_labels):
        pass

    @staticmethod
    def _count_labels(true_labels, pred_labels, label):
        if len(true_labels) != len(pred_labels):
            raise ValueError("The lists of labels must have the same length")
        # use list comprehension to get the indices where the true label is equal to the current label
        true_indices = [i for i, x in enumerate(true_labels) if x == label]
        # use list comprehension to get the indices where the predicted label is equal to the current label
        pred_indices = [i for i, x in enumerate(pred_labels) if x == label]
        # use set operations to get the counts of true positives, false positives and false negatives
        tp = len(set(true_indices) & set(pred_indices))
        fp = len(set(pred_indices) - set(true_indices))
        fn = len(set(true_indices) - set(pred_indices))
        return tp, fp, fn, len(true_indices)

    def _compute_metrics(self, tp, fp, fn, label, true_count):

        # calculate the precision, recall, accuracy and f1 score for the label
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        accuracy = tp / true_count if true_count > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        # store the metrics in the dictionary with the label as the key
        self._metrics[label] = {"precision": precision, "recall": recall, "accuracy": accuracy, "f1_score": f1_score}

    def evaluate(self, true_labels, pred_labels, corpus_type: Enum):
        self._metrics = {}
        if corpus_type.value == Type.whole_raw.value:
            for label in self.labels:
                tp, fp, fn, true_count = self._count_labels(true_labels, pred_labels, label)
                self._compute_metrics(tp, fp, fn, label, true_count)

        elif corpus_type.value == Type.sents_raw.value:
            for label in self.labels:
                # initialize the counts of true positives, false positives, false negatives and true negatives
                tp = 0
                fp = 0
                fn = 0
                true_count = 0
                for i in range(len(true_labels)):
                    s_tp, s_fp, s_fn, s_true_count = self._count_labels(true_labels[i], pred_labels[i], label)
                    tp += s_tp
                    fp += s_fp
                    fn += s_fn
                    true_count += s_true_count

                self._compute_metrics(tp, fp, fn, label, true_count)

        else:
            raise Exception("invalid Corpus Type")

    def show_metrics(self):
        headers = ["Label", "Precision", "Recall", "Accuracy", "F1 Score"]
        rows = []
        avg_precision = 0
        avg_recall = 0
        avg_accuracy = 0
        avg_f1 = 0
        for label, metric in self._metrics.items():
            row = [label, metric["precision"], metric["recall"], metric["accuracy"], metric["f1_score"]]
            avg_precision += metric["precision"]
            avg_recall += metric["recall"]
            avg_accuracy += metric["accuracy"]
            avg_f1 += metric["f1_score"]
            rows.append(row)
        rows.append([
            "Average",
            avg_precision/self.label_count,
            avg_recall/self.label_count,
            avg_accuracy/self.label_count,
            avg_f1/self.label_count,
        ])
        table = tabulate(rows, headers=headers, tablefmt="fancy_grid")

        print(table)

    def test_whole_raw(self):
        true_labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        pred_labels = [0, 1, 2, 1, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='weighted')

        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'F1 Score: {f1:.2f}')
        self.evaluate(true_labels, pred_labels, corpus_type=Type.whole_raw)
        self.show_metrics()

    def test_sents_raw(self):
        true_labels = [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                       [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                       [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                       [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]]
        pred_labels = [[0, 1, 2, 1, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                       [0, 1, 2, 1, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                       [0, 1, 2, 1, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                       [0, 1, 2, 1, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]]


        self.evaluate(true_labels, pred_labels, corpus_type=Type.sents_raw)
        self.show_metrics()
