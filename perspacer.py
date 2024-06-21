from sklearn.metrics import classification_report
import torch
from transformers import BertTokenizer, BertForTokenClassification
from corpus_processor import Type
from labeler import Labeler, Evaluator


# noinspection PyTypeChecker
class PerSpaCor:
    def __init__(self, model_path):
        self.__model_path = model_path
        self._tokenizer = BertTokenizer.from_pretrained(model_path)
        self._model = BertForTokenClassification.from_pretrained(model_path, num_labels=3)

        self._labeler = Labeler(tags=(1, 2),
                                regexes=(r'[^\S\r\n\v\f]', r'\u200c'),
                                chars=(" ", "‌"),
                                class_count=2)

    def _tokenize(self, chars, chunk_size=512):
        input_ids = []

        ids_ = [101] + [self._tokenizer.encode(char)[1] for char in chars] + [102]
        for i in range(0, len(ids_), chunk_size):
            chunked_ids = ids_[i:i + chunk_size]
            input_ids.append(chunked_ids)
        return input_ids

    def test(self):
        text = "منبع: (مجله سروش هفتگی، مصاحبه با رئیس دفتر الجزیره در تهران، یک هزار و سیصد و هشتاد) الجزیره هیچ ارتباط خاصی با طالبان ندارد. "
        chars, _ = self._labeler.label_text(text, corpus_type=Type.whole_raw)
        input_ids = self._tokenize(chars, chunk_size=512)
        input_ids = torch.tensor(input_ids)
        print(self._model)
        logits = self._model(input_ids).logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        true_labels = predictions
        # print(probabilities)
        # print(predictions)

        evaluator = Evaluator(labels=(0, 1, 2))
        evaluator.evaluate(true_labels, predictions, Type.whole_raw)
        evaluator.show_metrics()

    def correct(self, text, report=True):

        chars, labels = self._labeler.label_text(text, corpus_type=Type.whole_raw)
        input_ids = self._tokenize(chars, chunk_size=512)
        input_ids = torch.tensor(input_ids)

        labels = [0] + labels + [0]
        logits = self._model(input_ids).logits
        predicted_labels = torch.argmax(logits, dim=-1)
        if report:
            predictions_flat = [label for sample in predicted_labels.tolist() for label in sample]
            true_labels_flat = [label for sample in [labels] for label in sample]

            cls_report = classification_report(true_labels_flat, predictions_flat, digits=5)

            true_text = self._labeler.text_generator([' '] + chars + [' '],
                                                     [labels],
                                                     corpus_type=Type.whole_raw)
            print(" TRUE TEXT ")
            print(true_text)

            predicted_text = self._labeler.text_generator([' '] + chars + [' '],
                                                          predicted_labels,
                                                          corpus_type=Type.whole_raw)
            print(" PREDICTED TEXT ")
            print(predicted_text)
            # Print the report to the console
            print(cls_report)
        return self._labeler.text_generator([' '] + chars + [' '],
                                            predicted_labels,
                                            corpus_type=Type.whole_raw)


corrector = PerSpaCor(model_path="### MODEL PATH ###")
# Read the contents of the input file
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Correct the text
corrected_text = corrector.correct(text)

# Write the corrected text to the output file
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(corrected_text)
