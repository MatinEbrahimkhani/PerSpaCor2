import argparse
from sklearn.metrics import classification_report
import torch
from transformers import BertTokenizer, BertForTokenClassification
from corpus_processor import Type
from labeler import Labeler, Evaluator


class TextSpacer:
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
        attention_masks = []

        ids_ = [101] + [self._tokenizer.encode(char)[1] for char in chars] + [102]
        for i in range(0, len(ids_), chunk_size):
            chunked_ids = ids_[i:i + chunk_size]
            attention_mask = [1] * len(chunked_ids)

            if len(chunked_ids) != chunk_size:
                attention_mask += [0] * (chunk_size - len(chunked_ids))  # padding the attention mask accordingly
                chunked_ids += [0] * (chunk_size - len(chunked_ids))  # padding the last chunk to chunk size

            input_ids.append(chunked_ids)
            attention_masks.append(attention_mask)

        return input_ids, attention_masks

    def test(self):
        text = ("منبع: (مجله سروش هفتگی، مصاحبه با رئیس دفتر الجزیره در تهران، یک هزار و سیصد و هشتاد) الجزیره هیچ "
                "ارتباط خاصی با طالبان ندارد.")
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


class PerSpacer(TextSpacer):
    def __init__(self, model_path):
        super().__init__(model_path)

    def correct(self, text_raw, report=True):
        chars, labels = self._labeler.label_text(text_raw, corpus_type=Type.whole_raw)
        input_ids, attention_mask = self._tokenize(chars, chunk_size=512)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = [0] + labels + [0]
        logits = self._model(input_ids, attention_mask=attention_mask).logits
        # print(logits)
        # print(labels)

        predicted_labels = torch.argmax(logits, dim=-1)
        # if report:
        #     predictions_flat = [label for sample in predicted_labels.tolist() for label in sample]
        #     true_labels_flat = [label for sample in [labels] for label in sample]
        #
        #     cls_report = classification_report(true_labels_flat, predictions_flat, digits=5)
        #
        #     true_text = self._labeler.text_generator(chars,
        #                                              [labels[1:-1]],
        #                                              corpus_type=Type.whole_raw)
        #     print(" TRUE TEXT ")
        #     print(true_text)
        #
        #     predicted_text = self._labeler.text_generator([' '] + chars + [' '],
        #                                                   predicted_labels,
        #                                                   corpus_type=Type.whole_raw).strip()
        #     print(" PREDICTED TEXT ")
        #     print(predicted_text)
        #     # Print the report to the console
        #     print(cls_report)

        predictions_flat = [label for sample in predicted_labels.tolist() for label in sample]
        return self._labeler.text_generator([' '] + chars + [' '],
                                            [predictions_flat],
                                            corpus_type=Type.whole_raw).strip()


class PerSpaCor(TextSpacer):
    def __init__(self, model_path):
        super().__init__(model_path)

    def correct(self, text_raw, alpha=0.5):
        chars, labels = self._labeler.label_text(text_raw, corpus_type=Type.whole_raw)
        input_ids, attention_mask = self._tokenize(chars, chunk_size=512)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = [0] + labels + [0]
        logits = self._model(input_ids, attention_mask=attention_mask).logits
        sequence_length = logits.size(1)
        if len(labels) < sequence_length:
            labels.extend([0] * (sequence_length - len(labels)))  # Padding with 0
        else:
            labels = labels[:sequence_length]  # Truncate if longer

        num_classes = logits.size(-1)
        user_labels_tensor = torch.tensor(labels)
        user_labels_one_hot = torch.nn.functional.one_hot(user_labels_tensor, num_classes=num_classes).float()
        # Define the coefficient alpha
        # Adjust this value to control the influence of user labels
        # Expand dimensions to match logits shape
        user_labels_one_hot = user_labels_one_hot.unsqueeze(0)

        # Expand dimensions to match logits shape
        user_labels_one_hot = user_labels_one_hot.unsqueeze(0)

        # Combine logits and user labels
        combined_logits = logits * (1 - alpha) + user_labels_one_hot * alpha

        # Optionally, apply softmax to get probabilities
        combined_probs = torch.nn.functional.softmax(combined_logits, dim=-1)

        # Get the final predictions
        final_predictions = torch.argmax(combined_probs, dim=-1)

        # print(final_predictions)
        predicted_labels = final_predictions
        predictions_flat = [label for sample in predicted_labels.tolist() for label in sample]
        resultchars= [' '] + chars + [' ']
        resultlabels = predictions_flat[:len(chars)+2]
        result = self._labeler.text_generator(resultchars,resultlabels,
                                            corpus_type=Type.whole_raw).strip()
        return result
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Correct text using a BERT model.')
    parser.add_argument('--model', choices=['perspacor', 'perspacer'], default='perspacor',
                        help="The model to use for text spacing")
    parser.add_argument('--model_path', type=str, default="./model01-plain/Model/model/",
                        help='Path to the model directory')
    parser.add_argument('--input_file', type=str, default="./chapter_1.tex", help='Path to the input file')
    parser.add_argument('--output_file', type=str, default="./chapter_1perspaced.tex", help='Path to the output file')

    args = parser.parse_args()

    # if args.model == 'perspacor':
    #     corrector = PerSpaCor(args.model_path)
    # elif args.model == 'perspacer':
    #     corrector = PerSpacer(args.model_path)
    # else:
    #     raise ValueError("Invalid model choice")
    #
    # with open(args.input_file, 'r', encoding='utf-8') as file:
    #     text = file.read()
    test_text = 140*"در این سرای بی‌کسیکسیبهدرنمیزند"
    test_text = "یکچیزیخودتبنویس"

    corrected_text = PerSpacer(args.model_path).correct(test_text, report=False)
    print("PerSpacer\n", corrected_text)
    print("-----------------------------")
    corrected_text = PerSpaCor(args.model_path).correct(test_text, alpha=1)
    print("PerSpaCor\n", corrected_text)

    with open(args.output_file, 'w', encoding='utf-8') as file:
        file.write(corrected_text)
