import numpy as np
import itertools
from datasets import DatasetDict,Dataset,concatenate_datasets

def summery_chunked(input_ids_list, attention_mask_list, labels_list, show_data=False):
    def count_unique_types(two_d_list):
        unique_types = set()
        for sublist in two_d_list:
            for item in sublist:
                unique_types.add(type(item))
        return unique_types

    mask_shape = np.array(attention_mask_list).shape
    ids_shape = np.array(input_ids_list).shape
    lbl_shape = np.array(labels_list).shape
    print("Shapes\nIDs:\t\t\t", ids_shape)
    print("Labels:\t\t\t", lbl_shape)
    print("Attention Mask:\t", mask_shape)

    # Example usage:
    if show_data:
        print(f'input_ids\n{np.array(input_ids_list)}\n',
              f'labels_list\n{np.array(labels_list)}\n\n',
              f'attention_mask_list\n{np.array(attention_mask_list)}\n\n',
              f'Unique types in input_ids_list: {count_unique_types(input_ids_list)}\n\n',
              f'Unique types in labels_list: {count_unique_types(labels_list)}\n\n',
              f'Unique types in attention_mask_list: {count_unique_types(attention_mask_list)}\n\n')


def chunk_sentence(tokens: list, labels: list, chunk_size: int = 512) -> tuple:
    """
    Chunks a list of tokens and labels into smaller lists of size `chunk_size`.

    Args:
        tokens (list): A list of tokens.
        labels (list): A list of labels corresponding to the tokens.
        chunk_size (int, optional): The size of each chunk. Defaults to 512.

    Returns:
        tuple: A tuple containing two lists - one containing the chunked tokens and the other containing the corresponding labels.
    """
    if len(tokens) != len(labels):
        raise Exception("list sizes do not match")
    chunked_tokens = []
    chunked_labels = []
    for i in range(0, len(tokens), chunk_size):
        chunked_tokens.append(tokens[i:i + chunk_size])
        chunked_labels.append(labels[i:i + chunk_size])
    return chunked_tokens, chunked_labels

def chunk_pad_tokens(tokens, labels, chunk_size=512, padd=True, tok_pad=0, label_pad=-100, attention_pad=0,
                               summary=False):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    # Create chunks
    for i in range(0, len(tokens), chunk_size):  # We subtract 2 to account for special tokens
        chunked_tokens = tokens[i:i + chunk_size]
        chunk_attention_mask = [1] * len(chunked_tokens)
        chunk_label_ids = labels[i:i + chunk_size]

        while padd and len(chunked_tokens) < chunk_size:
            chunked_tokens.append(tok_pad)
            chunk_attention_mask.append(attention_pad)
            chunk_label_ids.append(label_pad)
        input_ids_list.append(chunked_tokens)
        attention_mask_list.append(chunk_attention_mask)
        labels_list.append(chunk_label_ids)
    if summary:
        summery_chunked(input_ids_list, attention_mask_list, labels_list, False)
    return input_ids_list, attention_mask_list, labels_list


def remove_large_elements(lst, threshold):
    for i in range(len(lst)):
        lst[i] = [x for x in lst[i] if len(str(x)) <= threshold]
    return lst


def map_ids_to_chars(ids: list[int], chars: list[str]) -> dict[int, list[str]]:
    """
    Maps IDs to their corresponding characters.

    Args:
        ids: A list of integers representing the IDs to map.
        chars: A list of strings representing the corresponding characters.

    Returns:
        A dictionary where each ID is mapped to a list of all the corresponding characters.
    """
    result = {}
    for i in range(len(ids)):
        if ids[i] not in result:
            result[ids[i]] = []
        for j in range(len(chars[i])):
            result[ids[i]].append(chars[i][j])
    return result


def flatten_2d_list(lst):
    return list(itertools.chain.from_iterable(lst))


def make_ready_for_ds(data, corpus_type, labeler, tokenizer, chunk_size=512):
    chars, labels = labeler.label_text(data, corpus_type)
    input_ids_list, labels_list = [], []
    for i in range(len(chars)):
        tokenized_sentence = [101] + [tokenizer.encode(char)[1] for char in chars[0]] + [102]
        labels[0].insert(0, 0)  # Adding CLS token label at the beginning of each sequence
        labels[0].append(0)
        input_ids_list += tokenized_sentence
        labels_list += labels[0]
        chars.pop(0)
        labels.pop(0)
    input_ids_list, attention_mask_list, labels_list = chunk_pad_tokens(input_ids_list, labels_list,
                                                                                     chunk_size, padd=True)
    # print(len(input_ids_list),len(attention_mask_list),len(labels_list))
    # for i in range(max(len(input_ids_list),len(attention_mask_list),len(labels_list))):
    #     print(len(input_ids_list[i]),len(attention_mask_list[i]),len(labels_list[i]))
    return input_ids_list, attention_mask_list, labels_list


def build_dataset(input_ids_list, attention_mask_list, labels_list, dataset_path):
    data_dict = {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }
    # Create a Hugging Face Dataset from the dictionary
    dataset = Dataset.from_dict(data_dict)
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=42)
    # Split the dataset into training, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, len(dataset)))
    # Combine the datasets into a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    dataset_dict._metadata = {"author": "Matin Ebrahimkhani"}
    dataset_dict.save_to_disk(dataset_path)
    return dataset_dict


def add_data_to_dataset(input_ids_list, attention_mask_list, labels_list, dataset_path, save_path):
    # Load the dataset from disk
    original_ds = DatasetDict.load_from_disk(dataset_path)
    # Create a dictionary from the input lists
    data_dict = {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }
    # Create a Hugging Face Dataset from the dictionary
    dataset = Dataset.from_dict(data_dict)
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=42)
    # Split the dataset into training, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, len(dataset)))

    # original_ds["train"] = original_ds["train"].concatenate(train_dataset)
    # original_ds["validation"] = original_ds["validation"].concatenate(val_dataset)
    # original_ds["test"] = original_ds["test"].concatenate(test_dataset)

    original_ds["train"] = concatenate_datasets([original_ds["train"], train_dataset])

    original_ds["validation"] = concatenate_datasets([original_ds["validation"], val_dataset])
    original_ds["test"] = concatenate_datasets([original_ds["test"], test_dataset])
    # Save the updated dataset to disk
    original_ds.save_to_disk(save_path)
    return original_ds


def chunk_2d_list(data, num_chunks):
    # Calculate the size of each chunk
    chunk_size = len(data) // num_chunks
    # Create a list of chunks for each input list
    data_chunk = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Return the list of chunks for each input list
    return data_chunk