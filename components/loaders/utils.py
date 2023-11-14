from typing import Tuple, List
from transformers import PreTrainedTokenizer

from utils.normalize import preprocessing


def convert_text_to_features(
        text: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 128,
        special_tokens_count: int = 2,
) -> Tuple[List[int], List[int]]:
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    text = preprocessing(text).split()

    tokens = []

    for word in text:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]
        tokens.extend(word_tokens)

    # Truncate data
    if len(tokens) > max_seq_len - special_tokens_count:
        tokens = tokens[:(max_seq_len - special_tokens_count)]

    tokens = [cls_token] + tokens + [sep_token]

    # convert tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length. This is static method padding
    padding_length = max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)

    assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(
        len(input_ids), max_seq_len
    )

    return input_ids, attention_mask
