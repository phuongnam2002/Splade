import os
from tqdm import tqdm
from typing import List


def batchify(
        args,
        X: List[str],
        desc="",
        tqdm_bar: bool = True
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    batchs = [X[pos:pos + args.train_batch_size] for pos in range(0, len(X), args.train_batch_size)]

    if tqdm_bar:
        for batch in tqdm(batchs, position=0, total=1 + len(X) // args.train_batch_size, desc=desc):
            yield batch

    yield from batchs
