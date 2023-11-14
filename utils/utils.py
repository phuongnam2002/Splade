import os
import torch
import random
import logging
import numpy as np
from transformers import (
    RobertaConfig,
    AutoTokenizer,
    set_seed
)
from components.models.splade import Splade

MODEL_CLASSES = {
    "unsim-cse-vietnamese": (RobertaConfig, Splade, AutoTokenizer),
    "sim-cse-vietnamese": (RobertaConfig, Splade, AutoTokenizer),
}

MODEL_PATH_MAP = {
    "unsim-cse-vietnamese": "VoVanPhuc/unsup-SimCSE-VietNamese-phobert-base",
    "sim-cse-vietnamese": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
}


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    data_dir = './logs/'
    os.makedirs(data_dir, exist_ok=True)
    file_handler = logging.FileHandler('{}/log.txt'.format(data_dir))
    file_handler.setFormatter(log_format)

    logger.handlers = [console_handler, file_handler]

    return logger


logger = _setup_logger()


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(
        args.model_name_or_path
    )
