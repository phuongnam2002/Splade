import json
from typing import Dict, Text, List


def load_json(file_path: Text) -> List[Dict]:
    with open(file_path, 'r') as f:
        config = json.load(f)

    return config


def write_json(data, file_path, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as pf:
        json.dump(data, pf, ensure_ascii=False, indent=4)


def load_jsonl(file_path: Text) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(file_path, data):
    with open(file_path, 'w') as pf:
        for item in data:
            obj = json.dumps(item, ensure_ascii=False)
            pf.write(obj + '\n')
