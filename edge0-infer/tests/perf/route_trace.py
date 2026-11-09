"""Summarize head predictions and background preparation from JSONL traces."""
import argparse
import json
from pathlib import Path


def summarize(rows):
    counts = {f'head_top{factor}k': dict(hits=0, predicted=0, actual=0) for factor in [1, 2]}
    initiated = consumed = matches = 0
    previous = None
    for row in rows:
        k = row['top_k']
        if not isinstance(k, int) or k < 1:
            raise ValueError('top_k must be positive')
        layers = {entry['layer']: entry for entry in row['layers']}
        if len(layers) != len(row['layers']):
            raise ValueError('Duplicate layer')
        for layer, entry in layers.items():
            initiated += len(entry['prepared'])
            if previous is None or previous['position'] + 1 != row['position']:
                continue
            if row.get('source_position', previous['position']) != previous['position']:
                continue
            owner = next((item for item in previous['layers'] if item['layer'] == layer-1), None)
            if owner is None or not owner['head']:
                continue
            prediction = owner['head'][:k]
            consumed += len(prediction)
            matches += sum(a == b for a, b in zip(prediction, entry['executed']))
            target = set(entry['actual'])
            for factor in [1, 2]:
                predicted = set(owner['head'][:factor*k])
                count = counts[f'head_top{factor}k']
                count['hits'] += len(predicted & target)
                count['predicted'] += len(predicted)
                count['actual'] += len(target)
        previous = row
    for count in counts.values():
        count['recall'] = count['hits'] / count['actual'] if count['actual'] else None
        count['precision'] = count['hits'] / count['predicted'] if count['predicted'] else None
    return dict(predictions=counts, prepared_experts=initiated,
                consumed_experts=consumed, consumed_matches=matches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize([json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]), indent=2))


if __name__ == '__main__':
    main()
