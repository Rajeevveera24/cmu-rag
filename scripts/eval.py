# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import string
from collections import Counter
from typing import Callable

import numpy as np
import regex
from rouge import Rouge

rouge = Rouge()

logger = logging.getLogger(__name__)

# Normalization and score functions from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em(prediction, ground_truth, normalize_fn):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def f1(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = round(1.0 * num_same / len(prediction_tokens), 2)
    recall = round(1.0 * num_same / len(ground_truth_tokens), 2)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 2)


def rouge_wrapper(prediction, ground_truth):
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0


def f1_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])


def exact_match_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])


def rouge_score(prediction, ground_truths):
    ground_truths = [x for x in ground_truths if len(x) > 0]
    if (
        len(prediction) == 0 or len(ground_truths) == 0
    ):  # check if empty prediction or if there is no hypothesis with len > 0
        return 0.0, 0.0, 0.0
    scores = [rouge_wrapper(prediction, gt) for gt in ground_truths]
    rouge1 = max(s[0] for s in scores)
    rouge2 = max(s[1] for s in scores)
    rougel = max(s[2] for s in scores)
    return rouge1, rouge2, rougel

if __name__ == "__main__":

    FILE_DIR = '/home/raj/nlp/cmu-rag/annotation/test/history'
    file_gold, file_1, file_2 = 'reference_answers.txt', 'llama2_answers.txt', 'answers.txt'

    with open(f"{FILE_DIR}/{file_gold}", 'r') as f:
        gold = f.readlines()
    with open(f"{FILE_DIR}/{file_1}", 'r') as f:
        sys1 = f.readlines()
    with open(f"{FILE_DIR}/{file_2}", 'r') as f:
        sys2 = f.readlines()

    gold = [x.strip() for x in gold]
    sys1 = [x.strip() for x in sys1]
    sys2 = [x.strip() for x in sys2]

    assert len(gold) == len(sys1)
    assert len(gold) == len(sys2)

    ground_truths = []
    for g in gold:
        ground_truths.append([g_.strip() for g_ in g.split(';')])
    
    sys1_scores, sys2_scores= [],[]

    for i in range(len(gold)):
        sys1_score = f1_score(sys1[i], ground_truths[i], normalize_fn=normalize_answer)
        sys2_score = f1_score(sys2[i], ground_truths[i], normalize_fn=normalize_answer)
        sys1_scores.append(sys1_score)
        sys2_scores.append(sys2_score)
    
    sys1_scores.sort()
    sys2_scores.sort()
    print(sys1_scores)
    print(np.mean(sys1_scores))
    print(np.mean(sys2_scores))



