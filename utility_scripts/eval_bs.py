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

FILE_DIR = '/home/raj/nlp/cmu-rag/rveerara/data/test/acads_lti/handbook/'

rouge = Rouge()

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

def precision(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = round(1.0 * num_same / len(prediction_tokens), 2)
    return precision

def recall(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    recall = round(1.0 * num_same / len(ground_truth_tokens), 2)
    return recall

def precision_score(prediction, ground_truths, normalize_fn = normalize_answer):
    return max([precision(prediction, gt, normalize_fn) for gt in ground_truths])

def recall_score(prediction, ground_truths, normalize_fn = normalize_answer):
    return max([recall(prediction, gt, normalize_fn) for gt in ground_truths])

def rouge_wrapper(prediction, ground_truth):
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0


def f1_score(prediction, ground_truths, normalize_fn = normalize_answer):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])


def exact_match_score(prediction, ground_truths, normalize_fn = normalize_answer):
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

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=str, default="reference_answers.txt")
    parser.add_argument("file1", type=str, default="llama2-text-only-answers.txt")
    parser.add_argument("file2", type=str, default="llama2-text-only-answers.txt")
    parser.add_argument('--eval', type=str, default='f1', help='The evaluation type (f1/rouge/em/recall/precision)', choices=['f1', 'rouge', 'em', 'recall', 'precision'])
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to use')
    parser.add_argument('--sample_ratio', type=float, default=0.5, help='Ratio of samples to use')
    return parser.parse_args()

def run(ground_truths, sys1, sys2, eval_fn):
    sys1_scores, sys2_scores = [], []
    for i in range(len(ground_truths)):
        sys1_score = eval_fn(sys1[i], ground_truths[i])
        sys1_scores.append(sys1_score)
        sys2_score = eval_fn(sys2[i], ground_truths[i])
        sys2_scores.append(sys2_score)
    # print(np.mean(sys1_scores), np.mean(sys2_scores))
    return round(np.mean(sys1_scores), 2) * 100, np.round(np.mean(sys2_scores), 2) * 100


if __name__ == "__main__":
    args = parse_args()
    gold, file1, file2 = args.gold, args.file1, args.file2
    num_samples, sample_ratio = args.num_samples, args.sample_ratio

    with open(gold, 'r') as f:
        gold = f.readlines()
    
    gold = [x.strip() for x in gold]
    ground_truths = []
    
    for g in gold:
        ground_truths.append([g_.strip() for g_ in g.split(';')])

    with open(file1, 'r') as f:
        sys1 = f.readlines()
        sys1 = [x.strip() for x in sys1]
    with open(file2, 'r') as f:
        sys2 = f.readlines()
        sys2 = [x.strip() for x in sys2]
    

    assert len(gold) == len(sys1), f"Length of gold and sys do not match for {file1} with {len(gold)} and {len(sys1)} respectively."
    assert len(gold) == len(sys1), f"Length of gold and sys do not match for {file2} with {len(gold)} and {len(sys2)} respectively."

    sys1_scores, sys2_scores = [], []

    eval_fn = f1_score if args.eval == 'f1' else (rouge_score if args.eval == 'rouge' else (exact_match_score if args.eval == 'em' else (recall_score if args.eval == 'recall' else precision_score)))
    
    n = len(gold)
    ids = list(range(n))
    wins = [0, 0, 0]

    print(file1, file2)
          
    for _ in range(num_samples):
        # Subsample the gold and system outputs
        reduced_ids = np.random.choice(ids,int(len(ids)*sample_ratio),replace=True)
        reduced_ground_truths = [ground_truths[i] for i in reduced_ids]
        reduced_sys1 = [sys1[i] for i in reduced_ids]
        reduced_sys2 = [sys2[i] for i in reduced_ids]

        sys1_score, sys2_score = run(reduced_ground_truths, reduced_sys1, reduced_sys2, eval_fn)

        if sys1_score > sys2_score:
            wins[0] += 1
        elif sys1_score < sys2_score:
            wins[1] += 1
        else:
            wins[2] += 1
        sys1_scores.append(sys1_score)
        sys2_scores.append(sys2_score)

        # Print win stats
    wins = [x/float(num_samples) for x in wins]
    print('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
    if wins[0] > wins[1]:
        print('(sys1 is superior with p value p=%.3f)' % (1-wins[0]))
    elif wins[1] > wins[0]:
        print('(sys2 is superior with p value p=%.3f)' % (1-wins[1]))

    # Print system stats
    sys1_scores.sort()
    sys2_scores.sort()

    # print(print(sys1_score[0]))

    print('sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]))
    print('sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]))

    print("\n")

    # print(scores_dict)
