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

FILE_DIR = '/home/raj/nlp/cmu-rag/rveerara/data/test/history'

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

def precision_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([precision(prediction, gt, normalize_fn) for gt in ground_truths])

def recall_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([recall(prediction, gt, normalize_fn) for gt in ground_truths])

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

    file_gold, file_1, file_2 = 'reference_answers.txt', 'llama2-text-only-answers.txt', 'bge-large-en-text-only-answers.txt'

    answer_files = ['gemma', 'llama2', 'mistral', 'neural-chat', 'openchat', 'tinyllama']
    files_eval = [f"{f}-text-only-answers.txt" for f in answer_files]
    files_eval.append('bge-large-en-text-only-answers.txt')
    # files_eval = ['llama2-text-only-answers.txt', 'bge-large-en-text-only-answers.txt']

    with open(f"{FILE_DIR}/{file_gold}", 'r') as f:
        gold = f.readlines()
    ground_truths = []
    for g in gold:
        ground_truths.append([g_.strip() for g_ in g.split(';')])
    gold = [x.strip() for x in gold]

    scores_dict = {k : {} for k in files_eval}

    for file in files_eval:
        with open(f"{FILE_DIR}/{file}", 'r') as f:
            sys = f.readlines()
        sys = [x.strip() for x in sys]

        assert len(gold) == len(sys), f"Length of gold and sys do not match for {file} with {len(gold)} and {len(sys)} respectively."

        scores_f1 = []
        scores_rogue = []
        scores_em = []
        scores_recall = []
        scores_precision = []

        for i in range(len(gold)):
            score_f1 = f1_score(sys[i], ground_truths[i], normalize_fn=normalize_answer)
            scores_f1.append(score_f1)
            score_rogue = rouge_score(sys[i], ground_truths[i])
            scores_rogue.append(score_rogue)
            score_em = exact_match_score(sys[i], ground_truths[i], normalize_fn=normalize_answer)
            scores_em.append(score_em)
            score_recall = recall_score(sys[i], ground_truths[i], normalize_fn=normalize_answer)
            scores_recall.append(score_recall)
            score_precision = precision_score(sys[i], ground_truths[i], normalize_fn=normalize_answer)
            scores_precision.append(score_precision)
        
        scores_dict[file]["f1"] = np.mean(scores_f1)
        scores_dict[file]["rogue"] = np.mean(scores_rogue)
        scores_dict[file]["em"] =   np.mean(scores_em)
        scores_dict[file]["recall"] =       np.mean(scores_recall)
        scores_dict[file]["precision"] =            np.mean(scores_precision)
    
        print(f"{file} mean f1 score: {np.mean(scores_f1)}")
        print(f"{file} mean rouge score: {np.mean(scores_rogue)}")
        print(f"{file} mean exact match score: {np.mean(scores_em)}")
        print(f"{file} mean recall score: {np.mean(scores_recall)}")
        print(f"{file} mean precision score: {np.mean(scores_precision)}")

    print(scores_dict)
    





