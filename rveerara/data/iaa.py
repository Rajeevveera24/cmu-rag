from sklearn.metrics import cohen_kappa_score
import numpy as np

with open('annotator_scores_1.txt', 'r') as f:
    annotator_scores_1 = f.read().splitlines()
with open('annotator_scores_2.txt', 'r') as f:
    annotator_scores_2 = f.read().splitlines()

print(annotator_scores_1, annotator_scores_2)
    
annotator_scores_1 = [int(x) for x in annotator_scores_1]
annotator_scores_2 = [int(x) for x in annotator_scores_2]

cohen_kappa_scores = []

print(annotator_scores_1, annotator_scores_2)
      
num_samples = 100
ids = annotator_scores_1
sample_ratio = 1.0
assert len(annotator_scores_1) == len(annotator_scores_2), "The two labelers must have the same number of annotations"

# for _ in range(1):
# Subsample the gold and system outputs
    # reduced_ids = np.random.choice(ids,int(len(ids)*sample_ratio),replace=True)
    # reduced_gold = [annotator_scores_1[i] for i in reduced_ids]
    # reduced_sys1 = [annotator_scores_2[i] for i in reduced_ids]

p_e = 0.75
p_o = 0
print(annotator_scores_1, annotator_scores_2)

for i in range(len(annotator_scores_1)):
    p_o += 1 if annotator_scores_1[i] == annotator_scores_2[i] else 0
p_o /= len(annotator_scores_1)

print(p_o, p_e)

cohen_kappa_scores.append((p_o - p_e) / (1 - p_e))



print(cohen_kappa_scores)
print(np.mean(cohen_kappa_scores))