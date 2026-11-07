# DATA 419 Homework #3: What's in a name?

**Name:** [YOUR NAME]  [, teammate: NAME]
**Kaggle team name:** [TEAM NAME]

## Features I engineered

Before computing any features, each name is lowercased and stripped of non-letters. From that I built:

- **Character n-grams (1–5 letters):** one 0/1 indicator per letter sequence of length 1–5 found in the name. Before extracting n-grams, the name is padded as `^name$`. This lets the model tell "ends in *a*" (`a$`) or "starts with *Sh*" (`^sh`) apart from "contains *a*" anywhere, so prefixes and suffixes come in automatically.
- **Explicit prefixes/suffixes:** the first 1, 2 and 3 letters and the last 1, 2 and 3 letters, as categorical features.
- **Numeric features:** name length, length², number of vowels, proportion of vowels, whether the name ends in a vowel, and whether it has a doubled letter.

## Encoding / scaling

- The n-grams come from `CountVectorizer(analyzer="char", ngram_range=(1,5), binary=True)`.
- The prefix/suffix features are one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`.
- The numeric features are standardized with `StandardScaler`.
- A `ColumnTransformer` combines them, and a `Pipeline` feeds the result into `LogisticRegression`. Because all preprocessing is inside the pipeline, it is refit on each cross-validation training fold. The same fitted pipeline transforms `test.csv`.

## How I evaluated models

Model choices were made using only `train.csv`, with 5-fold stratified cross-validation scored by log loss (the competition metric). First I compared feature sets of increasing complexity at C = 1. Then, for the best feature set, I tuned the regularization strength C, and tried elastic-net (`l1_ratio` = 0.5) and lasso (`l1_ratio` = 1).

| Feature set | CV log loss |
|---|---|
| all 0.5 | 0.6931 |
| constant = fraction female in train | [ ] |
| last letter only | [ ] |
| last 1–2 letters | [ ] |
| last 1–3 letters | [ ] |
| last 1–3 + first 1–3 | [ ] |
| + numeric features | [ ] |
| char 1–3 grams | [ ] |
| char 1–4 grams | [ ] |
| char 1–5 grams | [ ] |
| char 1–5 grams + suffixes + numeric | [ ] |

(Copy these numbers from the output of `hw3.py`.)

**Final model:** [feature set], logistic regression with ridge (L2) penalty, C = [ ], CV log loss [ ].

## Leaderboard score

**Public leaderboard log loss: [ ]**, compared with 0.69315 for the all-0.5 baseline. That is roughly a [ ]% reduction in log loss.

## What helped, what didn't, what surprised me

- **The last letter alone does most of the work.** Predicting from the last letter only cuts log loss from 0.693 to about [ ]. The single strongest feature is ending in *a* (`a$`), followed by ending in *i* and *e*. Male-leaning endings include *s*, *k*, *o* and *c*.
- **Character n-grams beat hand-picked features.** Letting the model learn weights for every letter sequence of length 1–5 beat my hand-built prefix/suffix and numeric features.
- **Adding the hand-built features on top of the n-grams made CV slightly *worse*.** They duplicate information the n-grams already contain, so they mostly add redundant parameters. More features did not mean a better model.
- **Regularization mattered a lot.** CV log loss was U-shaped in C: too much regularization (small C) underfits, too little (large C) overfits the thousands of sparse n-gram features. Moreover, overconfident wrong predictions are punished hard by log loss. L2 beat L1/elastic net, probably because many weak n-gram signals each contribute a little, and lasso throws some of them away.

## Why chasing the public leaderboard leads to overfitting

The public leaderboard score comes from a fixed ~30% subset of the test names. Each time I submit, see the score, and keep whichever variant scored best, I am using those names to make modeling decisions. In effect they become part of my training data. Over many submissions I start picking the model that happens to fit the quirks of *that particular subset*, including random noise, rather than the model that generalizes best. The public score then becomes optimistically biased, and the chosen model can do worse on the hidden 70% private set that determines the final ranking. Cross-validation on the training data is a better guide, because it averages over many held-out splits and doesn't use up the test set.
