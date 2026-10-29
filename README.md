# Hair loss classification

Supervised multiclass model that predicts whether a person has no hair loss, mild, moderate, or severe hair loss. The analysis and models are in `hair_health.ipynb`.

## The problem

The goal is a screening aid: given age, medical history, and hair-care habits, estimate which of the four hair-loss classes a person falls into. A clinician or a consumer screening tool could use that estimate to decide who needs a closer look, especially people who may already be in a more severe class.

Today this is done by a clinical visit. There is no model in this project that replaces that exam. The model only ranks risk from the recorded attributes.

This is offline supervised classification. The label `hair_loss_class` is known for every row, the data is a fixed table of 2,000 people, and the model is trained once rather than updated as new patients arrive. The four classes are ordered in severity, but the models treat them as four separate classes.

**Performance.** The main metric is macro-F1. Accuracy is reported only as a secondary number. Mild (928 people, 46%) and no hair loss (736, 37%) dominate the table. Moderate is 305 people (15%) and severe is 31 (1.6%). Accuracy can look strong while missing severe cases entirely. Macro-F1 gives each class equal weight, which matches the screening goal: missing the small severe group matters as much as getting the common classes right.

## Data Explorartion

`hair_health_dataset.csv` has 2,000 rows and no duplicate records. `hair_loss_class` is the target.

Attributes fall into three groups:

- **Numeric:** `age`, `bmi`, `sleep_hours`, `wash_frequency_per_week`, `protein_frequency_per_week`, `water_intake_liters`, `exercise_frequency_per_week`, plus `age_of_onset_years` and `hair_health_score` (both removed before modeling).
- **Ordered categories:** hair texture, density, fall frequency, fall level, scalp oiliness, dryness, itching, stress, alcohol use, heat styling, tight hairstyles, and environmental exposure.
- **Unordered categories:** gender, ethnicity, hair type, scalp type, dandruff, diet, smoking, nutritional deficiency, medical condition, hormonal life stage, hair-affecting medication, chemical treatment, recent weight loss, family history, and treatment history. `record_id` is an identifier.

`df.describe(include="all")` and the unique-value check did not turn up impossible categories or obvious data-entry noise. The only missing values are in `age_of_onset_years`: 736 rows, 36.8%. Those blanks are not random. Every blank row is `No_Hair_Loss`, and every mild, moderate, and severe row has an onset age.

Correlations with the ordered target (0 = no hair loss through 3 = severe):

| Attribute | Correlation |
|---|---:|
| `hair_health_score` | −0.92 |
| Family history (yes) | +0.46 |
| Alopecia areata | +0.25 |
| No medical condition | −0.25 |
| Age | +0.24 |
| Chemotherapy medication | +0.20 |
| No nutritional deficiency | −0.19 |
| Never smoked | −0.15 |

`hair_health_score` separates the four classes into non-overlapping ranges, so it is a restatement of the label. Family history is the strongest legitimate association. Age is higher in more severe classes on the box plot. The bar charts show family history, alopecia areata, chemotherapy, current smoking, and recent weight loss becoming more common as class severity increases, and “no medical condition,” “no deficiency,” and “never smoked” becoming less common.

**Promising transformations.** One-hot encode unordered categories. Map ordered categories to increasing integers. Do not impute onset age. Standardize features inside the model pipeline for models that use distances or coefficients.

**Extra data that would help.** More severe cases, a real clinical exam or photos, and how long hair loss has been present. Onset age among people who already have hair loss mostly repeats current age (correlation 0.97) and does not separate mild, moderate, and severe.

## Preparation

**Cleaning.** No outlier rows were removed. The category levels were consistent, and the numeric ranges were plausible.

`hair_health_score` was dropped. Its correlation of −0.92 and non-overlapping class ranges mean a model could classify everyone from that column alone. `age_of_onset_years` was dropped for a different reason: the blank is a perfect flag for no hair loss. The 736 rows stay in the data. Dropping them would delete the entire no-hair-loss class. `record_id` was dropped because it is an identifier.

**Encoding.** Unordered columns were one-hot encoded. Ordered columns, including the target used during exploration, were mapped with a fixed order (for example Fine < Medium < Coarse, and No hair loss < Mild < Moderate < Severe). Encoding was fit on the full table before the split. That does not leak the label, but a production pipeline should fit the encoder on the training fold only.

**Split and scaling.** Stratified 80/20 split, `random_state=42`, so each class keeps its share. The test set has 400 rows, including 6 of the 31 severe cases. `StandardScaler` and SMOTE sit inside an `imblearn` pipeline. SMOTE runs only on training folds during cross-validation. The test set is never resampled. Applying SMOTE before the split would copy synthetic neighbours of validation rows into training and inflate the scores.

Models were fit on every column left after the three drops, not only the five strongest EDA features. Those five (`family_history_Yes`, alopecia areata, no medical condition, age, and `smoking_status_Never`) are the associations the plots supported. Later tree importances pointed at the same family of columns.

## Measure and compare performance

Each model is a pipeline: scale, SMOTE, then the classifier. Comparison uses 5-fold stratified cross-validation on the training set, scored by macro-F1 (mean and standard deviation where a single untuned model was scored). The held-out test set is scored after that, and is not used to choose hyperparameters.

| Model | CV macro-F1 | CV std | Train macro-F1 | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.582 | 0.026 | — | 0.61 | 0.74 |
| KNN (k = 5) | 0.318 | 0.010 | — | 0.35 | — |
| Decision tree (depth 6) | 0.443 | — | 0.63 | 0.39 | 0.56 |
| Random forest (tuned) | 0.499 | 0.079 | 0.85 | 0.42 | 0.59 |
| XGBoost (tuned) | 0.530 | — | 0.90 | 0.52 | 0.73 |
| Logistic regression + XGBoost stack | 0.541 | — | 0.60 | 0.58 | 0.78 |

**Logistic regression** is the linear baseline. On the test set, no hair loss has precision 0.80, recall 0.86, and F1 0.83. Mild has precision 0.81, recall 0.68, and F1 0.74. Moderate precision is 0.56. It catches 3 of 6 severe cases (recall 0.50) but precision is 0.20, so most severe predictions are wrong. CV 0.58 and test 0.61 are close, so this fit is stable.

**KNN** is consistent and poor (CV 0.32, test 0.35). It recalls only about 15% of mild cases. Distances are weak across a mostly binary feature set, and SMOTE pulls neighbours toward the rare classes.

**Decision tree.** Depths from 3 to unlimited all land near CV macro-F1 0.42–0.44. Depth 6 is the best of that flat curve. Test macro-F1 is 0.39. The first split is family history, which matches the correlation screen. The tree catches none of the 6 severe test cases. Training macro-F1 0.63 against CV 0.44 shows remaining overfitting. With only 25 real severe rows in training, most of the SMOTE severe rows the tree sees are synthetic.

**What a human would use and these models do not.** A clinician would look at the scalp, the pattern of thinning, and how fast it changed. Those are not in the table. The models instead lean on family history, medical condition, smoking, and age. They confuse moderate with its neighbours, and severe counts are too small for a stable error rate.

**Shortlist, in order of usefulness for this metric.**

1. Logistic regression: best macro-F1, and the CV and test scores agree.
2. The logistic regression + XGBoost stack: best accuracy (0.78) and better moderate detection, but it misses every severe test case, so macro-F1 falls to 0.58.
3. XGBoost: accuracy 0.73, test macro-F1 0.52, and a large train-to-CV gap (0.90 vs 0.53).

KNN does not make a different, useful kind of error. The tree and forest are interpretable but less accurate, so they stay as explanations rather than candidates.

## Fine-tune the system

Grids were small, so grid search was used instead of random search. Every search scored macro-F1 with the same 5-fold stratified CV, and scaling and SMOTE stayed inside each fold. The selected CV score is optimistic relative to a fresh split, because it was the maximum over the grid.

**Decision tree.** The depth sweep above. Depth 6 was kept. It stopped the tree from memorizing the training set and did not raise accuracy.

**XGBoost.** Fixed settings were 500 rounds, depth 2, and learning rate 0.1. The grid chose `min_child_weight=1` and `reg_lambda=10`. Best CV macro-F1 was 0.530. Train macro-F1 of 0.90 against test 0.52 means the penalty did not remove overfitting. It did not beat logistic regression.

**Random forest.** 300 trees. Nine combinations of `max_depth` in {4, 6, 10} and `min_samples_leaf` in {1, 5, 10}. CV selected depth 10 and minimum 10 samples per leaf (CV macro-F1 0.499, std 0.079). Restricting the trees cut training macro-F1 from 1.00 on an unrestricted forest to 0.85, but test macro-F1 stayed at 0.42. The forest still overfits, and it still misses all 6 severe test cases.

Importances, added back from one-hot columns to the original field, rank family history (0.21), medical condition (0.14), smoking (0.08), dandruff (0.06), and nutritional deficiency (0.05). Age is next. `family_history_Yes` and `family_history_No` each take about 0.10 because they are the same fact split in two. Importance is not a direction: it does not say whether the feature pushes toward more hair loss.

**Stack.** The tuned XGBoost pipeline was stacked with logistic regression. Out-of-fold class probabilities train the final logistic regression. The stack CV of 0.541 is exploratory, because the XGBoost settings were already chosen on this same training data. Test accuracy rose to 0.78 and moderate F1 improved, but severe F1 went to zero and macro-F1 (0.58) stayed below logistic regression.

Test-set macro ROC curves for all six fits are written to `roc_curves/` by the last notebook section. They were not used to pick the model.

**Final model.** Logistic regression in the scale → SMOTE → classifier pipeline, with default regularization and `max_iter=2000`. Held-out macro-F1 is 0.61 and accuracy is 0.74, against a CV macro-F1 of 0.582 ± 0.026. No other model beat it on macro-F1. The hyperparameters were not chosen from the test set. The other models’ test scores were inspected as well, so the ranking among them is a comparison on one shared test set, not six independent confirmations.

## What worked and what did not

Dropping `hair_health_score` and `age_of_onset_years` was necessary. Keeping either one would have let the model read the label. Macro-F1, a stratified split, and SMOTE inside the pipeline kept the comparison from rewarding a model that only predicts mild or no hair loss.

KNN did not fit this feature mix. Depth limits and a larger forest did not close the gap with logistic regression. SMOTE balanced the training counts but could not invent new severe patients from 25 real training rows. Stacking raised accuracy and still lost on the metric that treats severe as equal to mild.

## Assumptions and limitations

- The label is treated as four classes. The integer codes are only a convenience. They are not an ordinal regression target.
- Family history, alopecia areata, and chemotherapy are associated with class. The models do not show that they cause hair loss.
- Severe has 31 people and 6 test cases. Recall of 0.50 is three people. A different split can move that number a lot.
- SMOTE assumes a synthetic neighbour is a plausible person. For severe, most training rows of that class are synthetic, and trees in particular learned rules that did not show up in the test set.
- One-hot encoding was fit before the split. Scaler and SMOTE were not.
- The EDA feature shortlist was not re-fit as its own model. The reported scores use the full cleaned feature set.
- There is no external clinic dataset. The test score is generalization to a random 20% of this same table, not to a new population.
- The model is not a diagnosis. A positive severe prediction is wrong about four times out of five even for the best model.

## Run the notebook

```bash
pip install -r requirements.txt
pip install xgboost
jupyter notebook hair_health.ipynb
```

`xgboost` is imported in the notebook and is not pinned in `requirements.txt`. The notebook reads `hair_health_dataset.csv` from the project directory.
