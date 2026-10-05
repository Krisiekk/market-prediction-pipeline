# ARX-style regression for executable H5

## Model and validation

Tested a regularized ARX-style regression for the continuous return from
`open(t+1)` to `close(t+5)`. Inputs combine the existing TECHNICAL + A+C features
with five lagged weekday XAU close-to-close returns. A `StandardScaler` and Ridge
regression predict the continuous forward return; the direction is its sign.
This is a linear ARX-style supervised forecast, not ARIMA. `statsmodels` is not
installed in the project environment, so this experiment uses the existing
scikit-learn dependency and adds no package requirement.

Five Ridge penalties (`alpha=0.01, 0.1, 1, 10, 100`) are compared inside each
outer training fold using 3 expanding chronological inner folds, `gap=5`, and
negative mean squared error. The selected model is evaluated on the untouched
outer fold. The same saved five-voter OOF predictions are joined by date and
checked for matching labels/returns.

The six-voter comparison uses the original five models plus ARX. A 3–3 tie
abstains (no signal). On the non-tied dates, accuracy is compared with both the
five-voter decision and Always UP on exactly those same dates.

## Results

| Model | Mean AUC | AUC > 0.5 folds | Directional accuracy | Balanced accuracy | Additional result |
| --- | ---: | ---: | ---: | ---: | --- |
| ARX Ridge | **0.5366** | **5/5** | 0.4931 | 0.5140 | Mean R² = -0.1174 |
| Five-voter majority | — | — | 0.4853 | 0.5065 | Coverage 100% |
| Six voters, abstain on 3–3 tie | — | — | 0.4944 | 0.5128 | Coverage 87.6% |
| Five-voter majority on those same covered dates | — | — | 0.4944 | 0.5128 | Coverage 87.6% |
| Always UP on those same covered dates | — | — | **0.5454** | 0.5000 | Coverage 87.6% |

ARX ranks observations modestly above chance by AUC, but its zero-return sign
decision is below 50% accurate and its negative R² means it predicts continuous
returns worse, on average across folds, than the reference given by each test
fold's own mean in the R² definition. This is not a separately evaluated,
deployable training-mean forecast. Adding it as an equal sixth
hard voter does not improve the direction of any non-tied vote: when the existing
five-voter majority is 3–2, an opposing sixth vote creates a 3–3 tie; otherwise
the old majority remains. Abstaining on these close disagreements raises
accuracy relative to its full-sample result, but the five-voter model has exactly
the same accuracy on those covered dates. Always UP performs better on that subset.

**Decision:** do not add ARX to the hard-voting ensemble at this stage. It can
remain a research candidate for ranking or for a separately designed abstention
rule, but these results do not support a trading edge.

These are development walk-forward results on previously explored data, not a
holdout. No costs, threshold optimization, or live/paper test were run.

## Reproduction and output

```bash
.venv/bin/python -m src.models.evaluate_h5_arx
.venv/bin/python -m unittest tests.test_h5_execution_arx -v
```

Code: `src/models/evaluate_h5_arx.py`. Machine outputs:
`data/processed/h5_execution_arx/`.
