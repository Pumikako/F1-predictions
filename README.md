# Formula 1 Teammate Prediction

A supervised machine-learning project that predicts which Formula 1 driver is more likely to finish ahead of their teammate in a race.

## Business question

> Using information available before a race, can we predict which driver will outperform their teammate?

This project does **not** predict the exact finishing position. It focuses only on the head-to-head result between teammates in the same constructor.

## Unit of analysis and target

Each row represents **one driver in one race**.

The target variable is `beat_teammate`:

- `Y`: the driver finished ahead of their teammate.
- `N`: the driver finished behind their teammate.

Only driver-race-team groups with exactly two teammates are included.

## Data source

The project uses the [Formula 1 World Championship dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020), containing historical Formula 1 race information.

The analysis uses:

- `results.csv`
- `races.csv`
- `drivers.csv`
- `constructors.csv`

## Project structure

```text
F1-predictions/
├── Main.ipynb
├── README.md
├── requirements.txt
├── src/
│   └── features.py
└── data/
    └── raw/
        ├── results.csv
        ├── races.csv
        ├── drivers.csv
        └── constructors.csv
```

Raw data files are not included in the repository. Download them and place them inside `data/raw/`.

## Setup

Install the required libraries:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook from the project folder:

```bash
jupyter notebook Main.ipynb
```

Run the notebook cells from top to bottom.

## Data loading

The notebook uses relative paths, so the project can run on another computer without changing personal file paths.

```python
from pathlib import Path

project_folder = Path.cwd()
data_folder = project_folder / "data" / "raw"

results = pd.read_csv(data_folder / "results.csv")
races = pd.read_csv(data_folder / "races.csv")
drivers = pd.read_csv(data_folder / "drivers.csv")
constructors = pd.read_csv(data_folder / "constructors.csv")
```

## Data preparation

The raw datasets are merged using their shared identifiers:

- `raceId`
- `driverId`
- `constructorId`

The dataset is sorted chronologically by `year` and `round` before creating historical features.

## Features

The final model uses pre-race information only:

- starting grid position;
- grid gap to teammate;
- season and race round;
- driver age;
- prior driver points in the current season;
- prior points gap to teammate;
- prior teammate wins;
- prior teammate win rate;
- prior Formula 1 seasons;
- prior Formula 1 races;
- encoded driver ID.

Historical variables use only data from races before the current race. This prevents target leakage.

## Models tested

Two classification models were compared:

1. **K-Nearest Neighbors**
   - numerical features standardised with `StandardScaler`;
   - driver ID encoded with `OneHotEncoder`.

2. **Decision Tree Classifier**
   - tuned using `GridSearchCV`;
   - best parameter: `max_depth=3`.

3. **Random Forest Classifier**
   - an ensemble-model comparison using the same train/test split;
   - evaluated in the notebook alongside the other models.

## Model evaluation

The dataset was divided into training and test sets using:

- `test_size=0.20`
- `random_state=28`
- stratified target splitting.

Models were evaluated using:

- accuracy;
- confusion matrix;
- precision;
- recall;
- F1-score.

## Results

| Model | Test accuracy |
|---|---:|
| KNN with StandardScaler | 58.39% |
| Decision Tree, `max_depth=3` | **60.14%** |
| Random Forest, 200 trees | 59.42% |

The Decision Tree was selected as the final model.

GridSearchCV confirmed that `max_depth=3` was the best configuration, with a mean five-fold cross-validation accuracy of **62.15%** on the training data.

The Random Forest improved on the KNN baseline but did not outperform the tuned Decision Tree, so the Decision Tree remains the final model.

## Data-quality review

The notebook checks missing values in the final model columns and lists, without automatically removing, any duplicated `raceId` + `driverId` records.

One historical case was identified: Harald Ertl appears in the 1978 Italian Grand Prix results for two different constructors (ATS and Ensign). The record is retained for manual review rather than silently deleted.

## Presentation

The final presentation is available on [Google Slides](https://docs.google.com/presentation/d/1bBtC91_eTNXAIYMdahWHqKs9j3kxqiw5MucHEhqr9ks/edit?slide=id.p5#slide=id.p5).

## Conclusion

The final model correctly predicts the teammate head-to-head result in approximately 60% of unseen historical races.

This is better than the KNN baseline, but it should be interpreted carefully: Formula 1 results are influenced by unpredictable events such as reliability issues, weather, strategy, safety cars, and race incidents.

## Limitations and future work

- The model predicts teammate performance, not exact finishing position.
- The target is based on historical race results and does not guarantee a correct prediction for an individual future race.
- Formula 1 regulations, point systems, cars, teams, and driver line-ups change over time.
- Future improvements could include a strictly time-based train/test split, where earlier seasons are used for training and later seasons are held out for testing.
- Future versions could test additional valid pre-race information, such as updated weather forecasts or current-season data.
