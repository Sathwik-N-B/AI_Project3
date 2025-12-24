# Mushroom Classification using Naive Bayes

This project implements a Naive Bayes classifier from scratch to predict whether a mushroom is edible or poisonous based on its physical characteristics.

## Dataset

The project uses the **Mushroom Dataset** from the UCI Machine Learning Repository:
- **Source**: [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data)
- **Instances**: 8,124 mushroom samples (after cleaning: 5,644)
- **Features**: 22 categorical attributes describing physical characteristics
- **Target**: Binary classification (edible or poisonous)

### Features Include:
- Cap shape, surface, and color
- Bruises presence
- Odor
- Gill attachment, spacing, size, and color
- Stalk shape, root, surface, and color
- Veil type and color
- Ring number and type
- Spore print color
- Population and habitat

## Implementation

### Naive Bayes Classifier
The implementation includes:
- **Manual encoding** of categorical variables
- **Laplace smoothing** (add-one smoothing) to handle unseen feature values
- **Log probabilities** to prevent numerical underflow
- **From-scratch implementation** without using scikit-learn

### Training Process
1. Calculate class priors P(class)
2. Calculate conditional probabilities P(feature|class) with Laplace smoothing
3. Use log probabilities for numerical stability

### Prediction Process
For each test instance:
- Calculate log posterior probability for each class
- Predict the class with maximum log posterior

## Tasks

### Task 1: Classification with All Features
- Uses all 22 features for classification
- Train/test split: 60/40
- Reports accuracy on test set

### Task 2: Classification with Selected Features
- Uses only 10 carefully selected features:
  - `odor`, `gill-color`, `spore-print-color`, `ring-type`, `gill-size`
  - `bruises`, `population`, `habitat`, `cap-color`, `cap-surface`
- Train/test split: 60/40
- Compares accuracy with full feature set

## Requirements

```
pandas
numpy
```

## Usage

### Running the Jupyter Notebook
1. Open `sathwik_AI_project3.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially
3. Observe the accuracy results for both tasks

### Running the Python Script
```bash
python sathwik_ai_project3.py
```

## Results

The implementation outputs:
- **Task 1 Accuracy**: Classification accuracy using all 22 features
- **Task 2 Accuracy**: Classification accuracy using selected 10 features

## Project Structure

```
AI_Project3/
├── README.md                      # Project documentation
├── sathwik_AI_project3.ipynb     # Jupyter notebook implementation
├── sathwik_ai_project3.py        # Python script version
└── mushrooms.csv                 # Dataset (if downloaded locally)
```

## Key Concepts

- **Naive Bayes Assumption**: Features are conditionally independent given the class
- **Laplace Smoothing**: Adds 1 to all counts to avoid zero probabilities
- **Log Space Computation**: Prevents numerical underflow when multiplying many small probabilities
- **Categorical Encoding**: Converts categorical features to numerical values for processing

## License

This project uses publicly available data from the UCI Machine Learning Repository.
