# REALM Task 5.1 Post-Market Evaluation

## General Task Description

Components developed in Task 5.1 focus on the post-market evaluation of synthetically generated medical data that are used in various medical applications such as lung tumor segmentation, pharmacogenomics, blood glucose prediction, COPD prediction, etc.

The post-market evaluation is performed to ensure that the synthetic data generated is of high quality and is similar to the real data. To evaluate the quality of the synthetic data, examination along three main axes is performed:

1. **Expert Knowledge**: Evaluates the synthetic data based on domain-specific rules and medical knowledge to ensure anatomical correctness and clinical validity.
2. **Statistical Analysis**: Examines statistical and distributional properties of the synthetic data compared to ensure their validity from a statistical standpoint.
3. **Adversarial Evaluation**: Compares the performance of SOTA machine/deep learning models on the synthetic data with their performance on the real data to ensure that the two datasets (real and synthetic) yield comparable results.

## Use Case 3 Specific Description

This repository implements a comprehensive post-market evaluation pipeline for synthetic time series data in JSON format for blood glucose prediction, analyzing their quality and similarity to real data through three distinct evaluation approaches:

**Key Components:**

- **Expert Knowledge Evaluation**: Evaluates the feasibility of the blood glucose and insulin time series measurements and events based on the two following rules:
    - The valid humanly plausible ranges for blood glucose are [1.2, 110] mmol/L according to [Barry (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7242494/) and [Manappallil (2017)](http://researchgate.net/publication/317551782_The_survival_story_of_a_diabetic_ketoacidosis_patient_with_blood_sugar_levels_of_1985_mgdL).
    - According to [Walsh et al. (2014)](https://www.researchgate.net/publication/262787177_Confusion_Regarding_Duration_of_Insulin_Action_A_Potential_Source_for_Major_Insulin_Dose_Errors_by_Bolus_Calculators), subcutaneous insulin may not have been administered in the last 6 hours prior the period considered (so 12hours before the time of evaluation).

- **Statistical Analysis**: Conducts a thorough validation of the blood glucose and supporting insulin and nutrition time series data, as well as any supporting metadata in each patient's corresponding JSON file based on the following checks:
    - Required fields exist (diabeticStatus, startTime, bloodGlucose, insulinInfusion, insulinBolus, nutritionInfusion, nutritionBolus)
    - Both IV insulin and nutrition rates cannot be null at the same time
    - diabeticStatus has a valid value (0, 1, 2).
    - There exist at least 3 blood glucose measurements in the last 6 hours

- **Adversarial Evaluation**: TODO

## Getting Started

### Prerequisites

TODO

### Data Preparation

TODO

## Running Locally

TODO

## Kubeflow Pipeline Component

TODO

## Accessing the Generated Artifacts

TODO

## 📜 License & Usage

All rights reserved by MetaMinds Innovations.
