# 🏥 AI for Healthcare: Patient Selection for Diabetes Drug Clinical Trials

### Healthcare AI • Electronic Health Records • Clinical Trial Selection • TensorFlow Probability • Fairness Analysis

![Python](https://img.shields.io/badge/Python-Healthcare_AI-blue?style=for-the-badge\&logo=python\&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Clinical_Prediction-orange?style=for-the-badge\&logo=tensorflow\&logoColor=white)
![Healthcare](https://img.shields.io/badge/Healthcare-EHR_Analytics-green?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine_Learning-Patient_Selection-purple?style=for-the-badge)
![Fairness](https://img.shields.io/badge/AI-Fairness_Analysis-red?style=for-the-badge)

---

# Overview

This project explores how machine learning can assist clinical trial planning by identifying patients who may be suitable candidates for a diabetes drug study.

The objective was to build a predictive model capable of estimating hospitalization duration using Electronic Health Record (EHR) data and then use those predictions to support patient selection decisions for a hypothetical Phase III clinical trial.

The project demonstrates the complete healthcare machine learning workflow:

* Exploratory Data Analysis
* Clinical feature engineering
* Healthcare data preparation
* Predictive modeling
* Uncertainty estimation
* Bias and fairness assessment

---

# Clinical Motivation

The hypothetical diabetes treatment requires:

* Hospital admission
* Frequent monitoring
* Medication adherence training
* Multi-day observation

Not every patient is an appropriate candidate.

Selecting patients likely to remain hospitalized for the required duration can:

* Improve trial efficiency
* Reduce operational costs
* Increase patient safety
* Improve treatment adherence

Machine learning can support these decisions by identifying patterns in historical healthcare records.

---

# Project Objectives

Build a system capable of:

1. Predicting expected hospitalization duration.
2. Identifying potential trial candidates.
3. Estimating prediction uncertainty.
4. Evaluating fairness across demographic groups.

---

# Dataset

The project uses a modified version of the well-known UCI Diabetes dataset.

Dataset characteristics include:

* Hospital encounters
* Demographic information
* Diagnoses
* Procedures
* Medication history
* Clinical outcomes

The dataset was transformed from line-level healthcare records into encounter-level and patient-level representations appropriate for machine learning.

---

# Project Workflow

```text
Raw EHR Data
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Regression Model
      │
      ▼
Length of Stay Prediction
      │
      ▼
Patient Selection
      │
      ▼
Fairness Analysis
```

---

# Exploratory Data Analysis

The project investigated:

## Missing Values

Identification of fields with:

* High null rates
* High sparsity
* Limited predictive value

Examples included:

* Weight
* Payer information

These features were evaluated before inclusion in the model.

---

## Demographic Analysis

Patient distributions were explored across:

* Age
* Gender
* Admission characteristics

This analysis helped identify potential dataset imbalances and informed feature engineering decisions.

---

## High Cardinality Features

Healthcare datasets often contain large medical coding systems.

Examples include:

* Diagnosis codes
* Drug identifiers
* Procedure codes

The project examined high-cardinality fields and developed strategies for reducing dimensionality while preserving clinical information.

---

# Feature Engineering

## Medical Code Mapping

National Drug Codes (NDCs) were mapped to generic drug names to create clinically meaningful features.

This step reduced dimensionality while improving interpretability.

---

## Numerical Features

Examples included:

* Number of medications
* Number of procedures
* Number of diagnoses
* Laboratory measurements

Features were normalized prior to training.

---

## Categorical Features

TensorFlow Feature Columns were used to encode:

* Race
* Gender
* Admission type
* Medication information
* Other clinical variables

---

# Data Leakage Prevention

Healthcare datasets require special handling to avoid leakage.

The project ensured:

* Patient-level separation
* Encounter-level separation
* Independent train, validation and test sets

This prevents the model from learning information about the same patient across multiple datasets.

---

# Predictive Modeling

## Regression Task

The primary objective was to predict:

```text
Expected Length of Hospital Stay
```

This prediction was then converted into a patient selection decision.

---

## Uncertainty Estimation

TensorFlow Probability was used to estimate:

* Prediction mean
* Prediction standard deviation

This provides a confidence estimate alongside each prediction.

In clinical applications, uncertainty is often as important as the prediction itself.

---

# Model Evaluation

Performance was evaluated using:

* AUC
* Precision
* Recall
* F1 Score

These metrics provide a balanced assessment of model utility for patient selection.

---

# Fairness & Bias Analysis

Healthcare AI systems must be evaluated for demographic fairness.

The project used the Aequitas toolkit to analyze:

## Race

Evaluation of prediction quality across racial groups.

## Gender

Evaluation of prediction quality across gender groups.

Metrics included:

* True Positive Rate
* False Positive Rate
* Group comparisons

This analysis helps identify potential disparities before deployment.

---

# Skills Demonstrated

## Healthcare Analytics

* Electronic Health Records
* Clinical trial design
* Patient cohort selection
* Medical code systems

## Machine Learning

* Regression modeling
* Feature engineering
* TensorFlow Probability
* Uncertainty estimation

## Data Engineering

* Healthcare data cleaning
* Encounter aggregation
* Leakage prevention
* High-cardinality feature reduction

## Responsible AI

* Fairness analysis
* Bias reporting
* Healthcare ethics
* Clinical deployment considerations

---

# Key Deliverables

* Exploratory Data Analysis
* Healthcare Feature Engineering Pipeline
* Hospitalization Duration Prediction Model
* Patient Selection Framework
* Uncertainty Estimation Analysis
* Aequitas Fairness Report
* Clinical Interpretation and Recommendations

---

# Educational Context

This project was completed as part of the AI for Healthcare Nanodegree.

It focuses on the practical challenges of applying machine learning to Electronic Health Records and demonstrates how predictive models can support clinical trial operations while maintaining awareness of fairness, bias, and uncertainty.

---

## Author

S. Palis

AI Systems • Healthcare AI • Clinical Machine Learning • Responsible AI
