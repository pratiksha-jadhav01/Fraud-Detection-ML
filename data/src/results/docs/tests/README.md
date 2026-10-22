# Testing Documentation

## Purpose
This directory is intended for tests that help verify the reliability of the fraud detection prototype.

## Planned Tests
1. **Dataset loading:** Verify that the dataset loads successfully.
2. **Target validation:** Check that `FraudIndicator` contains the expected class labels, `0` and `1`.
3. **Preprocessing:** Verify that the prepared feature matrix contains valid values.
4. **Data splitting:** Confirm that training and testing data are separated before SMOTE is applied.
5. **Model prediction:** Check that trained models return valid predictions.
6. **Evaluation:** Verify that the required evaluation metrics can be calculated.

## Current Status
These are planned tests. Individual automated test scripts should be added as they are implemented.
