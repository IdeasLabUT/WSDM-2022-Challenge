# WSDM-2022-Challenge
IDEAS Lab UT's submission to the WSDM 2022 Temporal Link Prediction challenge.

## Generate Predictions

To generate our predictions, run the following two Jupyter notebooks:

1. `construct_features.ipynb`: This will construct the edge features for the final test set and save them to CSV.
2. `predict_edges.ipynb`: This will train a logistic regression model on the edge features and generate predictions on the final test set.

The feature construction may take a few hours, so we have included also the constructed features with the filenames beginning with `featureA_` and `featureB_` so you can directly run `predict_edges.ipynb` to generate predictions from our pre-computed features.

## Input Files

The following input data files are assumed to be in the root directory:

### Dataset A

- `edges_train_A.csv`
- `node_features.csv`
- `edge_type_features.csv`
- `input_A_initial.csv`
- `input_A.csv`

### Dataset B

- `edges_train_B.csv`
- `input_B_initial.csv`
- `input_B.csv`

## Output Files

Predictions will be generated also in the root directory:

- `output_A.csv`
- `output_B.csv`

## Dependencies

See `requirements.txt` for required packages. The code for the CHIP model is included in the directory `CHIP-Network-Model`, which is added to the system path when loading `chip_features.py`.
