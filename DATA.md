# Data

The runner expects a prepared directory for one dataset:

```text
prepared_dataset/
  config.txt
  data/
    neg_exer_ours.json
    Eedi/
      val_set.json
      test_set.json
```

`Eedi` is the loader's fixed directory name for validation and test files, including when another dataset is used. It does not identify the dataset inside those files.

## Configuration

`config.txt` has a header and a comma-separated second line:

```text
students,items,concepts
100,200,10
```

Counts must cover all referenced identifiers.

## Training responses

`data/neg_exer_ours.json` is a JSON array:

```json
[{"user_id": 0, "exer_id": 0, "score": 1, "knowledge_code": [1]}]
```

## Validation and test responses

Each file is a JSON array grouped by student:

```json
[{"user_id": 1, "logs": [{"exer_id": 2, "score": 0, "knowledge_code": [1]}]}]
```

Training student/item IDs are zero-based by default; validation/test IDs are one-based. Set `--train-id-base` and `--eval-id-base` explicitly for a different convention. Concept IDs are one-based in all inputs. Responses must be binary and each exercise must have at least one concept.

The Q-matrix is assembled from supplied exercise–concept metadata across partitions; validation and test response labels do not enter the training objective. Candidate pools exclude each student's observed training exercises.

Use the actual released data version, preprocessing rules, and saved partitions corresponding to the result being reproduced. Dataset acquisition, exact source partitions, raw responses, and checkpoints are not bundled in this release. Do not treat the example schema as sufficient to recreate the published partitions or redistribute third-party data without permission.

