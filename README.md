# BCI Competition 2020 Track 3 Imagined Speech EEG Dataset

This project uses the imagined speech EEG dataset from BCI Competition 2020 Track 3.

The original dataset is available from the official OSF repository:

[BCI Competition 2020 Track 3 Dataset](https://osf.io/pq7vb/overview)

The official MATLAB-format data were organized into three NumPy files:

- `data.npy`
- `label.npy`
- `data_subject.npy`

These three files must have the same trial order and matching first dimensions.

For a given trial index `i`, `data[i]`, `label[i]`, and `data_subject[i]` represent the EEG data, class label, and subject identifier of the same trial, respectively.

## Data Files

### `data.npy`

- Shape: `(5250, 64, 795)`
- Data type: `float64`

The dimensions represent:

- Number of trials: `5250`
- Number of EEG channels: `64`
- Number of time points: `795`

The data are stored in the following format:

`(number of trials, number of channels, number of time points)`

### `label.npy`

- Shape: `(5250,)`
- Data type: `int64`

This file stores the class label for each trial.

The project uses zero-based class labels:

| Label | Official class |
|---:|---|
| 0 | Hello |
| 1 | Help me |
| 2 | Stop |
| 3 | Thank you |
| 4 | Yes |

### `data_subject.npy`

- Shape: `(5250,)`
- Data type: `int64`

This file stores the subject identifier corresponding to each trial.

## Data Preparation

The NumPy files were generated from the official Track 3 imagined speech data using the following procedure:

1. The Track 3 imagined speech training and validation data were downloaded from the official OSF repository.
2. The EEG data and class labels were read from the MATLAB files for each subject.
3. The original data format, `(time, channels, trials)`, was transposed to `(trials, channels, time)`.
4. The training and validation trials were concatenated for each subject.
5. The data from all 15 subjects were concatenated along the trial dimension.
6. The resulting EEG data were saved as `data.npy`.
7. The corresponding class labels were saved as `label.npy`.
8. The corresponding subject identifiers were saved as `data_subject.npy`.

## File Placement

The three NumPy files should be placed in the same directory as the main program:

```text
./IIT-IS.py
./data.npy
./label.npy
./data_subject.npy
```

The `data.npy` file is not included in the repository because of its large file size. Users should download the original data from OSF and prepare the NumPy files according to the procedure described above.

