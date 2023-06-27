# M5: Retail demand forecasting

## Available models

The models that can be trained on the M5 data are:

- NHiTS ([arXiv](https://arxiv.org/abs/2201.12886))
- NBEATS ([arXiv](https://arxiv.org/abs/1905.10437))
- TFT ([arXiv](https://arxiv.org/abs/1912.09363))

## Model training

```bash
./start.sh training -m <model>
```

The training will populate the following folders:

- `training_logs/ckpts/<model>`: folder containing checkpoints
- `training_logs/logs/<model>`: tensorboard logs

Inspect the tensorboard logs with:

```bash
tensorboard --port <port> --logdir training_logs/logs
```

## Model inference

Run the inference pipeline for one of the available models passing the trained
model's checkpoint.

```bash
./start.sh inference -m <model> -p training_logs/ckpts/<model>/trained_<model>.ckpt
```

The output is a `.csv` file stored to the
`./training_logs/<model>/<model>_preds.csv` file.
