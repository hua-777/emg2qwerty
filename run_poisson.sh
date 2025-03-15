#!/bin/bash

python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu trainer.devices=1 \
  model=transformer_ctc \
  transforms=log_spectrogram_with_poisson \
  --multirun

python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu trainer.devices=1 \
  model=lstm_ctc \
  transforms=log_spectrogram_with_poisson \
  --multirun

python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu trainer.devices=1 \
  model=cnn_lstm_ctc \
  transforms=log_spectrogram_with_poisson \
  --multirun


python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu trainer.devices=1 \
  model=tds_conv_ctc \
  transforms=log_spectrogram_with_poisson \
  --multirun