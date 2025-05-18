# Deep Learning Assignment 2

## Project Directory Structure

assignment 2/
├── dataset/
│   ├── train_images.npy
│   └── test_images.npy
├── models/
│   ├── vae_lstm_conv_decoder.pt
│   ├── vae_lstm_conv_full.pt
│   ├── vae_conv_conv_decoder.pt
│   ├── vae_conv_conv_full.pt
│   ├── cvae_conv_conv_full.pt
│   └── classifier_a1.pt
├── lstm_ae.py
├── cae.py
├── classifier.py
├── compare.py
├── conditional_vae.py
├── cvae_second.py
├── vae_conv_encoder_decode.py
└── vae_rnn_encoder_conv_decoder.py


## Task 1

### Dependencies:
    pip install torch numpy matplotlib scikit-learn torchinfo

### How to Run (Task 1)

1. Train LSTM Autoencoder and get t-SNE visualization:
    python lstm_ae.py

1. Train Convolutional Autoencoder and get t-SNE visualization:
    python cae.py



## Task 2

### Dependencies:
    pip install torch torchvision numpy matplotlib torchmetrics torchinfo

### How to Run (Task 2)

1. Train the classifier:
    python classifier.py

2. Train the LSTM + Conv VAE:
    python vae_rnn_encoder_conv_decoder.py

3. Train the Conv + Conv VAE:
    python vae_conv_encoder_decode.py

4. Train the Conditional VAE (3 classes: rabbit, yoga, snowman):
    python conditional_vae.py

5. Generate conditional samples and classify them:
    python cvae_second.py

6. Compare models using IS and FID:
    python compare.py

## Notes
- Run training scripts first to generate required model files before generation/evaluation.
