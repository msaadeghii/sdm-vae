# SDM-VAE

This is the PyTorch implementation of the SDM-VAE model inspired by [this repository](https://github.com/XiaoyuBIE1994/DVAE):

## VAE Models

The standard VAE and SDM-VAE models are provided in `./model`. 

## Training

Set the training properties (e.g., network architectures, STFT parameters, etc.) in the `config` files provided inside `./configuration`. The training then proceeds as follows:

```
# Train a VAE model:
python train_model.py --cfg ./configuration/cfg_VAE.ini

# Train an SDM-VAE model:
python train_model.py --cfg ./configuration/cfg_SDM_VAE.ini
```

## Evaluation

You can evaluate a trained VAE model on speech generative modeling via `test_speech_analysis_resynthesis.py`.

## Reference

[*] Mostafa Sadeghi and Paul Magron, "[A Sparsity-promoting Dictionary Model for Variational Autoencoders](https://arxiv.org/abs/2203.15758)," Interspeech 2022 (submitted).


