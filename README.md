

# Generative Adversarial Nets [![hb-research](https://img.shields.io/badge/hb--research-experiment-green.svg?style=flat&colorA=448C57&colorB=555555)](https://github.com/hb-research)

TensorFlow implementation of [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661).

![images](images/gan-architecture.jpeg)

- [images source](http://www.kdnuggets.com/2017/01/generative-adversarial-networks-hot-topic-machine-learning.html)

## Requirements

- Python 3.6
- TensorFlow >= 1.4
- [hb-config](https://github.com/hb-research/hb-config) (Singleton Config)
- requests
- [Slack Incoming Webhook URL](https://my.slack.com/services/new/incoming-webhook/)
- Matplotlib


## Project Structure

init Project by [hb-base](https://github.com/hb-research/hb-base)

    .
    ├── config                  # Config files (.yml, .json) using with hb-config
    ├── data                    # dataset path
    ├── generative_adversarial_nets   # GAN architecture graphs (from input to logits)
        └── __init__.py               # Graph logic
    ├── data_loader.py          # download data -> generate_batch (using Dataset)
    ├── main.py                 # define experiment_fn
    └── model.py                # define EstimatorSpec

Reference : [hb-config](https://github.com/hb-research/hb-config), [Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator), [experiments_fn](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment), [EstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec)

## To do

- Using [GANEstimator](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/gan/estimator/GANEstimator)
- Since it is not currently compatible with the GANESTimator and [Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment), I will be working on it later.


## Config

Can control all **Experimental environment**.

example: mnist.yml

```yml
model:
  batch_size: 32
  z_dim: 20
  n_output: 784

  encoder_h1: 512
  encoder_h2: 256
  encoder_h3: 128

  decoder_h1: 128
  decoder_h2: 256
  decoder_h3: 512

train:
  learning_rate: 0.00001
  optimizer: 'Adam'                # Adagrad, Adam, Ftrl, Momentum, RMSProp, SGD

  train_steps: 200000
  model_dir: 'logs/mnist'

  save_checkpoints_steps: 1000
  check_hook_n_iter: 1000
  min_eval_frequency: 10

  print_verbose: True
  debug: False

slack:
  webhook_url: ""                   # after training notify you using slack-webhook
```

* debug mode : using [tfdbg](https://www.tensorflow.org/programmers_guide/debugger)


## Usage

Install requirements.

```pip install -r requirements.txt```

Then, start training model

```
python main.py --config mnist
```

After training, generate image from latent vector.

```
python generate.py --config mnist --batch_size 100
```


### Experiments modes

:white_check_mark: : Working  
:white_medium_small_square: : Not tested yet.

- : white_medium_small_square: `evaluate` : Evaluate on the evaluation data.
- :white_medium_small_square: `extend_train_hooks` :  Extends the hooks for training.
- :white_medium_small_square: `reset_export_strategies` : Resets the export strategies with the new_export_strategies.
- :white_medium_small_square: `run_std_server` : Starts a TensorFlow server and joins the serving thread.
- :white_medium_small_square: `test` : Tests training, evaluating and exporting the estimator for a single step.
- : white_medium_small_square: `train` : Fit the estimator using the training data.
- : white_medium_small_square: `train_and_evaluate` : Interleaves training and evaluation.

---


### Tensorboar

```tensorboard --logdir logs```

## Reference
- [hb-research/notes - Generative Adversarial Nets](https://github.com/hb-research/notes/blob/master/notes/gan.md)
- [Paper - Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
- [TensorFlow-GAN (TFGAN)](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/gan) (GANEstimator)

## Author

[Dongjun Lee](https://github.com/DongjunLee) (humanbrain.djlee@gmail.com)
