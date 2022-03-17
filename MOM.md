# Psykit Augment 

A Psychological signal augmentation library.

## Objective: 
To develop the augmentation library for EEG.

### Datasets:
---
1. DEAP
2. SEED

### Experiments
---
- Baseline Classifier
    -!(EEGNet)[https://github.com/vlawhern/arl-eegmodels]               
- Similar Samples Augmentation
1. WGAN
2. DCGAN
<!-- 3. Cyclic GAN
4. Conditional GAN -->

### Experiment Design:
----
* metric: f1-score

1. Leave one-out cross validation
    for n times:
        1. n => train(n-1); test(1)

2. Hyper parameter tuning (Bayesian Optimization)
    Tool:
        - Weights & Biases (Sweeps) (Optuna / Botorch)
        - Bayesian Optimization
            - Tree Parzen Estimator (TPE) Sampler

    Objective Function
        1. F1 Score of the validation set

3. Report Test Number

### Platform
---
1. Develop either on your laptop or google colab
2. Run on bigger GPUs
3. Pytorch
4. Optuna (TPE Bayesian Optimization)
5. Weights and Biases only for Logging


## Date: March 17, 2022
---

### TODOS:
---
- [ ] Add to the corresponding git repo - Chirag
- [ ] Create W&B account and add to the project - Chirag

- [ ] Download and explore the DEAP dataset - Adaya and Himalaya
- [ ] Baseline Results to be pushed to the W&B Project as a reference - Adaya and Himalaya

- [ ] WGAN -> Himalaya
- [ ] DCGAN -> Adaya
- [ ] cGAN -> Chirag
