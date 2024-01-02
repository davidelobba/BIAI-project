This is the repository for the Bio Inspired AI project.

### Introduction

This project explores the use of advanced algorithms to generate images that deceive pretrained neural networks, specifically targeting ResNet18, ResNet34 and ResNet50 models. The study utilizes Genetic Algorithm (GA), Covariance Matrix Adaptation Evolution Strategy (CMA-ES), and Compositional Pattern Producing Network (CPPN) firstly in an unconstrained and then in a constrained environment to generate images that are meaningful to the neural networks but abstract to humans.

### Installation

In this section are presented the instructions for setting up the project environment and installing necessary dependencies.

```bash
# Create a virtual environment and activate it
python -m venv biai-venv
source biai-venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the experiments

```bash
# How to run GA evaluating ResNet50 on Cifar10 constrained on class airplane
python main.py  --algorithm ga --config-path config.yaml --weights-path weights --network resnet50 --output-dir results --dataset cifar10 --save --adaptive-mutation-crossover --class-constraint 0 --wandb
```