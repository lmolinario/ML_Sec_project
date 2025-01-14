# Machine Learning Security Project
This project is developed for the UNICA.IT University Machine Learning Security exam. 

> **Master's degree in Computer Engineering, Cybersecurity and Artificial Intelligence - University of Cagliari**

> **Machine Learning Security - Supervisor: Prof. Battista Biggio**

> **Author**: Lello Molinario: 70/90/00369


***
# Table of Contents
1. [Installation](#installation)
2. [Project Goal](#project-goal)
4. [Solution Design](#solution-design)
5. [Conclusions](#conclusions)


***
## Installation

- Download the ZIP code or clone the repository inside Raspberry PI 3 B+ with
  ```bash
  git clone https://github.com/lmolinario/ML_Sec_project.git
  ```
- Install the requirements with

  ```bash
  pip3 install -r requirements.txt
  ```

- Run the file `entrypoint.py` to start the program (in UNIX-like System)

### Before running it, make sure you have Python 3.9 installed.

## Project goal
The goal of this project is to re-evaluate 5 RobustBench models using another attack algorithm (e.g., FMN) and identify samples for which one attack succeeds while the other fails. In other words, we aim to compare the effectiveness of different attacks against robust models, to analyze in which cases one type of attack is effective while another fails, thus contributing to a deeper understanding of the robustness of models and attack algorithms.

FMN (Fast Gradient Method)

FMN, also referred to as FGSM (Fast Gradient Sign Method), is one of the most common attacks against neural networks. The basic idea is to compute the gradient of the model with respect to the input image and add a perturbation in the direction of the gradient to maximize the loss. This type of attack can be implemented as follows:

δ=ϵ⋅sign(∇x​J(θ,x,y))

Where:

    δ is the generated perturbation.
    ϵ is the perturbation magnitude (i.e., the strength of the attack).
    ∇xJ(θ,x,y) is the gradient of the loss function J with respect to the input image x, computed for the model parameters θ.
    sign(⋅) refers to the function that takes the sign of each gradient value.


## Solution Design
#### Modularity: 
The project will be structured in a modular way to allow the replacement of attack models and algorithms without having to redo the entire flow. Each component (model, attack, evaluation) will be a separate module.
#### Scalability: 
The system will be scalable to be able to add more RobustBench models or try different attack algorithms in the future.


## Conclusions

