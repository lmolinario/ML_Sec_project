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
### Before running it, make sure you have Python 3.10 installed for compatibility between all libraries.

- Download the ZIP code or clone the repository inside Raspberry PI 3 B+ with
  ```bash
  git clone https://github.com/lmolinario/ML_Sec_project.git
  ```
- Install the requirements with

  ```bash
  python3.10 -m pip install git+https://github.com/RobustBench/robustbench.git

  git clone https://github.com/alirezaabdollahpour/SuperDeepFool.git SuperDeepFool


- Modificare il file requirements.txt per ignorare le versioni specifiche non compatibili
  ```bash
  sed -i '/numpy==1.19.0/d' SuperDeepFool/requirements.txt
  sed -i '/matplotlib==3.4.2/d' SuperDeepFool/requirements.txt
  sed -i '/numpy==1.19.0/d' SuperDeepFool/requirements.txt
  
-Installare il resto delle dipendenze
  ```bash
  pip3 install -r SuperDeepFool/requirements.txt

-Installare il resto delle dipendenze
  ```bash
  pip3 install -r requirements.txt
  
- Make sure you have the following libraries installed:

  ```bash
  pip3 install git+https://github.com/pralab/secml
  pip3 install git+https://github.com/RobustBench/robustbench.git
  ```
- Run the file `entrypoint.py` to start the program 


## Project goal
The goal of this project is to re-evaluate 5 RobustBench models using another attack algorithm (e.g., FMN) and identify samples for which one attack succeeds while the other fails. In other words, we aim to compare the effectiveness of different attacks against robust models, to analyze in which cases one type of attack is effective while another fails, thus contributing to a deeper understanding of the robustness of models and attack algorithms.

## Solution Design
#### Attack algorithm
As indicated in our project I took as a reference the FMN attack, also known as FGSM (Fast Gradient Sign Method), which is one of the most common attacks against neural networks.
The basic idea of ​​this attack is to calculate the gradient of the model with respect to the input image and add a perturbation in the direction of the gradient to maximize the loss. This type of attack can be implemented as follows:

 ![δ=ϵ⋅sign(∇xJ(θ,x,y))](misc/Formula FMN.png)

Where:

δ is the generated perturbation.

ϵ is the magnitude of the perturbation (i.e. the strength of the attack).

∇xJ(θ,x,y) is the gradient of the loss function J with respect to the input image x, calculated for the model parameters θ.

sign(⋅) refers to the function that takes the sign of each gradient value.

The norm is a fundamental concept in all adversarial attacks.
It defines how to measure and bound the size of the perturbation added to the original image. It is a mathematical measure that establishes the "strength" of the perturbation and is used to control how much the original image is modified.

In our case of FMN attack, the norm directly influences the creation of the adversarial perturbation and its control.
Depending on the type of norm chosen, the perturbation can have different characteristics:

L2 norm (Euclidean): measures the Euclidean distance between the original and perturbed images. The L2 norm is generally sensitive to large changes in the images, but may not capture very small local perturbations that affect recognition.

L∞ norm (maximum norm): measures the maximum difference for each pixel between the original and perturbed images. The L∞ norm is useful for testing robustness against attacks that limit the perturbation to the maximum value for each pixel.

L1 norm: measures the absolute sum of the pixel-by-pixel differences. It is less sensitive than the L2 norm to large perturbations, but can be effective for detecting small uniformly distributed changes.

Lp norm (where p is a value between 1 and ∞): It is a generalization of the L1, L2 and L∞ norms.

For our project, we will use and compare "L2" and "L∞".

#### Modularity: 
The project is structured in a modular way to allow the replacement of attack models and algorithms without having to redo the entire flow.

To do this I used to divide the code into "functions", "classes" and I used "pattern designs".

#### Scalability: 
The system will be scalable to be able to add more RobustBench models or try different attack algorithms in the future.


## Conclusions

