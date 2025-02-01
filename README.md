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

- Download the ZIP code or clone the repository with
  ```bash
  git clone https://github.com/lmolinario/ML_Sec_project.git
  ```
- Enter inside a repository dire`ctory and `install the requirements with

  ```bash
  pip3 install -r requirements.txt
  ```
- Run the file `entrypoint.py` to start the program 
 
  ```bash
  python3.10 entrypoint.py
  ```

## Project goal
The goal of this project is to re-evaluate 5 RobustBench models using another attack algorithm (e.g., FMN) and identify samples for which one attack succeeds while the other fails. In other words, we aim to compare the effectiveness of different attacks against robust models, to analyze in which cases one type of attack is effective while another fails, thus contributing to a deeper understanding of the robustness of models and attack algorithms.


## Solution Design
To re-evaluate the FNM model, we use as a basis for comparison the results of "AutoAttack - Robustbench"
calculated on the same epsilon (in this case epsilon = 8/255 with "L-inf" norm) and we take into account the samples that successfully perturb the image with epsilon < 8/255.

#### Attack algorithm
As indicated in our project I took as a reference the FMN attack, also known as FGSM (Fast Gradient Sign Method), which is one of the most common attacks against neural networks.
The basic idea of this attack is to calculate the gradient of the model with respect to the input image and add a perturbation in the direction of the gradient to maximize the loss. This type of attack can be implemented as follows:

![δ=ϵ⋅sign(∇xJ(θ,x,y))](misc/Formula FMN.png)

Where:

**δ** is the generated perturbation.

**ϵ** is the magnitude of the perturbation (i.e. the strength of the attack).

**∇xJ(θ,x,y)** is the gradient of the loss function J with respect to the input image x, calculated for the model parameters θ.

**sign(⋅)** refers to the function that takes the sign of each gradient value.

The **norm** is a fundamental concept in all adversarial attacks.
It defines how to measure and bound the size of the perturbation added to the original image. It is a mathematical measure that establishes the "strength" of the perturbation and is used to control how much the original image is modified.

In our case of **FMN attack**, the norm directly influences the creation of the adversarial perturbation and its control.
Depending on the type of norm chosen, the perturbation can have different characteristics:

**L∞ norm (maximum norm)**: measures the maximum difference for each pixel between the original and perturbed images. The L∞ norm is useful for testing robustness against attacks that limit the perturbation to the maximum value for each pixel.

**L2 norm (Euclidean)**: measures the Euclidean distance between the original and perturbed images. The L2 norm is generally sensitive to large changes in the images, but may not capture very small local perturbations that affect recognition.

**L1 norm**: measures the absolute sum of the pixel-by-pixel differences. It is less sensitive than the L2 norm to large perturbations, but can be effective for detecting small uniformly distributed changes.

**Lp norm** (where p is a value between 1 and ∞): It is a generalization of the L1, L2 and L∞ norms.

**For our project, we will use and compare "L∞".**

The distance "L∞" is a fundamental measure in adversarial attack problems, as it represents the maximum change that is applied to a single pixel in the image during the generation of adversarial samples.
Limiting the distance "L∞" to a specific epsilon value ( in our case 8/255 ) has several motivations and importance:
Adversarial attacks must be visually imperceptible to a human observer. If the perturbation exceeds a certain limit, the adversarial sample may appear distorted or artificial.
A smaller perturbation (epsilon < 8/255 ) ensures that the changes in pixels are minimal, keeping the image visually similar to the original.

Additionally, the CIFAR-10 dataset uses normalized images with pixel values ​​between 0 and 1.
A value of epsilon = 8/255 represents a very small change (about 3% of the full scale), which is consistent with the idea of ​​a “sneaky” perturbation that exploits the model’s vulnerability without excessively changing the image.
The choice of epsilon = 8/255 is not arbitrary: it is a standardized value in many adversarial attack studies, especially for models tested on CIFAR-10 with the “L∞” norm.
It allows direct comparison of adversarial and defense results, since many benchmarks use the same bound.

Generating adversarial samples with smaller “L∞” constraints requires less exploration of the perturbation space, making the attacks more efficient to compute.
Larger perturbations may trigger model- or dataset-specific artifacts, compromising the generalizability of the results.

#### Modularity: 
The project is structured in a modular way to allow the replacement of attack models and algorithms without having to redo the entire flow.

To do this I used to divide the code into "functions", "classes" and I used "pattern designs".

#### Scalability: 
The system will be scalable to be able to add more RobustBench models or try different attack algorithms in the future.

## Conclusions


Motivazioni per cui un Attacco può Fallire

Dopo aver individuato i campioni con risultati contrastanti, possiamo fare alcune ipotesi sulle motivazioni per cui un attacco può fallire:

	FMN può generare perturbazioni più piccole
		FMN è progettato per trovare la minima perturbazione che induce un errore. Se la perturbazione necessaria supera il budget (ε = 8/255), l'attacco potrebbe fallire.

	AutoAttack utilizza una strategia più aggressiva
		AutoAttack combina più metodi (PGD, APGD, Square Attack) ed è più probabile che trovi un punto debole nel modello.

	La robustezza del modello può influenzare gli attacchi in modo diverso
		Alcuni modelli potrebbero essere più vulnerabili a perturbazioni sparse (come Square Attack di AutoAttack) rispetto a perturbazioni minimali (come FMN).

	Diverse classi possono avere sensibilità diverse agli attacchi
		Analizzando i risultati per classe, possiamo scoprire se certe classi sono più facili da attaccare con un metodo rispetto all'altro.




Interpretazione dei risultati

FMN e AutoAttack funzionano in modo diverso
FMN cerca la minima perturbazione necessaria per modificare la classificazione. Se il modello è particolarmente robusto, potrebbe non essere in grado di trovare una perturbazione sufficiente.
AutoAttack è più aggressivo e combina più tecniche, quindi potrebbe avere successo su alcuni campioni che FMN non riesce a ingannare.

Ci sono immagini che sono più difficili da attaccare con FMN che con AutoAttack (o viceversa)
Se AutoAttack ha successo e FMN fallisce, significa che la strategia di minima perturbazione di FMN non è sufficiente a forzare il fallimento del modello.
Se FMN ha successo e AutoAttack fallisce, potrebbe significare che AutoAttack non riesce a trovare una buona strategia di perturbazione per quel campione specifico.

Possibili spiegazioni per le differenze
Dipendenza dalla classe: alcune classi di immagini potrebbero essere più difficili da attaccare con FMN che con AutoAttack. Effetto interruzione: AutoAttack può creare interruzioni più "dirompenti", mentre FMN lavora su piccole modifiche che potrebbero non essere sempre efficaci.
Robustezza del modello: il modello potrebbe resistere a un attacco meglio dell'altro a seconda delle sue caratteristiche di robustezza.