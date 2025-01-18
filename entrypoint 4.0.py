import os
import pickle
import numpy as np
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierPyTorch
from secml.array import CArray
from secml.ml.features.normalization import CNormalizerMinMax
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.peval.metrics import CMetricAccuracy
from secml.adv.attacks.evasion import CAttackEvasionFoolbox
from robustbench.utils import load_model as robustbench_load_model
import foolbox as fb
from secml.explanation import CExplainerIntegratedGradients
import matplotlib.pyplot as plt

# Funzione per caricare un modello da RobustBench
def load_model(model_name):
    model = robustbench_load_model(model_name=model_name)
    clf = CClassifierPyTorch(
        model,
        input_shape=(3, 32, 32),
        pretrained=True,
        pretrained_classes=CArray(list(range(10))),
        preprocess=None
    )
    return clf

# Caricamento dei modelli RobustBench
model_names = [
    "Wong2020Fast",
    "Engstrom2019Robustness",
    "Andriushchenko2020Understanding",
    "Cui2023Decoupled_WRN-28-10",
    "Chen2020Adversarial"
]
models = [load_model(name) for name in model_names]

# Caricamento dei dati CIFAR-10 e normalizzazione
tr, ts = CDataLoaderCIFAR10().load()
normalizer = CNormalizerMinMax().fit(tr.X)
ts.X = normalizer.transform(ts.X)  # Normalizza i dati
ts = ts[:64, :]  # Seleziona i primi 64 campioni di test

# Funzione per attaccare i modelli con vincolo L∞
def attack_models_linf(samples, labels):
    init_params = dict(steps=500, max_stepsize=1.0, gamma=0.05)
    attack_data_linf = dict()
    for idx, model in enumerate(models):
        print(f"Starting L∞ attack on model {idx+1}...")
        try:
            attack = CAttackEvasionFoolbox(
                model,
                epsilons=8 / 255,
                fb_attack_class=fb.attacks.LInfFMNAttack,
                **init_params
            )
            y_pred, _, adv_ds, _ = attack.run(samples, labels)
            attack_data_linf[idx] = {
                'x_seq': attack.x_seq,
                'y_pred_adv': y_pred,
                'adv_ds': adv_ds
            }
            print(f"L∞ Attack complete on model {idx+1}!")
        except Exception as e:
            print(f"Error during L∞ attack on model {idx+1}: {e}")
    return attack_data_linf

# Funzione per attaccare i modelli con vincolo L2
def attack_models_l2(samples, labels):
    init_params = dict(steps=500, max_stepsize=1.0, gamma=0.05)
    attack_data_l2 = dict()
    for idx, model in enumerate(models):
        print(f"Starting L2 attack on model {idx+1}...")
        try:
            attack = CAttackEvasionFoolbox(
                model,
                epsilons=0.5,  # Epsilon per L2
                fb_attack_class=fb.attacks.L2FMNAttack,
                **init_params
            )
            y_pred, _, adv_ds, _ = attack.run(samples, labels)
            attack_data_l2[idx] = {
                'x_seq': attack.x_seq,
                'y_pred_adv': y_pred,
                'adv_ds': adv_ds
            }
            print(f"L2 Attack complete on model {idx+1}!")
        except Exception as e:
            print(f"Error during L2 attack on model {idx+1}: {e}")
    return attack_data_l2

# Percorsi dei file per salvare/caricare i dati degli attacchi
file_path_linf = 'Vecchi file /attack_data_linf.bin'
file_path_l2 = 'Vecchi file /attack_data_l2.bin'

# Gestione degli attacchi L∞
if os.path.exists(file_path_linf):
    print(f"Il file '{file_path_linf}' esiste. Caricamento dei dati...")
    with open(file_path_linf, 'rb') as file:
        attack_data_linf = pickle.load(file)
    print("Dati L∞ caricati con successo.\n")
else:
    print(f"Il file '{file_path_linf}' non esiste. Esecuzione dell'attacco L∞...")
    attack_data_linf = attack_models_linf(ts.X, ts.Y)
    with open(file_path_linf, 'wb') as file:
        pickle.dump(attack_data_linf, file)
    print(f"Dati L∞ salvati con successo in '{file_path_linf}'.")

# Gestione degli attacchi L2
if os.path.exists(file_path_l2):
    print(f"Il file '{file_path_l2}' esiste. Caricamento dei dati...")
    with open(file_path_l2, 'rb') as file:
        attack_data_l2 = pickle.load(file)
    print("Dati L2 caricati con successo.\n")
else:
    print(f"Il file '{file_path_l2}' non esiste. Esecuzione dell'attacco L2...")
    attack_data_l2 = attack_models_l2(ts.X, ts.Y)
    with open(file_path_l2, 'wb') as file:
        pickle.dump(attack_data_l2, file)
    print(f"Dati L2 salvati con successo in '{file_path_l2}'.")

import numpy as np


# Funzione per mostrare un campione di immagini
import numpy as np

"""
# Funzione per mostrare un campione di immagini
def plot_images(originals, adversarials_linf, adversarials_l2, labels, model_idx, num_images=5):
	fig, axes = plt.subplots(num_images, 3, figsize=(10, 2 * num_images))

	for i in range(num_images):
		# Controlla la forma delle immagini originali
		print(f"Original shape: {originals[i].shape}")

		# Se l'immagine è un array 1D di 3072 elementi, la rimodelliamo in (3, 32, 32)
		if len(originals[i].shape) == 1 and originals[i].shape[0] == 3072:
			originals_image = originals[i].reshape(3, 32, 32)  # Rimodella come immagine RGB 32x32
			originals_image = originals_image.transpose(1, 2, 0)  # Converti da (C, H, W) a (H, W, C)
		else:
			raise ValueError(f"Immagine originale con forma inattesa: {originals[i].shape}")

		# Visualizza immagine originale
		axes[i, 0].imshow(originals_image)
		axes[i, 0].set_title(f"Original {labels[i]}")
		axes[i, 0].axis('off')

		# Visualizza immagine avversaria L∞
		print(f"Adversarial L∞ shape: {adversarials_linf[i].shape}")
		if len(adversarials_linf[i].shape) == 1 and adversarials_linf[i].shape[0] == 3072:
			adv_linf_image = adversarials_linf[i].reshape(3, 32, 32).transpose(1, 2, 0)
		else:
			adv_linf_image = adversarials_linf[i].transpose(1, 2, 0)

		axes[i, 1].imshow(adv_linf_image)
		axes[i, 1].set_title(f"Adv L∞ {labels[i]}")
		axes[i, 1].axis('off')

		# Visualizza immagine avversaria L2
		print(f"Adversarial L2 shape: {adversarials_l2[i].shape}")
		if len(adversarials_l2[i].shape) == 1 and adversarials_l2[i].shape[0] == 3072:
			adv_l2_image = adversarials_l2[i].reshape(3, 32, 32).transpose(1, 2, 0)
		else:
			adv_l2_image = adversarials_l2[i].transpose(1, 2, 0)

		axes[i, 2].imshow(adv_l2_image)
		axes[i, 2].set_title(f"Adv L2 {labels[i]}")
		axes[i, 2].axis('off')

	# Salva l'immagine come file PNG
	plt.savefig('attacks_images.png')  # Salva l'immagine come file PNG


	plt.tight_layout()
	plt.show()
	
	
	# Funzione per estrarre e visualizzare le immagini
def display_attack_images(attack_data_linf, attack_data_l2, num_images=5):
    # Seleziona alcune immagini di attacco (per esempio, le prime 5)
    for idx, model_data in attack_data_linf.items():
        if idx >= len(_models):
            continue
        adv_images_linf = model_data['adv_ds'].X.get_data()  # Accedi ai dati grezzi come array NumPy
        adv_images_l2 = attack_data_l2[idx]['adv_ds'].X.get_data()

        # Seleziona un campione di immagini, considerando che `X` potrebbe essere un array 4D
        plot_images(ts.X.get_data()[:num_images], adv_images_linf[:num_images], adv_images_l2[:num_images],
                    ts.Y.get_data()[:num_images], model_idx=idx, num_images=num_images)
                    
                    
"""

def plot_images(originals, adversarials_linf, adversarials_l2, labels, num_images=5):
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 2 * num_images))

    for i in range(num_images):
        # Visualizza l'immagine originale
        originals_image = originals[i].reshape(3, 32, 32).transpose(1, 2, 0)
        axes[i, 0].imshow(originals_image)
        axes[i, 0].set_title(f"Original {labels[i]}")
        axes[i, 0].axis('off')

        # Visualizza immagine avversaria L∞
        adv_linf_image = adversarials_linf[i].reshape(3, 32, 32).transpose(1, 2, 0)
        axes[i, 1].imshow(adv_linf_image)
        axes[i, 1].set_title(f"Adv L∞ {labels[i]}")
        axes[i, 1].axis('off')

        # Visualizza immagine avversaria L2
        adv_l2_image = adversarials_l2[i].reshape(3, 32, 32).transpose(1, 2, 0)
        axes[i, 2].imshow(adv_l2_image)
        axes[i, 2].set_title(f"Adv L2 {labels[i]}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'attack_images_sample.png')  # Salva su file

    plt.tight_layout()
    plt.show()

# Funzione per estrarre e visualizzare le immagini
def display_attack_images(attack_data_linf, attack_data_l2, num_images=5):
    # Seleziona alcune immagini di attacco (per esempio, le prime 5)
    for idx, model_data in attack_data_linf.items():
        if idx >= len(models):
            continue
        adv_images_linf = model_data['adv_ds'].X.get_data()  # Accedi ai dati grezzi come array NumPy
        adv_images_l2 = attack_data_l2[idx]['adv_ds'].X.get_data()

        # Seleziona un campione di immagini, considerando che `X` potrebbe essere un array 4D
        plot_images(ts.X.get_data()[:num_images], adv_images_linf[:num_images], adv_images_l2[:num_images],
                    ts.Y.get_data()[:num_images])  # Rimuovi 'model_idx'



# Visualizza un campione di immagini per gli attacchi L∞ e L2
display_attack_images(attack_data_linf, attack_data_l2, num_images=5)