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


# Calcolo delle accuratezze sui dati avversari
metric = CMetricAccuracy()
for idx, model in enumerate(models):
    accuracy_adv_linf = metric.performance_score(
        y_true=ts.Y,
        y_pred=attack_data_linf[idx]['y_pred_adv']
    )
    accuracy_adv_l2 = metric.performance_score(
        y_true=ts.Y,
        y_pred=attack_data_l2[idx]['y_pred_adv']
    )
    print(f"\nModel idx: {idx+1} - Accuracy under L∞ attack: {(accuracy_adv_linf * 100):.2f} %")
    print(f"Model idx: {idx+1} - Accuracy under L2 attack: {(accuracy_adv_l2 * 100):.2f} %")

# Indici dei campioni da analizzare
model_id = 4
samples = [0, 1, 2, 3, 4, 5]  # Indici dei campioni da analizzare

# Assumendo che `ts.X` contenga i dati del modello in formato (64, 3072)
ts_X_reshaped = ts.X.reshape((64, 3, 32, 32))  # Ristruttura i dati per immagini 32x32x3

# Dati avversari L∞ (assumendo che siano già caricati correttamente)
attack_data_linf_reshaped = attack_data_linf[model_id]['adv_ds'].X.reshape((64, 3, 32, 32))

# Dati avversari L2 (assumendo che siano già caricati correttamente)
attack_data_l2_reshaped = attack_data_l2[model_id]['adv_ds'].X.reshape((64, 3, 32, 32))

# Verifica la forma dell'array
print(f"Shape of ts.X: {ts.X.shape}")
print(f"Shape of attack_data_linf[model_id]['adv_ds'].X: {attack_data_linf[model_id]['adv_ds'].X.shape}")
import matplotlib.pyplot as plt
import numpy as np

# Funzione per visualizzare le immagini
def plot_image(ax, image, title, cmap='viridis'):
    ax.imshow(np.transpose(image, (1, 2, 0)))  # Riorganizza l'array per visualizzarlo come immagine
    ax.set_title(title)
    ax.axis('off')

# Funzione per visualizzare l'immagine originale, avversaria, la perturbazione e la spiegazione
def plot_images(original, adversarial, perturbation, explanation, sample_idx):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Visualizza l'immagine originale
    plot_image(axes[0], original[sample_idx], "Original Image", cmap='viridis')

    # Visualizza l'immagine avversaria
    plot_image(axes[1], adversarial[sample_idx], "Adversarial Image", cmap='viridis')

    # Visualizza la perturbazione
    perturbation_img = adversarial[sample_idx] - original[sample_idx]  # Perturbazione come differenza
    plot_image(axes[2], perturbation_img, "Perturbation", cmap='seismic')

    # Visualizza la spiegazione (Integrated Gradients)
    plot_image(axes[3], explanation[sample_idx], "Explanation", cmap='viridis')

    plt.tight_layout()
    plt.show()

# Funzione per calcolare la spiegazione tramite Integrated Gradients
def compute_explanation(model, sample):
    explainer = CExplainerIntegratedGradients(model)
    explanation = explainer.explain(sample)  # Ottieni la spiegazione per il campione
    return explanation

# Ciclo per visualizzare le immagini per un campione specifico (esempio: campione 0)
sample_idx = 0  # Indice del campione da analizzare

# Assicurati di avere l'indicizzazione corretta per accedere ai dati
original_image = ts_X_reshaped
adversarial_linf_image = attack_data_linf[model_id]['adv_ds'].X.reshape((64, 3, 32, 32))
adversarial_l2_image = attack_data_l2[model_id]['adv_ds'].X.reshape((64, 3, 32, 32))

# Estrai il campione e il suo target
single_image = original_image[sample_idx, :]  # Campione
target = ts.Y[sample_idx]  # Target associato al campione

# Ora chiama la funzione di spiegazione passando sia l'immagine che il target
explanation = compute_explanation(models[model_id], CArray(single_image), target)

# Visualizza le immagini
plot_images(original_image, adversarial_linf_image, adversarial_linf_image - original_image, explanation, sample_idx)
