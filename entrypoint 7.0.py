import os
import pickle
import numpy as np
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierPyTorch
from secml.array import CArray
from secml.ml.classifiers.loss import CSoftmax
from secml.ml.features.normalization import CNormalizerMinMax
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.peval.metrics import CMetricAccuracy
from secml.adv.attacks.evasion import CAttackEvasionFoolbox
from robustbench.utils import load_model as robustbench_load_model
import foolbox as fb
from secml.explanation import CExplainerIntegratedGradients
from misc import logo

# Caricamento dei modelli di RobustBench
model_names = [
	'Standard',
	"Wong2020Fast",
	"Engstrom2019Robustness",
	"Andriushchenko2020Understanding",
	"Cui2023Decoupled_WRN-28-10"
]

# Funzione per caricare i modelli RobustBench
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

# Caricamento del dataset CIFAR-10
tr, ts = CDataLoaderCIFAR10().load()
normalizer = CNormalizerMinMax().fit(tr.X)
ts.X = normalizer.transform(ts.X)


# Selezione di 1000 campioni bilanciati
np.random.seed(42)
indices = []
for class_idx in range(10):
    class_indices = np.where(ts.Y.tondarray() == class_idx)[0]
    selected_indices = np.random.choice(class_indices, 100, replace=False)
    indices.extend(selected_indices)
ts_balanced = ts[indices, :]

# Selezione di 64 campioni bilanciati
balanced_indices = []
for class_idx in range(10):
    class_indices = np.where(ts_balanced.Y.tondarray() == class_idx)[0]
    selected_indices = np.random.choice(class_indices, 6, replace=False)
    balanced_indices.extend(selected_indices)
ts_selected = ts_balanced[balanced_indices, :]



models = [load_model(name) for name in model_names]


# Calcolo delle predizioni e accuratezza dei modelli
metric = CMetricAccuracy()
models_preds = [clf.predict(ts.X) for clf in models]
accuracies = [metric.performance_score(y_true=ts.Y, y_pred=y_pred) for y_pred in models_preds]

# Stampa delle accuratezze
for idx in range(len(models)):
    print(f"Model idx: {idx+1} - Clean model accuracy: {(accuracies[idx] * 100):.2f} %")




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
                fb_attack_class=fb.attacks.L1FMNAttack,
                steps=500,
                max_stepsize=1.0,
                gamma=0.05
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
                steps=500,
                max_stepsize=1.0,
                gamma=0.05
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
file_path_linf = 'attack_data_linf.bin'
file_path_l2 = 'attack_data_l2.bin'



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
