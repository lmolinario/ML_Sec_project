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
from secml.explanation import CExplainerIntegratedGradients
import foolbox as fb

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
ts = ts[:64, :]# Select first 64 test samples

# Loading templates
models = [load_model(name) for name in model_names]

# Loading CIFAR-10 data and normalization
tr, ts = CDataLoaderCIFAR10().load()
normalizer = CNormalizerMinMax().fit(tr.X)
ts = ts[:64, :]  # Select the first 64 test samples
ts.X = normalizer.transform(ts.X)


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




model_id = 4
samples = [0, 1, 2, 3, 4, 5]  # Indices of samples to analyze

# Verifica la forma dell'array
print(f"Shape of labels.X: {ts.X.shape}")
print(f"Shape of labels.X: {ts.X.shape}")
print(f"Shape of attack_data_linf{model_id}{'adv_ds'}.X: {attack_data_linf[model_id]['adv_ds'].X.shape}")
# Ristrutturazione di labels.X e dei dati avversari in formato (3, 32, 32)
ts_X_reshaped = ts.X.reshape((64, 3, 32, 32))
attack_data_linf_reshaped = attack_data_linf[model_id]['adv_ds'].X.reshape((64, 3, 32, 32))

# Ora accedi ai campioni correttamente
for idx in samples:
    img = ts_X_reshaped[idx]
    img_adv = attack_data_linf_reshaped[idx]
    print(f"Campione {idx} selezionato.")

# Verifica gli indici dei campioni
for idx in samples:
    if idx < 0 or idx >= ts.X.shape[0]:
        print(f"Indice {idx} fuori dalla gamma, l'array ha {ts.X.shape[0]} campioni.")
    else:
        img = ts.X[idx]  # Accedi ai campioni correttamente
        print(f"Campione {idx} selezionato.")
import numpy as np

# Converti labels.X in un array numpy
ts_X_numpy = np.array(ts.X)

# Accedi al campione desiderato
img = ts_X_numpy[idx]

print(img.shape)



attributions = []
expl_name = 'Integrated Gradients'  # Name of the explainer
explainer = CExplainerIntegratedGradients(models[model_id])
for idx in samples:
    x_adv = attack_data_linf[model_id]['adv_ds'].X[idx, :]
    attr = CArray.empty(shape=(ts.classes.size, x_adv.size))
    for c in ts.classes:
        attr_c = explainer.explain(x_adv, y=c)
        attr[c, :] = attr_c
    attributions.append(attr)

model_id = 4
samples = [0, 1, 2, 3, 4, 5]  # Indices of samples to analyze
attributions = []
expl_name = 'Integrated Gradients'  # Name of the explainer
explainer = CExplainerIntegratedGradients(models[model_id])
for idx in samples:
    x_adv = attack_data_l2[model_id]['adv_ds'].X[idx, :]
    attr = CArray.empty(shape=(ts.classes.size, x_adv.size))
    for c in ts.classes:
        attr_c = explainer.explain(x_adv, y=c)
        attr[c, :] = attr_c
    attributions.append(attr)


"""    
# Run explainability
for ydx, idx in enumerate(samples):
    attributions.append(dict())  # Initialize an empty dictionary for this sample

    # Sample adversarial input
    x = attack_data[model_id]['adv_ds'].X[idx, :]

    print(f"Computing explanations using '{expl_name}' for sample {idx+1}...")


    # Initialize an empty array for attributions across all classes
    attr = CArray.empty(shape=(labels.classes.size, x.size))


    # Compute attributions for each class
    for c in labels.classes:
        # Compute explanation for class `c`
        attr_c = explainer.explain(x, y=c)
        attr[c, :] = attr_c

    # Store computed attributions for this sample
    attributions[ydx] = attr
"""

# Funzione di visualizzazione corretta
def visualize_results(fig, img, img_adv, label, pred, expl, idx):
    diff_img = (img_adv - img).tondarray()
    diff_img = (diff_img - diff_img.min()) / (diff_img.max() - diff_img.min())

    fig.subplot(5, 4, idx * 4 + 1)
    fig.sp.imshow(img.tondarray().transpose(1, 2, 0))
    fig.sp.title(f"Originale: {label}")
    fig.sp.xaxis.set_visible(False)
    fig.sp.yaxis.set_visible(False)

    fig.subplot(5, 4, idx * 4 + 2)
    fig.sp.imshow(img_adv.tondarray().transpose(1, 2, 0))
    fig.sp.title(f"Adversarial: {pred}")
    fig.sp.xaxis.set_visible(False)
    fig.sp.yaxis.set_visible(False)

    fig.subplot(5, 4, idx * 4 + 3)
    fig.sp.imshow(diff_img.transpose(1, 2, 0))
    fig.sp.title("Perturbazione")
    fig.sp.xaxis.set_visible(False)
    fig.sp.yaxis.set_visible(False)

    fig.subplot(5, 4, idx * 4 + 4)
    fig.sp.imshow(expl[0].tondarray(), cmap="seismic")
    fig.sp.title("Spiegazione")
    fig.sp.xaxis.set_visible(False)
    fig.sp.yaxis.set_visible(False)

# Creazione e salvataggio delle figure
fig = CFigure(height=20, width=6)
for idx, sample_id in enumerate(samples):
    img = ts.X[sample_id]
    img_adv = attack_data_linf[model_id]['adv_ds'].X[sample_id]
    expl = attributions[idx]
    label = ts.Y[sample_id].item()
    pred = attack_data_linf[model_id]['y_pred_adv'][sample_id].item()
    visualize_results(fig, img, img_adv, label, pred, expl, idx)

fig.tight_layout()
fig.savefig("explainability_attack_data_linf.jpg")
fig.show()

fig = CFigure(height=20, width=6)
for idx, sample_id in enumerate(samples):
    img = ts.X[sample_id]
    img_adv = attack_data_l2[model_id]['adv_ds'].X[sample_id]
    expl = attributions[idx]
    label = ts.Y[sample_id].item()
    pred = attack_data_l2[model_id]['y_pred_adv'][sample_id].item()
    visualize_results(fig, img, img_adv, label, pred, expl, idx)

fig.tight_layout()
fig.savefig("explainability_attack_data_l2.jpg")
fig.show()
