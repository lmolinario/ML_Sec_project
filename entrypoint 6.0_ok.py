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

n_samples = 64
dataset_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

models = [load_model(name) for name in model_names]

# Caricamento dei dati CIFAR-10 e normalizzazione
tr, ts = CDataLoaderCIFAR10().load()
normalizer = CNormalizerMinMax().fit(tr.X)
ts.X = normalizer.transform(ts.X)  # Normalizza i dati
ts = ts[:64, :]  # Seleziona i primi 64 campioni di test

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


model_id = 4
samples = [0, 1, 2, 3, 4, 5]  # Ipotesi
import os
import pickle

# Percorso del file per salvare/caricare le attribuzioni
attributions_file_path = 'attributions_data.pkl'


# Funzione per salvare le attribuzioni
def save_attributions(attributions, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(attributions, file)


# Funzione per caricare le attribuzioni
def load_attributions(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


# Controlla se il file delle attribuzioni esiste
if os.path.exists(attributions_file_path):
    print(f"Il file '{attributions_file_path}' esiste. Caricamento dei dati...")
    attributions = load_attributions(attributions_file_path)
    print("Dati delle attribuzioni caricati con successo.\n")
else:
    print(f"Il file '{attributions_file_path}' non esiste. Calcolo delle attribuzioni...")
    attributions = []

    # Calcola le attribuzioni per ogni campione
    explainer = CExplainerIntegratedGradients(models[model_id])

    for ydx, idx in enumerate(samples):
        attributions.append(dict())

        # Sample
        x = attack_data_linf[model_id]['adv_ds'].X[idx, :]

        # Compute explanations (attributions) wrt each class
        print(f"Computing explanations for sample {idx + 1}...")

        # Empty array where to collect attributions values
        attr = CArray.empty(shape=(ts.num_classes, x.size))

        # Loop over classes
        for c in ts.classes:
            # Compute the explanation
            attr_c = explainer.explain(x, y=c)
            attr[c, :] = attr_c

        attributions[ydx] = attr

    # Salva le attribuzioni per analisi future
    save_attributions(attributions, attributions_file_path)
    print(f"Dati delle attribuzioni salvati con successo in '{attributions_file_path}'.\n")





def convert_image(image):
    return image.tondarray().reshape((3, 32, 32)).transpose(1, 2, 0)

def show_image(fig, idx, img, img_adv, expl, label, pred):
    diff_img = img_adv - img
    diff_img -= diff_img.min()
    diff_img /= diff_img.max()

    # True sample
    fig.subplot(10, 4, idx*4+1)
    fig.sp.imshow(convert_image(img))
    fig.sp.title(f"True: {label}")
    fig.sp.xticks([])
    fig.sp.yticks([])

    # Adv sample
    fig.subplot(10, 4, idx*4+2)
    fig.sp.imshow(convert_image(img_adv))
    fig.sp.title(f'Adv: {pred}')
    fig.sp.xticks([])
    fig.sp.yticks([])

    # Perturbation
    fig.subplot(10, 4, idx*4+3)
    fig.sp.imshow(convert_image(diff_img))
    fig.sp.title('Perturbation')
    fig.sp.xticks([])
    fig.sp.yticks([])

    # Explanation
    expl = convert_image(expl)
    r = np.fabs(expl[:, :, 0])
    g = np.fabs(expl[:, :, 1])
    b = np.fabs(expl[:, :, 2])
    expl = np.maximum(np.maximum(r, g), b)

    fig.subplot(10, 4, idx*4+4)
    fig.sp.imshow(expl, cmap='seismic')
    fig.sp.title('Explain')
    fig.sp.xticks([])
    fig.sp.yticks([])

# Main

adv_ds_Linf = attack_data_linf[model_id]['adv_ds']
y_adv_Linf  = attack_data_linf[model_id]['y_pred_adv']

fig = CFigure(height=20, width=6, fontsize=10)

for ydx, idx in enumerate(samples):
    img = ts.X[idx, :]
    img_adv = adv_ds_Linf.X[idx, :]
    expl = attributions[ydx][y_adv_Linf[idx].item(), :]

    show_image(
        fig,
        ydx,
        img,
        img_adv,
        expl,
        dataset_labels[ts.Y[idx].item()],
        dataset_labels[y_adv_Linf[idx].item()]
    )

fig.tight_layout(rect=[0, 0.003, 1, 0.94])
fig.savefig("explainabilityLInf.jpg")
fig.show()
print(f"explainabilityLInf.jpg salvato con successo\n")



# To check if the attack has properly converged to a good local minimum
sample_id = 2
sample = ts.X[sample_id, :]
label = ts.Y[sample_id]

attack_data = attack_models_linf(sample, label)

fig = CFigure(width=30, height=4, fontsize=10, linewidth=2)

for model_id in range(5):
    n_iter = attack_data[model_id]['x_seq'].shape[0]
    itrs = CArray.arange(n_iter)

    # Classify all the points in the attack path
    scores = models[model_id].predict(
        attack_data[model_id]['x_seq'],
        return_decision_function=True
    )[1]

    # Apply the softmax to the score to have value in [0,1]
    scores = CSoftmax().softmax(scores)

    fig.subplot(1, 5, model_id+1)

    if model_id == 0:
        fig.sp.ylabel('confidence')

    fig.sp.xlabel('iteration')

    fig.sp.plot(itrs, scores[:, label], linestyle='--', c='black')
    fig.sp.plot(itrs, scores[:, attack_data[0]['y_pred_adv'][0]], c='black')

    fig.sp.xlim(top=25, bottom=0)

    fig.sp.title(f"Confidence Sample {sample_id+1} - Model: {model_id+1}")
    fig.sp.legend(['Confidence True Class', 'Confidence Adv. Class'])

fig.tight_layout()
fig.savefig("ConfidenceInf.jpg")
fig.show()
print(f"ConfidenceInf.jpg salvato con successo\n")



adv_ds_l2 = attack_data_l2[model_id]['adv_ds']
y_adv_l2 = attack_data_l2[model_id]['y_pred_adv']

fig = CFigure(height=20, width=6, fontsize=10)

for ydx, idx in enumerate(samples):
    img = ts.X[idx, :]
    img_adv = adv_ds_l2.X[idx, :]
    expl = attributions[ydx][y_adv_l2[idx].item(), :]

    show_image(
        fig,
        ydx,
        img,
        img_adv,
        expl,
        dataset_labels[ts.Y[idx].item()],
        dataset_labels[y_adv_l2[idx].item()]
    )

fig.tight_layout(rect=[0, 0.003, 1, 0.94])
fig.savefig("explainabilityL2.jpg")
fig.show()
print(f"explainabilityL2.jpg salvato con successo\n")



# To check if the attack has properly converged to a good local minimum
sample_id = 2
sample = ts.X[sample_id, :]
label = ts.Y[sample_id]

attack_data = attack_data_l2(sample, label)

fig = CFigure(width=30, height=4, fontsize=10, linewidth=2)

for model_id in range(5):
    n_iter = attack_data[model_id]['x_seq'].shape[0]
    itrs = CArray.arange(n_iter)

    # Classify all the points in the attack path
    scores = models[model_id].predict(
        attack_data[model_id]['x_seq'],
        return_decision_function=True
    )[1]

    # Apply the softmax to the score to have value in [0,1]
    scores = CSoftmax().softmax(scores)

    fig.subplot(1, 5, model_id+1)

    if model_id == 0:
        fig.sp.ylabel('confidence')

    fig.sp.xlabel('iteration')

    fig.sp.plot(itrs, scores[:, label], linestyle='--', c='black')
    fig.sp.plot(itrs, scores[:, attack_data[0]['y_pred_adv'][0]], c='black')

    fig.sp.xlim(top=25, bottom=0)

    fig.sp.title(f"Confidence Sample {sample_id+1} - Model: {model_id+1}")
    fig.sp.legend(['Confidence True Class', 'Confidence Adv. Class'])

fig.tight_layout()
fig.savefig("ConfidenceL2.jpg")
fig.show()
print(f"ConfidenceL2.jpg salvato con successo\n")