"""# Machine Learning Security Project - Project 1
Re-evaluate 5 RobustBench models with another attack algorithm (e.g. FMN)
and identify samples for which one attack works and the other doesn't. Explain the results
 - i.e., provide some motivations on why one of the attacks did not work properly, while the other did.


La scelta del miglior vincolo di norma (\(L_\infty\), \(L_2\), o \(L_1\)) dipende dal contesto e dagli obiettivi specifici del tuo progetto. Ogni vincolo offre vantaggi e svantaggi, e la loro efficacia può variare in base al modello, al dataset e al tipo di analisi. Ecco un confronto per aiutarti a decidere:

---

### **1. \(L_\infty\):**
- **Descrizione:**
  Limita la perturbazione massima per singolo pixel (valore massimo della perturbazione).
- **Vantaggi:**
  - È il più popolare e comunemente usato negli attacchi avversari (es. PGD, FGSM).
  - Facile da interpretare: controlla direttamente il massimo cambiamento che un pixel può subire.
  - Modelli robusti a \(L_\infty\) sono spesso robusti anche ad altri vincoli.
- **Svantaggi:**
  - Le perturbazioni possono risultare artificiali (non naturali) e distribuite uniformemente sull'immagine.
- **Migliore per:**
  Valutare la robustezza del modello in ambienti altamente controllati.

---

### **2. \(L_2\):**
- **Descrizione:**
  Limita la distanza euclidea tra l'immagine originale e quella avversaria.
- **Vantaggi:**
  - Genera perturbazioni più naturali e distribuite.
  - Può essere più rilevante in contesti pratici (es. immagini visivamente realistiche).
- **Svantaggi:**
  - Può essere meno efficace in termini di trasferibilità degli attacchi rispetto a \(L_\infty\).
- **Migliore per:**
  Scenari in cui si cerca di mantenere un aspetto naturale nelle immagini perturbate.

---

### **3. \(L_1\):**
- **Descrizione:**
  Minimizza la somma delle differenze assolute tra i pixel.
- **Vantaggi:**
  - Produce perturbazioni sparse, modificando solo pochi pixel.
  - Può essere utile per analizzare la sensibilità locale del modello.
- **Svantaggi:**
  - Meno studiato rispetto a \(L_\infty\) e \(L_2\).
  - Più difficile da ottimizzare.
- **Migliore per:**
  Scenari in cui la sparseness è un fattore critico (es. compressione o attacchi mirati).

---

### **Quale scegliere?**
- **Se il tuo obiettivo è valutare la robustezza generale:** \(L_\infty\) è spesso il primo vincolo da considerare, essendo lo standard nella ricerca sulla robustezza avversaria.
- **Se cerchi perturbazioni realistiche e distribuite:** \(L_2\) è più indicato, specialmente per analisi visive.
- **Se vuoi esplorare la vulnerabilità rispetto a perturbazioni locali:** \(L_1\) offre una prospettiva unica e complementare.

Puoi anche combinare più norme per un'analisi completa. Se hai bisogno di implementazioni specifiche o vuoi confrontare i risultati su un dataset, posso aiutarti con il codice o l'analisi. 😊




import os
try:
    import secml
except ImportError:
    os.system('pip install git+https://github.com/pralab/secml')

try:
    import foolbox as fb
except ImportError:
    os.system('pip install foolbox')

try:
    import robustbench
except ImportError:
    os.system('pip install git+https://github.com/RobustBench/robustbench.git')

try:
    import robustbench
except ImportError:
    import os
    os.system('pip install git+https://github.com/RobustBench/robustbench.git')

"""

import os
import pickle
import numpy as np
from secml.ml.classifiers import CClassifierPyTorch
from secml.array import CArray
from secml.ml.features.normalization import CNormalizerMinMax
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.peval.metrics import CMetricAccuracy
from secml.adv.attacks.evasion import CAttackEvasionFoolbox
from secml.explanation import CExplainerIntegratedGradients
from secml.figure import CFigure
from robustbench.utils import load_model as robustbench_load_model
import foolbox as fb

from secml.ml.classifiers import CClassifierPyTorch
from secml.array import CArray
from secml.ml.features.normalization import CNormalizerMinMax
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.peval.metrics import CMetricAccuracy

from robustbench.utils import load_model as robustbench_load_model

# Load Robustbench model
def load_model(model_name):
    # Use the correct function to load the model from Robustbench
    model = robustbench_load_model(model_name=model_name)  # Load the raw model from Robustbench

    # Wrap the model with CClassifierPyTorch
    clf = CClassifierPyTorch(
        model,
        input_shape=(3, 32, 32),  # CIFAR-10 image shape
        pretrained=True,
        pretrained_classes=CArray(list(range(10))),  # CIFAR-10 classes
        preprocess=None
    )

    return clf

# Load models
model_names = [
    "Wong2020Fast",
    "Engstrom2019Robustness",
    "Andriushchenko2020Understanding",
    "Cui2023Decoupled_WRN-28-10",
    "Chen2020Adversarial"
]

models = [load_model(name) for name in model_names]

# Load test data and normalize it
tr, ts = CDataLoaderCIFAR10().load()
normalizer = CNormalizerMinMax().fit(tr.X)

#ts.X = normalizer.transform(ts.X[:64, :])  # Normalizza i primi 64 campioni
#ts.Y = ts.Y[:64]  # Aggiorna le etichette corrispondenti
ts.X = normalizer.transform(ts.X)# Select first 64 test samples
#ts = ts[:64, :]  # Select first 64 test samples
ts = ts[:64, :]


# Accuracy metric
metric = CMetricAccuracy()

# Model predictions
models_preds = [clf.predict(ts.X) for clf in models]

# Calculate accuracies
accuracies = [metric.performance_score(y_true=ts.Y, y_pred=y_pred) for y_pred in models_preds]

# Print results
for idx in range(5):
    print(f"Model idx: {idx+1} - Clean model accuracy: {(accuracies[idx] * 100):.2f} %")



from secml.adv.attacks.evasion import CAttackEvasionFoolbox
import foolbox as fb

def attack_models(samples, labels):
    init_params = dict(steps=500, max_stepsize=1.0, gamma=0.05)
    attack_data = dict()

    # Run the attack on each model
    for idx, model in enumerate(models):
        print(f"Starting attack on model {idx+1}...")

        try:
            # Untargeted attack
            attack = CAttackEvasionFoolbox(
                model,
                #y_target=None,
                epsilons=8 / 255,
                fb_attack_class=fb.attacks.LInfFMNAttack,  # Ensure this attack is supported in your Foolbox version
                **init_params
            )

            # Run the attack and get predictions
            y_pred, _, adv_ds, _ = attack.run(samples, labels)

            # Store attack results
            attack_data[idx] = {
                'x_seq': attack.x_seq,
                'y_pred_adv': y_pred,
                'adv_ds': adv_ds
            }

            print(f"Attack complete on model {idx+1}!")
        except Exception as e:
            print(f"Error during attack on model {idx+1}: {e}")

    return attack_data

# Percorso del file per salvare/caricare i dati degli attacchi
file_path = 'attack_data.bin'

# Controllo e gestione del file
if os.path.exists(file_path):
    print(f"Il file '{file_path}' esiste. Caricamento dei dati...")
    with open(file_path, 'rb') as file:
        attack_data = pickle.load(file)
    print("Dati caricati con successo.")
else:
    print(f"Il file '{file_path}' non esiste. Esecuzione dell'attacco...")
    attack_data = attack_models(ts.X, ts.Y)
    with open(file_path, 'wb') as file:
        pickle.dump(attack_data, file)
    print(f"Dati salvati con successo in '{file_path}'.")





for idx, model in enumerate(models):
    accuracy_adv = metric.performance_score(
        y_true=ts.Y,
        y_pred=attack_data[idx]['y_pred_adv']
    )

    print(f"Model idx: {idx+1} - Model accuracy under attack: {(accuracy_adv * 100):.2f} %")



model_id = 4
samples = [0, 1, 2, 3, 4, 5]  # Indices of samples to analyze
attributions = []
expl_name = 'Integrated Gradients'  # Name of the explainer
explainer = CExplainerIntegratedGradients(models[model_id])
for idx in samples:
    x_adv = attack_data[model_id]['adv_ds'].X[idx, :]
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
    attr = CArray.empty(shape=(ts.classes.size, x.size))


    # Compute attributions for each class
    for c in ts.classes:
        # Compute explanation for class `c`
        attr_c = explainer.explain(x, y=c)
        attr[c, :] = attr_c

    # Store computed attributions for this sample
    attributions[ydx] = attr
"""

# Funzione di visualizzazione
def visualize_results(fig, img, img_adv, label, pred, expl, idx):
    diff_img = (img_adv - img).tondarray()
    diff_img = (diff_img - diff_img.min()) / (diff_img.max() - diff_img.min())

    fig.subplot(5, 4, idx * 4 + 1)
    fig.sp.imshow(img.tondarray().transpose(1, 2, 0))
    fig.sp.title(f"Originale: {label}")
    fig.sp.axis("off")

    fig.subplot(5, 4, idx * 4 + 2)
    fig.sp.imshow(img_adv.tondarray().transpose(1, 2, 0))
    fig.sp.title(f"Adversarial: {pred}")
    fig.sp.axis("off")

    fig.subplot(5, 4, idx * 4 + 3)
    fig.sp.imshow(diff_img.transpose(1, 2, 0))
    fig.sp.title("Perturbazione")
    fig.sp.axis("off")

    fig.subplot(5, 4, idx * 4 + 4)
    fig.sp.imshow(expl[0].tondarray(), cmap="seismic")
    fig.sp.title("Spiegazione")
    fig.sp.axis("off")

fig = CFigure(height=20, width=6)
for idx, sample_id in enumerate(samples):
    img = ts.X[sample_id]
    img_adv = attack_data[model_id]['adv_ds'].X[sample_id]
    expl = attributions[idx]
    label = ts.Y[sample_id].item()
    pred = attack_data[model_id]['y_pred_adv'][sample_id].item()
    visualize_results(fig, img, img_adv, label, pred, expl, idx)
fig.tight_layout()
fig.savefig("explainability.jpg")
fig.show()
