"""# Machine Learning Security Project - Project 1
Re-evaluate 5 RobustBench _models with another attack algorithm (e.g. FMN)
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



"""
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

# Load _models
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
ts = ts[:64, :]  # Select first 64 test samples
ts.X = normalizer.transform(ts.X)

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
        print(f"Starting attack on model {idx+1}")

        try:
            # Untargeted attack
            attack = CAttackEvasionFoolbox(
                model,
                y_target=None,
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

            print(f"Attack complete on model {idx+1}")
        except Exception as e:
            print(f"Error during attack on model {idx+1}: {e}")

    return attack_data

# Run the attack on the test samples
attack_data = attack_models(ts.X, ts.Y)

# Optionally, print or analyze the results from `attack_data`


import pickle
from google.colab import files

# Check if attack_data is defined
if 'attack_data' in globals():
    # Save the attack_data dictionary to a binary file
    with open('attack_data.bin', 'wb') as file:
        pickle.dump(attack_data, file)

    print("File saved successfully as 'attack_data.bin'.")

    # Download the file to your local machine
    files.download('attack_data.bin')
else:
    print("Error: 'attack_data' is not defined.")



for idx, model in enumerate(models):
    accuracy = metric.performance_score(
        y_true=ts.Y,
        y_pred=attack_data[idx]['y_pred_adv']
    )

    print(f"Model idx: {idx+1} - Model accuracy under attack: {(accuracy * 100):.2f} %")



model_id = 4
samples = [0, 1, 2, 3, 4, 5]  # Indices of samples to analyze
attributions = []
expl_name = 'Integrated Gradients'  # Name of the explainer
explainer = CExplainerIntegratedGradients(models[model_id])

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


def convert_image(image):
    return image.tondarray().reshape(input_shape).transpose(1, 2, 0)

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

    # Explaination
    # Calculate the maximum error for each pixel
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

adv_ds = attack_data[model_id]['adv_ds']
y_adv  = attack_data[model_id]['y_pred_adv']

fig = CFigure(height=20, width=6, fontsize=10)

for ydx, idx in enumerate(samples):
    img     = ts.X[idx, :]
    img_adv = adv_ds.X[idx, :]
    expl    = attributions[ydx][y_adv[idx].item(), :]

    show_image(
        fig,
        ydx,
        img,
        img_adv,
        expl,
        dataset_labels[ts.Y[idx].item()],
        dataset_labels[y_adv[idx].item()]
    )

fig.tight_layout(rect=[0, 0.003, 1, 0.94])
fig.savefig("explainability.jpg")
fig.show()




sample_id = 2
sample    = ts.X[sample_id, :]
label     = ts.Y[sample_id]

attack_data = attack_models(sample, label)


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
fig.show()

