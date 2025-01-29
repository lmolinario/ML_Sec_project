import os
import pickle
import numpy as np

import robustbench
import secml
from secml.adv.attacks.evasion import CAttackEvasionFoolbox
from secml.ml import CClassifierPyTorch
from secml.array import CArray
from secml.ml.classifiers.loss import CSoftmax
from secml.ml.peval.metrics import CMetricAccuracy
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.features.normalization import CNormalizerMinMax
from secml.explanation import CExplainerIntegratedGradients
from secml.figure import CFigure
import foolbox as fb
from misc import logo


print(f"RobustBench version: {robustbench.__name__}")
print(f"SecML version: {secml.__version__}")
print(f"Foolbox version: {fb.__version__}")
print(f"Numpy version: {np.__version__}")


"""## Global Variables
Contains definition of global variables
"""

input_shape    = (3, 32, 32)

model_names    = [
    "Ding2020MMA",
    "Wong2020Fast",
    "Andriushchenko2020Understanding",
    "Sitawarin2020Improving",
    "Cui2023Decoupled_WRN-28-10"
]

n_samples      = 64

dataset_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

"""## Loading models

Loads five models from robustbench. We have chosen the following models:
*   1 - Ding2020MMA
*   2 - Wong2020Fast
*   3 - Andriushchenko2020Understanding
*   4 - Sitawarin2020Improving
*   5 - Cui2023Decoupled_WRN-28-10

CIFAR-10
Linf, eps=8/255

"""


def load_model(model_name):
    """Carica un modello da RobustBench e lo avvolge in CClassifierPyTorch."""
    try:
        model = robustbench.utils.load_model(
            model_name=model_name, dataset='cifar10', threat_model='Linf'
        )
        return CClassifierPyTorch(
            model=model, input_shape=input_shape,
            pretrained=True, pretrained_classes=CArray(range(10)), preprocess=None
        )
    except Exception as e:
        print(f"Errore durante il caricamento del modello {model_name}: {e}")
        return None
models = [load_model(name) for name in model_names if load_model(name) is not None]



"""## Loading  CIFAR-10
Loads 64 samples from CIFAR-10 dataset with shape (3, 32, 32)
"""
tr, ts = CDataLoaderCIFAR10().load()

# Normalizzazione con backup
normalizer = CNormalizerMinMax().fit(tr.X)
ts_original = ts.deepcopy()  # Backup prima della normalizzazione
ts.X = normalizer.transform(ts.X)

# Ridurre a 64 campioni e reshaping
ts = ts[:n_samples, :]
ts.X = CArray(ts.X.tondarray().reshape(-1, 3, 32, 32))


"""## Fast-Minimum-Norm (FMN) attack
Computes the accuracy of the models, just to confirm that it is working properly.
"""

# Calcolo delle predizioni e accuratezza dei modelli
metric        = CMetricAccuracy()
models_preds  = [clf.predict(ts.X) for clf in models]
accuracies    = [metric.performance_score(y_true=ts.Y, y_pred=y_pred) for y_pred in models_preds]

print("-" * 90)
# Stampa delle accuratezze
for idx in range(len(model_names)):
    print(f"Model name: {model_names[idx]:<40} - Clean model accuracy: {(accuracies[idx] * 100):.2f} %")
print("-" * 90)



def compute_explainability(explainer_class, model, adv_ds, num_classes):
    """Calcola l'explicabilità per i campioni avversari."""
    attributions = []
    if explainer_class:
        explainer = explainer_class(model)
        for idx in range(adv_ds.X.shape[0]):
            x_adv = adv_ds.X[idx, :]
            attr = CArray.empty(shape=(num_classes, x_adv.size))
            for c in range(num_classes):
                attr[c, :] = explainer.explain(x_adv, y=c)
            attributions.append(attr)
    return attributions


def FNM_attack(samples, labels, model, explainer_class=None, num_classes=10):
    """Esegue l'attacco FMN e raccoglie spiegabilità, perturbazione e confidenza."""
    init_params = dict(steps=500, max_stepsize=1.0, gamma=0.05)

    try:
        attack = CAttackEvasionFoolbox(
            classifier=model, y_target=None, epsilons=8/255,
            fb_attack_class=fb.attacks.LInfFMNAttack, **init_params
        )

        y_pred, _, adv_ds, _ = attack.run(samples, labels)

        # Calcola la spiegabilità in una funzione separata
        attributions = compute_explainability(explainer_class, model, adv_ds, num_classes) if explainer_class else None

        return {
            'x_seq': attack.x_seq,
            'y_pred_adv': y_pred,
            'adv_ds': adv_ds,
            'attributions': attributions,
            'confidence': CSoftmax().softmax(model.predict(attack.x_seq, return_decision_function=True)[1]),
            'iterations': CArray.arange(attack.x_seq.shape[0])
        }

    except Exception as e:
        print(f"Errore durante l'attacco: {e}")
        return {'error': str(e)}





"""##Saves or loads  attack data on the disk"""
for idx, model in enumerate(models):
    if not isinstance(model, CClassifierPyTorch):
        print(f"Errore: Il modello {model_names[idx]} non è un'istanza di CClassifierPyTorch.")
    else:
        print(f"Il modello {model_names[idx]} è caricato correttamente come CClassifierPyTorch.")



def load_results(file_path):
    """Carica i risultati dell'attacco da file, gestendo errori di corruzione."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
                if isinstance(results, list):
                    return results
        except (pickle.PickleError, EOFError) as e:
            print(f"Errore nel caricamento di '{file_path}': {e}")
    return []

results_FNM = load_results('extracted_data/data_attack_result_FNM.pkl')
if not results_FNM:
    results_FNM = []
    for model, name in zip(models, model_names):
        print(f"Eseguendo attacco su \"{name}\"...")
        attack_result = FNM_attack(ts.X, ts.Y, model, CExplainerIntegratedGradients, len(dataset_labels))
        results_FNM.append({'model_name': name, 'result': attack_result})

    with open('extracted_data/data_attack_result_FNM.pkl', 'wb') as f:
        pickle.dump(results_FNM, f)


# from google.colab import files

# with open('attack_data.bin', 'wb') as file:
#     pickle.dump(attack_data, file)

# files.download('attack_data.bin')

with open('extracted_data/data_attack_result_FNM.pkl', 'rb') as file:
    attack_data = pickle.load(file)




# Calcolo delle predizioni e accuratezza dei modelli
metric        = CMetricAccuracy()
models_preds  = [clf.predict(ts.X) for clf in models]
accuracies    = [metric.performance_score(y_true=ts.Y, y_pred=y_pred) for y_pred in models_preds]


print("-" * 90)
# Stampa delle accuratezze dopo l'attacco

for idx in range(len(model_names)):
    accuracy = metric.performance_score(
        y_true=ts.Y,
        y_pred=attack_data[idx]['result']['y_pred_adv']
    )
    print(f"Model name: {model_names[idx]:<40} - Model accuracy under attack: {(accuracy * 100):.2f} %")
print("-" * 90)

#############################################################################################



def convert_image(image):
    """
    Converte un'immagine da CArray o NumPy in formato (H, W, C).
    """
    if hasattr(image, "tondarray"):  # Se è un CArray
        image = image.tondarray()  # Converti in NumPy
    return image.reshape(input_shape).transpose(1, 2, 0)


def show_image(fig, local_idx, img, img_adv, expl, label, pred):
    """
    Mostra l'immagine originale, avversa, perturbazione e spiegazione.
    """
    fsize=28
    # Calcolo della perturbazione
    diff_img = img_adv - img
    diff_img -= diff_img.min()
    diff_img /= diff_img.max()

    # Calcola la posizione del subplot nella griglia
    fig.subplot(3, 4, local_idx + 1)  # `local_idx` parte da 0

    # Immagine originale
    fig.sp.imshow(convert_image(img))
    fig.sp.title(f"True: {label}", fontsize=fsize)
    fig.sp.xticks([])
    fig.sp.yticks([])

    # Immagine avversa
    fig.subplot(3, 4, local_idx + 2)
    fig.sp.imshow(convert_image(img_adv))
    fig.sp.title(f'Adv: {pred}', fontsize=fsize)
    fig.sp.xticks([])
    fig.sp.yticks([])

    # Perturbazione
    fig.subplot(3, 4, local_idx + 3)
    fig.sp.imshow(convert_image(diff_img))
    fig.sp.title('Perturbation', fontsize=fsize)
    fig.sp.xticks([])
    fig.sp.yticks([])

    # Spiegazione
    expl = convert_image(expl)
    r = np.fabs(expl[:, :, 0])
    g = np.fabs(expl[:, :, 1])
    b = np.fabs(expl[:, :, 2])
    expl = np.maximum(np.maximum(r, g), b)

    fig.subplot(3, 4, local_idx + 4)
    fig.sp.imshow(expl, cmap='seismic')
    fig.sp.title('Explain', fontsize=fsize)
    fig.sp.xticks([])
    fig.sp.yticks([])




# Numero massimo di subplot per figura
max_subplots = 20  # 5 righe x 4 colonne
n_cols = 4  # Numero fisso di colonne
epsilon = 8 / 255  # Limite di perturbazione

# Itera sui modelli
for model_id in range(len(models)):
    print(f"\nVisualizzazione per il modello: {model_names[model_id]}")

    adv_ds = attack_data[model_id]['result']['adv_ds']
    y_adv = attack_data[model_id]['result']['y_pred_adv']
    attributions = attack_data[model_id]['result']['attributions']

    # Reshape delle immagini in formato (n_samples, 3, 32, 32)
    adv_images = adv_ds.X.tondarray().reshape(-1, 3, 32, 32)
    original_images = ts.X.tondarray().reshape(-1, 3, 32, 32)

    # Calcola la distanza L∞ per tutti i campioni
    distances = np.abs(adv_images - original_images).max(axis=(1, 2, 3))

    # Filtra i campioni che soddisfano le condizioni
    selected_indices = [
        idx for idx in range(ts.X.shape[0])
        if (distances[idx] < epsilon and y_adv[idx] != ts.Y[idx])
    ]

    print(f"Campioni selezionati per il modello {model_names[model_id]}: {len(selected_indices)}")

    valid_indices = []  # Per salvare i campioni validi
    for idx in selected_indices:
        img = original_images[idx]
        img_adv = adv_images[idx]
        diff_img = img_adv - img

        # Controllo per evitare divisione per zero
        if diff_img.max() > 1e-6:
            valid_indices.append(idx)

        # Interrompe quando abbiamo 3 campioni validi
        if len(valid_indices) == 3:
            break

    print(f"Campioni validi per il modello {model_names[model_id]}: {len(valid_indices)}")

    if len(valid_indices) > 0:
        # Crea una nuova figura per i campioni selezionati
        n_rows = len(valid_indices)  # Una riga per ogni campione
        fig = CFigure(height=n_rows * 6, width=18)

        # Aggiungi manualmente il titolo sopra la figura accedendo alla figura Matplotlib interna
        fig.title(f"Explainability for Model: {model_names[model_id]}", fontsize=32)

        for ydx, idx in enumerate(valid_indices):
            img = original_images[idx]
            img_adv = adv_images[idx]
            expl = attributions[idx][y_adv[idx].item(), :]

            # Calcola la differenza e normalizza
            diff_img = img_adv - img
            diff_img /= diff_img.max()  # Sicuro, poiché controllato prima

            # Calcola l'indice locale per il subplot
            local_idx = ydx * 4

            # Mostra l'immagine nel subplot calcolato
            show_image(
                fig,
                local_idx,
                img,
                img_adv,
                expl,
                dataset_labels[ts.Y[idx].item()],
                dataset_labels[y_adv[idx].item()]
            )

        # Completa e salva la figura per i campioni selezionati
        fig.tight_layout(rect=[0, 0.003, 1, 0.94])
        fig.savefig(f"results/Explainability_model_{model_names[model_id]}.jpg")


#############################################################################################################


results_file_confidence = 'extracted_data/data_attack_result_FNM_CONFIDENCE.pkl'
num_samples_to_process = 5  # Numero di samples da processare

# Controlla se il file esiste
if os.path.exists(results_file_confidence):
    print(f"Il file '{results_file_confidence}' esiste già. Caricamento dei risultati salvati...")
    try:
        with open(results_file_confidence, 'rb') as f:
            CONFIDENCE_results_FNM = pickle.load(f)
            if isinstance(CONFIDENCE_results_FNM, list) and all(isinstance(r, list) for r in CONFIDENCE_results_FNM):
                print("Risultati caricati correttamente.")
            else:
                print("Attenzione: il formato dei dati caricati non è quello previsto.")
    except Exception as e:
        print(f"Errore durante il caricamento: {e}")
else:
    print(f"Il file '{results_file_confidence}' non esiste. Generando nuovi risultati...")
    CONFIDENCE_results_FNM = []
    
    for sample_id in range(num_samples_to_process):
        sample = ts.X[sample_id, :].atleast_2d()
        label = CArray(ts.Y[sample_id])
        sample_results = []
        
        for idx, model in enumerate(models):
            print(f"Analizzando il campione {sample_id} con il modello \"{model_names[idx]}\"...")
            attack_result = FNM_attack(
                samples=sample,
                labels=label,
                model=model,
                explainer_class=CExplainerIntegratedGradients,
                num_classes=len(dataset_labels)
            )
            sample_results.append({
                'model_name': model_names[idx],
                'sample_id': sample_id,
                'result': attack_result
            })
        CONFIDENCE_results_FNM.append(sample_results)
        print(f"Attacco completato per il campione {sample_id}.")

    try:
        with open(results_file_confidence, 'wb') as f:
            pickle.dump(CONFIDENCE_results_FNM, f)
            print(f"Risultati salvati nel file '{results_file_confidence}'.")
    except Exception as e:
        print(f"Errore durante il salvataggio dei risultati: {e}")




# Creazione della figura per i primi 5 campioni
for sample_id in range(num_samples_to_process):
    fig = CFigure(width=30, height=4, fontsize=10, linewidth=2)
    label_true = ts.Y[sample_id].item()

    for model_id in range(5):
        attack_result = CONFIDENCE_results_FNM[sample_id][model_id]['result']
        x_seq = attack_result['x_seq']
        n_iter = x_seq.shape[0]
        itrs = CArray.arange(n_iter)

        scores = models[model_id].predict(x_seq, return_decision_function=True)[1]
        scores = CSoftmax().softmax(scores)

        fig.subplot(1, 5, model_id + 1)
        if model_id == 0:
            fig.sp.ylabel('Confidence')
        fig.sp.xlabel('Iteration')

        label_adv = attack_result['y_pred_adv'][0].item()

        fig.sp.plot(itrs, scores[:, label_true], linestyle='--', c='black')
        fig.sp.plot(itrs, scores[:, label_adv], c='black')
        fig.sp.xlim(top=25, bottom=0)

        fig.sp.title(f"Confidence Sample {sample_id+1} - Model: {model_id+1}")
        fig.sp.legend(['Confidence True Class', 'Confidence Adv. Class'])

    fig.tight_layout()
    fig.savefig(f"results/Confidence_Sample_{sample_id+1}.jpg")
