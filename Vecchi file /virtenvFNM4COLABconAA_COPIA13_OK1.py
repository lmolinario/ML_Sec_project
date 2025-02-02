import robustbench
import secml
from secml.adv.attacks.evasion import CAttackEvasionFoolbox
import os
import pickle

from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.features.normalization import CNormalizerMinMax
from secml.explanation import CExplainerIntegratedGradients
from secml.figure import CFigure
import foolbox as fb
from secml.ml.peval.metrics import CMetricAccuracy
import torch
from autoattack import AutoAttack
from secml.array import CArray
import matplotlib.pyplot as plt

import numpy as np
from secml.ml.classifiers.loss import CSoftmax

from secml.ml import CClassifierPyTorch

print(f"RobustBench version: {robustbench.__name__}")
print(f"SecML version: {secml.__version__}")
print(f"Foolbox version: {fb.__version__}")
print(f"Numpy version: {np.__version__}")

"""## Global Variables
Contains definition of global variables
"""

# Percorsi dei file per salvare i risultati degli attacchi
results_file_AA = "../extracted_data/data_attack_result_AA.pkl"
results_file_FMN = '../extracted_data/data_attack_result_FMN.pkl'
results_file_confidence = '../extracted_data/data_attack_result_FMN_CONFIDENCE.pkl'

# Forma dell'input (canali, altezza, larghezza) per i modelli di deep learning
input_shape = (3, 32, 32)

# Lista dei modelli selezionati per l'analisi da RobustBench
model_names = [
    "Ding2020MMA",  # Modello MMA di Ding (2020)
    "Wong2020Fast",  # Modello Fast di Wong (2020)
    "Andriushchenko2020Understanding",  # Modello di Andriushchenko (2020)
    "Sitawarin2020Improving",  # Modello migliorato di Sitawarin (2020)
    "Cui2023Decoupled_WRN-28-10"  # Modello Wide ResNet 28-10 di Cui (2023)
]

# Numero di campioni da utilizzare nei test
n_samples = 64

# Etichette delle classi del dataset CIFAR-10
dataset_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Configurazione della visualizzazione dei risultati
max_subplots = 20  # Numero massimo di subplot per figura (5 righe x 4 colonne)
n_cols = 4  # Numero fisso di colonne nei plot

# Limite massimo della perturbazione per l'attacco adversariale (Linf norm)
epsilon = 8 / 255  # Converte il valore in scala [0,1]


def load_model(model_name):
    """
    Carica un modello pre-addestrato da RobustBench e lo avvolge in CClassifierPyTorch
    per l'integrazione con SecML.

    Parametri:
    - model_name (str): Nome del modello da caricare.

    Ritorna:
    - CClassifierPyTorch: Modello avvolto in un classificatore compatibile con SecML.
    - None: Se il caricamento del modello fallisce.
    """
    try:
        # Caricamento del modello da RobustBench (dataset CIFAR-10, attacco Linf)
        model = robustbench.utils.load_model(
            model_name=model_name, dataset='cifar10', threat_model='Linf'
        )

        # Avvolge il modello in un classificatore PyTorch compatibile con SecML
        return CClassifierPyTorch(
            model=model,
            input_shape=input_shape,  # Forma dell'input: (3, 32, 32)
            pretrained=True,  # Usa i pesi pre-addestrati
            pretrained_classes=CArray(range(10)),  # Classi di output (0-9 per CIFAR-10)
            preprocess=None  # Nessuna pre-elaborazione aggiuntiva
        )

    except Exception as e:
        print(f"⚠️ Errore durante il caricamento del modello '{model_name}': {e}")
        return None


def compute_explainability(explainer_class, model, adv_ds, num_classes):
    """
    Calcola l'explicabilità per i campioni avversari utilizzando un metodo di spiegazione.

    Parametri:
    - explainer_class: Classe dell'explainer da usare (es. CExplainerIntegratedGradients).
    - model: Il modello su cui calcolare le spiegazioni.
    - adv_ds: Dataset contenente i campioni avversari generati dall'attacco.
    - num_classes: Numero di classi nel dataset.

    Ritorna:
    - attributions (list): Lista contenente le attribuzioni di ogni campione avversario.
    """
    attributions = []

    if explainer_class:
        explainer = explainer_class(model)  # Inizializza l'explainer per il modello

        # Itera su tutti i campioni avversari
        for idx in range(adv_ds.X.shape[0]):
            x_adv = adv_ds.X[idx, :]  # Estrai il campione avversario corrente

            # Crea un array vuoto per memorizzare le attribuzioni per ogni classe
            attr = CArray.empty(shape=(num_classes, x_adv.size))

            # Calcola l'attribuzione per ciascuna classe
            for c in range(num_classes):
                attr[c, :] = explainer.explain(x_adv, y=c)

            attributions.append(attr)  # Aggiungi l'attribuzione alla lista

    return attributions




def AA_attack(samples, labels, models, model_names, explainer_class=None, num_classes=10):
    """
    Esegue AutoAttack sui modelli specificati e raccoglie i risultati,
    tra cui spiegabilità, perturbazione e confidenza.

    Parametri:
    - samples (CArray): Campioni di input da attaccare.
    - labels (CArray): Etichette corrispondenti ai campioni.
    - models (list): Lista di modelli CClassifierPyTorch su cui eseguire l'attacco.
    - model_names (list): Nomi dei modelli corrispondenti.
    - explainer_class (opzionale): Classe di spiegazione (es. CExplainerIntegratedGradients).
    - num_classes (int): Numero di classi nel dataset.

    Ritorna:
    - dict: Un dizionario con i risultati dell'attacco, incluse predizioni avversarie e accuratezza post-attacco.
    """

    try:
        # Imposta il dispositivo per l'attacco (GPU se disponibile, altrimenti CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        metric = CMetricAccuracy()  # Metodologia per calcolare l'accuratezza

        # Converti i campioni in tensori PyTorch e sposta su GPU/CPU
        x_test_torch = torch.tensor(samples.tondarray(), dtype=torch.float32).to(device)
        y_test_torch = torch.tensor(labels.tondarray(), dtype=torch.long).to(device)

        # Assicurarsi che i dati abbiano la forma corretta [batch, channels, height, width]
        if len(x_test_torch.shape) == 2:
            x_test_torch = x_test_torch.view(-1, *input_shape)

        results = []

        # Itera su ciascun modello per eseguire AutoAttack
        for idx, model in enumerate(models):
            try:
                print(f"\n🔍 Esecuzione di AutoAttack su: {model_names[idx]}")

                # Estrai il modello PyTorch nativo da CClassifierPyTorch
                pytorch_model = model.model.to(device)
                pytorch_model.eval()  # Imposta il modello in modalità valutazione

                # Creazione dell'attaccante AutoAttack (norma Linf con epsilon 8/255)
                adversary = AutoAttack(pytorch_model, norm='Linf', eps=8 / 255)
                adversary.apgd.n_restarts = 1  # Numero di restart per l'APGD

                # Esegui l'attacco
                x_adv_torch = adversary.run_standard_evaluation(x_test_torch, y_test_torch)

                # Converti i dati avversari da PyTorch Tensor a SecML CArray
                x_adv = CArray(x_adv_torch.cpu().detach().numpy())

                # Predizioni del modello sui dati avversari
                y_pred_adv = model.predict(x_adv)

                # Calcolo dell'accuratezza dopo l'attacco
                accuracy_under_attack = metric.performance_score(y_true=labels, y_pred=y_pred_adv)

                # Calcolo della spiegabilità, se richiesto
                attributions = compute_explainability(explainer_class, model, x_adv, num_classes) if explainer_class else None

                # Salvataggio dei risultati per il modello corrente
                results.append({
                    'model_name': model_names[idx],
                    'x_adv': x_adv,  # Immagini avversarie
                    'y_pred_adv': y_pred_adv,  # Predizioni dopo l'attacco
                    'accuracy_under_attack': accuracy_under_attack,  # Accuratezza del modello sotto attacco
                    'attributions': attributions  # Spiegabilità, se disponibile
                })

            except Exception as e:
                print(f"⚠️ Errore durante l'attacco su {model_names[idx]}: {e}")

        return results  # Restituisce l'elenco dei risultati per tutti i modelli analizzati

    except Exception as e:
        print(f"❌ Errore generale nell'esecuzione di AutoAttack: {e}")
        return {'error': str(e)}




def FMN_attack(samples, labels, model, explainer_class=None, num_classes=10):
    """
    Esegue l'attacco Fast-Minimum-Norm (FMN) utilizzando Foolbox e raccoglie i risultati.

    Parametri:
    - samples (CArray): Campioni di input da attaccare.
    - labels (CArray): Etichette corrispondenti ai campioni.
    - model (CClassifierPyTorch): Modello target dell'attacco.
    - explainer_class (opzionale): Classe di spiegazione (es. CExplainerIntegratedGradients).
    - num_classes (int): Numero di classi nel dataset.

    Ritorna:
    - dict: Dizionario con i risultati dell'attacco, incluse predizioni avversarie, perturbazioni e confidenza.
    """

    # Parametri dell'attacco FMN
    init_params = {
        'steps': 500,          # Numero massimo di iterazioni dell'attacco
        'max_stepsize': 1.0,   # Passo massimo dell'aggiornamento
        'gamma': 0.05          # Parametro di scalatura per la ricerca della perturbazione minima
    }

    try:
        # Configura l'attacco FMN utilizzando Foolbox all'interno di SecML
        attack = CAttackEvasionFoolbox(
            classifier=model,                  # Modello da attaccare
            y_target=None,                     # Attacco non mirato (target-free)
            epsilons=epsilon,                  # Limite di perturbazione (Linf)
            fb_attack_class=fb.attacks.LInfFMNAttack,  # Classe di attacco FMN di Foolbox
            **init_params                      # Parametri dell'attacco
        )

        # Esegui l'attacco e raccogli i risultati
        y_pred, _, adv_ds, _ = attack.run(samples, labels)

        # Calcola la spiegabilità se richiesto
        attributions = compute_explainability(explainer_class, model, adv_ds, num_classes) if explainer_class else None

        # Restituisce i risultati dell'attacco
        return {
            'x_seq': attack.x_seq,  # Sequenza dei campioni generati durante l'attacco
            'y_pred_adv': y_pred,   # Predizioni del modello dopo l'attacco
            'adv_ds': adv_ds,       # Dataset contenente le immagini avversarie
            'attributions': attributions,  # Attribuzioni di spiegabilità (se richieste)
            'confidence': CSoftmax().softmax(model.predict(attack.x_seq, return_decision_function=True)[1]),  # Confidenza sulle classi
            'iterations': CArray.arange(attack.x_seq.shape[0])  # Numero di iterazioni dell'attacco
        }

    except Exception as e:
        print(f"❌ Errore durante l'esecuzione dell'attacco FMN: {e}")
        return {'error': str(e)}




def load_results(file_path):
    """
    Carica i risultati da un file pickle, gestendo eventuali errori di corruzione o assenza del file.

    Parametri:
    - file_path (str): Percorso del file contenente i dati salvati.

    Ritorna:
    - list: I risultati caricati se il file è valido, altrimenti una lista vuota.
    """
    if os.path.exists(file_path):  # Verifica se il file esiste
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)

                # Verifica che il formato dei dati sia corretto
                if isinstance(results, list):
                    return results
                else:
                    print(f"⚠️ Attenzione: il formato dei dati caricati da '{file_path}' non è valido.")

        except (pickle.PickleError, EOFError, FileNotFoundError, OSError) as e:
            print(f"⚠️ Errore nel caricamento di '{file_path}': {e}")

    return []  # Ritorna una lista vuota se il caricamento fallisce


def save_results(file_path, data):
    """
    Salva i risultati in un file pickle, gestendo errori di scrittura e assicurando la validità del file.

    Parametri:
    - file_path (str): Percorso del file in cui salvare i dati.
    - data (list/dict): Dati da salvare nel file.

    Ritorna:
    - None
    """
    try:
        # Creazione della directory se non esiste
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Salvataggio dei dati con Pickle
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            f.flush()  # Forza la scrittura su disco

        print(f"✅ Risultati salvati con successo in '{file_path}'.")

    except (OSError, pickle.PickleError) as e:
        print(f"⚠️ Errore durante il salvataggio in '{file_path}': {e}")


def convert_image(image):
    """
    Converte un'immagine da CArray o NumPy in formato (H, W, C).
    """
    try:
        # Se l'immagine è un CArray, convertila in NumPy
        if hasattr(image, "tondarray"):
            image = image.tondarray()

        # Se l'immagine è un array 1D con dimensione 3072, deve essere trasformata
        if image.shape == (1, 3072) or image.shape == (3072,):
            image = image.reshape(3, 32, 32)

        # Controlla nuovamente la dimensione
        if image.shape != input_shape:
            raise ValueError(f"Dimensioni errate: attese {input_shape}, trovate {image.shape}")

        # Converti l'immagine da (C, H, W) a (H, W, C)
        return image.transpose(1, 2, 0)

    except Exception as e:
        print(f"⚠️ Errore nella conversione dell'immagine: {e}")
        return None  # Ritorna None in caso di errore



def show_image(fig, local_idx, img, img_adv, expl, label, pred):
    """
    Mostra una sequenza di quattro immagini in un grafico: originale, avversa, perturbazione e spiegazione.

    Parametri:
    - fig (CFigure): Figura SecML per la visualizzazione.
    - local_idx (int): Indice locale per il subplot.
    - img (CArray/ndarray): Immagine originale.
    - img_adv (CArray/ndarray): Immagine avversa generata dall'attacco.
    - expl (CArray/ndarray): Mappa di spiegazione del modello.
    - label (str): Etichetta della classe originale.
    - pred (str): Etichetta della classe predetta dopo l'attacco.

    Ritorna:
    - None: La funzione modifica la figura in-place.
    """

    fsize = 28  # Dimensione dei titoli delle immagini

    try:
        # Calcolo della perturbazione normalizzata tra 0 e 1
        diff_img = img_adv - img
        diff_img -= diff_img.min()
        diff_img /= max(diff_img.max(), 1e-6)  # Evita divisioni per zero

        # Immagine originale
        fig.subplot(3, 4, local_idx + 1)  # `local_idx` parte da 0
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

        # Spiegazione (normalizzata per migliore visualizzazione)
        expl = convert_image(expl)
        r, g, b = np.fabs(expl[:, :, 0]), np.fabs(expl[:, :, 1]), np.fabs(expl[:, :, 2])
        expl = np.maximum(np.maximum(r, g), b)

        fig.subplot(3, 4, local_idx + 4)
        fig.sp.imshow(expl, cmap='seismic')
        fig.sp.title('Explain', fontsize=fsize)
        fig.sp.xticks([])
        fig.sp.yticks([])

    except Exception as e:
        print(f"⚠️ Errore durante la visualizzazione delle immagini: {e}")


def generate_confidence_results(num_samples, models, model_names, dataset_labels, ts):
    """
    Genera i risultati dell'attacco FMN per un numero limitato di campioni.

    Parametri:
    - num_samples (int): Numero di campioni da processare.
    - models (list): Lista di modelli CClassifierPyTorch su cui eseguire l'attacco.
    - model_names (list): Nomi dei modelli corrispondenti.
    - dataset_labels (list): Etichette delle classi del dataset.
    - ts (CDataset): Dataset di test contenente le immagini e le etichette.

    Ritorna:
    - list: Lista contenente i risultati dell'attacco per ogni modello e campione.
    """

    results = []  # Lista per memorizzare i risultati

    try:
        for sample_id in range(num_samples):
            sample = ts.X[sample_id, :].atleast_2d()  # Estrai il campione come matrice 2D
            label = CArray(ts.Y[sample_id])  # Etichetta del campione corrente
            sample_results = []  # Risultati dell'attacco per ogni modello

            # Itera su ciascun modello per eseguire l'attacco FMN
            for idx, model in enumerate(models):
                try:
                    print(f"🔍 Analizzando il campione {sample_id} con il modello \"{model_names[idx]}\"...")

                    attack_result = FMN_attack(
                        samples=sample,
                        labels=label,
                        model=model,
                        explainer_class=CExplainerIntegratedGradients,
                        num_classes=len(dataset_labels)
                    )

                    # Salva i risultati per il modello corrente
                    sample_results.append({
                        'model_name': model_names[idx],
                        'sample_id': sample_id,
                        'result': attack_result
                    })

                except Exception as e:
                    print(f"⚠️ Errore durante l'attacco sul modello \"{model_names[idx]}\" per il campione {sample_id}: {e}")

            results.append(sample_results)  # Aggiungi i risultati del campione alla lista principale
            print(f"✅ Attacco completato per il campione {sample_id}.")

    except Exception as e:
        print(f"❌ Errore generale nella generazione dei risultati dell'attacco FMN: {e}")

    return results  # Restituisce la lista dei risultati

def explainability_analysis(models, model_names, results_FMN, ts, dataset_labels, input_shape, epsilon=8 / 255):
    """
    Esegue l'analisi di explainability per una lista di modelli attaccati con FMN.

    Parametri:
    - models (list): Lista dei modelli.
    - model_names (list): Nomi dei modelli corrispondenti.
    - results_FMN (list): Risultati dell'attacco FMN.
    - ts (CDataset): Dataset originale.
    - dataset_labels (list): Etichette delle classi del dataset.
    - input_shape (tuple): Forma dell'input delle immagini (C, H, W).
    - epsilon (float, opzionale): Soglia per la selezione dei campioni (default 8/255).

    Ritorna:
    - None: Salva i risultati della explainability come immagini.
    """

    print("🧐 Inizio dell'analisi di Explainability")

    try:
        for model_id, model_name in enumerate(model_names):
            print(f"\n🔍 Analizzando il modello: {model_name}")

            # Estrazione dei dati relativi all'attacco
            adv_ds = results_FMN[model_id]['result'].get('adv_ds')
            y_adv = results_FMN[model_id]['result'].get('y_pred_adv')
            attributions = results_FMN[model_id]['result'].get('attributions')

            if adv_ds is None or y_adv is None or attributions is None:
                print(f"⚠️ Dati mancanti per il modello {model_name}. Passo al successivo.")
                continue

            # Reshape delle immagini nel formato corretto (batch, C, H, W)
            adv_images = adv_ds.X.tondarray().reshape(-1, *input_shape)
            original_images = ts.X.tondarray().reshape(-1, *input_shape)

            # Calcola la distanza L∞ tra immagini originali e avversarie
            distances = np.abs(adv_images - original_images).max(axis=(1, 2, 3))

            # Selezione dei campioni validi: attacco riuscito e perturbazione < epsilon
            selected_indices = [
                idx for idx in range(ts.X.shape[0])
                if (distances[idx] < epsilon and y_adv[idx] != ts.Y[idx])
            ]

            print(f"✅ Campioni validi per il modello {model_name}: {len(selected_indices)}")

            valid_indices = []
            for idx in selected_indices:
                img = original_images[idx]
                img_adv = adv_images[idx]
                diff_img = img_adv - img

                # Evita la divisione per zero e controlla se la perturbazione è significativa
                if diff_img.max() > 1e-6:
                    valid_indices.append(idx)

                if len(valid_indices) == 3:  # Seleziona i primi 3 campioni validi
                    break

            print(f"📊 Selezionati {len(valid_indices)} campioni per la visualizzazione.")

            if valid_indices:
                # Imposta la figura per la visualizzazione
                n_rows = len(valid_indices)
                fig = CFigure(height=n_rows * 6, width=18)
                fig.title(f"Explainability for Model: {model_name}", fontsize=32)

                for ydx, idx in enumerate(valid_indices):
                    img = original_images[idx]
                    img_adv = adv_images[idx]
                    expl = attributions[idx][y_adv[idx].item(), :]

                    # Normalizza la perturbazione per migliorare la visualizzazione
                    diff_img = img_adv - img
                    diff_img /= diff_img.max()

                    local_idx = ydx * 4  # Posizione del subplot

                    # Mostra le immagini nella figura
                    show_image(
                        fig,
                        local_idx,
                        img,
                        img_adv,
                        expl,
                        dataset_labels[ts.Y[idx].item()],
                        dataset_labels[y_adv[idx].item()]
                    )

                # Finalizza e salva la figura
                fig.tight_layout(rect=[0, 0.003, 1, 0.94])
                fig.savefig(f"results/Explainability_model_{model_name}.jpg")
                print(f"📸 Immagine di explainability salvata per il modello {model_name}.")

    except Exception as e:
        print(f"❌ Errore durante l'analisi di explainability: {e}")

def confidence_analysis(models, model_names, ts, dataset_labels, results_file_confidence, num_samples=5):
    """
    Calcola e genera i plot della confidence per i primi N campioni.

    Parametri:
    - models (list): Lista dei modelli.
    - model_names (list): Nomi dei modelli corrispondenti.
    - ts (CDataset): Dataset originale.
    - dataset_labels (list): Etichette delle classi del dataset.
    - results_file_confidence (str): Percorso del file per salvare/caricare i risultati.
    - num_samples (int, opzionale): Numero di campioni da analizzare (default: 5).

    Ritorna:
    - None: Salva i grafici delle confidence.
    """

    print("\n📊 Analisi della confidence per i campioni attaccati...")

    # Caricamento o generazione dei risultati
    confidence_results_FMN = load_results(results_file_confidence)

    if not confidence_results_FMN:  # Se il file non esiste o è corrotto
        print(f"⚠️ Il file '{results_file_confidence}' non esiste o è corrotto. Generando nuovi risultati...")
        confidence_results_FMN = generate_confidence_results(
            num_samples, models, model_names, dataset_labels, ts
        )
        save_results(results_file_confidence, confidence_results_FMN)

    # Itera sui primi `num_samples` campioni
    for sample_id in range(num_samples):
        try:
            print(f"\n🔍 Generazione del plot della Confidence per il Sample n.{sample_id + 1}")

            # Creazione della figura per visualizzare la confidence per ogni modello
            fig = CFigure(width=30, height=4, fontsize=10, linewidth=2)
            label_true = ts.Y[sample_id].item()  # Etichetta vera del campione

            for model_id, model_name in enumerate(model_names):
                try:
                    attack_result = confidence_results_FMN[sample_id][model_id]['result']

                    # Estrai la sequenza delle immagini avversarie generate durante l'attacco
                    x_seq = attack_result['x_seq']
                    n_iter = x_seq.shape[0]
                    itrs = CArray.arange(n_iter)  # Assegna le iterazioni per l'asse X

                    # Calcola le confidence per la classe vera e la classe avversaria
                    scores = models[model_id].predict(x_seq, return_decision_function=True)[1]
                    scores = CSoftmax().softmax(scores)  # Normalizza i punteggi con softmax

                    # Determina la classe avversaria (predetta dopo l'attacco)
                    label_adv = attack_result['y_pred_adv'][0].item()

                    # Creazione del subplot per ogni modello
                    fig.subplot(1, len(models), model_id + 1)

                    # Etichetta per il primo subplot
                    if model_id == 0:
                        fig.sp.ylabel('Confidence')

                    fig.sp.xlabel('Iteration')

                    # Disegna le confidence per la classe vera e avversaria
                    fig.sp.plot(itrs, scores[:, label_true], linestyle='--', c='green')  # True class
                    fig.sp.plot(itrs, scores[:, label_adv], c='red')  # Adv class
                    fig.sp.xlim(top=25, bottom=0)

                    # Titolo del subplot con info sulla classificazione
                    fig.sp.title(f"Sample {sample_id + 1} - Model: {model_name}\n"
                                 f"True Label: {dataset_labels[label_true]} ({label_true})\n"
                                 f"Adv Label: {dataset_labels[label_adv]} ({label_adv})")

                    fig.sp.legend(['Confidence True Class', 'Confidence Adv. Class'])

                except Exception as e:
                    print(f"⚠️ Errore nell'elaborazione della confidence per il modello {model_name}, Sample {sample_id + 1}: {e}")

            print(f"✅ Generazione completata per Sample n.{sample_id + 1}")

            # Salvataggio del plot
            fig.tight_layout()
            output_path = f"results/Confidence_Sample_{sample_id + 1}.jpg"
            fig.savefig(output_path)
            print(f"📸 Immagine salvata in {output_path}")

        except Exception as e:
            print(f"❌ Errore generale nella generazione della confidence per il Sample {sample_id + 1}: {e}")



import numpy as np
import matplotlib.pyplot as plt
import os

import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_comparison(original, adv_AA, adv_FMN, title, sample_idx, dataset_labels, y_true, y_pred_AA, y_pred_FMN, conf_AA, conf_FMN):
    """
    Visualizza immagini originali, attaccate da AutoAttack e FMN,
    e le perturbazioni generate. Inoltre, salva i casi in cui
    FMN ha successo e AutoAttack no.
    """

    # Riorganizza le immagini nel formato (H, W, C)
    original = original.reshape(3, 32, 32).transpose(1, 2, 0)
    adv_AA = adv_AA.reshape(3, 32, 32).transpose(1, 2, 0)
    adv_FMN = adv_FMN.reshape(3, 32, 32).transpose(1, 2, 0)

    diff_AA = np.abs(adv_AA - original)  # Perturbazione AA
    diff_FMN = np.abs(adv_FMN - original)  # Perturbazione FMN

    # Nome della classe dell'immagine originale e predette dopo gli attacchi
    label_original = dataset_labels[y_true.Y[sample_idx].item()]
    label_AA = dataset_labels[y_pred_AA[sample_idx].item()]
    label_FMN = dataset_labels[y_pred_FMN[sample_idx].item()]

    # Se la confidence è "N/A", impostiamo un valore di default 0.0
    conf_AA = conf_AA.item() if conf_AA != "N/A" else 0.0
    conf_FMN = conf_FMN.item() if conf_FMN != "N/A" else 0.0

    # Verifica se gli attacchi hanno avuto successo
    fmn_success = label_FMN != label_original
    aa_success = label_AA != label_original

    print(f"📊 Sample {sample_idx}: FMN Success = {fmn_success}, AA Success = {aa_success}")
    print(f"   Confidence AA: {conf_AA:.4f}, Confidence FMN: {conf_FMN:.4f}")

    # Calcolo delle perturbazioni L∞
    l_inf_AA = np.max(diff_AA)
    l_inf_FMN = np.max(diff_FMN) if fmn_success else 0.0

    # Funzione per plottare e salvare i risultati
    def save_plot(case, filename):
        fig, axes = plt.subplots(1, 5, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title(f"Original\nLabel: {label_original}")

        axes[1].imshow(adv_AA)
        axes[1].set_title(f"AutoAttack\nL∞={l_inf_AA:.4f}\nPred: {label_AA}\nConf: {conf_AA:.4f}")

        axes[2].imshow(adv_FMN)
        axes[2].set_title(f"FMN\nL∞={l_inf_FMN:.4f}\nPred: {label_FMN}\nConf: {conf_FMN:.4f}")

        axes[3].imshow(diff_AA / max(l_inf_AA, 1e-6), cmap="hot")
        axes[3].set_title("Perturb. AA")

        axes[4].imshow(diff_FMN / max(l_inf_FMN, 1e-6), cmap="hot" if fmn_success else "gray")
        axes[4].set_title("Perturb. FMN")

        for ax in axes:
            ax.axis('off')

        plt.suptitle(f"{title} - Sample {sample_idx} ({case})")

        # Salvataggio
        os.makedirs("../results", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        plt.savefig(f"results/{filename}_{safe_title}_Sample_{sample_idx}.png")
        plt.close(fig)  # Chiude la figura per risparmiare memoria

    # 🔹 Caso 1: AA ha successo, FMN fallisce
    if aa_success and not fmn_success:
        save_plot("AA Success - FMN Fail", "AA_Success_FMN_Fail")

    # 🔹 Caso 2: Entrambi hanno successo
    if aa_success and fmn_success:
        save_plot("AA Success - FMN Success", "AA_Success_FMN_Success")

    # 🔹 Caso 3: FMN ha successo, AA fallisce
    if fmn_success and not aa_success:
        print(f"📌 Caso speciale: FMN ha successo mentre AutoAttack no (Sample {sample_idx})")
        save_plot("FMN Success - AA Fail", "FMN_Success_AA_Fail")


def plot_comparison1(original, adv_AA, adv_FMN, title, sample_idx, dataset_labels, y_true, y_pred_AA, y_pred_FMN):
    """
    Visualizza immagini originali, attaccate da AutoAttack e FMN,
    e le perturbazioni generate. Inoltre, salva i casi in cui
    FMN ha successo e AutoAttack no.
    """
    # Riorganizza le immagini nel formato (H, W, C)
    original = original.reshape(3, 32, 32).transpose(1, 2, 0)
    adv_AA = adv_AA.reshape(3, 32, 32).transpose(1, 2, 0)
    adv_FMN = adv_FMN.reshape(3, 32, 32).transpose(1, 2, 0)

    diff_AA = np.abs(adv_AA - original)  # Perturbazione AA
    diff_FMN = np.abs(adv_FMN - original)  # Perturbazione FMN


    # Nome della classe dell'immagine originale
    label_original = dataset_labels[y_true.Y[sample_idx].item()]

    # Nome della classe predetta dopo AutoAttack
    label_AA = dataset_labels[y_pred_AA[sample_idx].item()]

    # Nome della classe predetta dopo FMN
    label_FMN = dataset_labels[y_pred_FMN[sample_idx].item()]

    # Verifica se ha modificato l'immagine
    fmn_success = (label_FMN!= label_original)
    print(f"FMN Attack Success on Sample {sample_idx}: {fmn_success}")
    aa_success = (label_AA != label_original)
    print(f"AA Attack Success on Sample {sample_idx}: {aa_success}")

    # Se FMN non ha modificato nulla, impostiamo un valore di default
    l_inf_FMN = np.max(diff_FMN, axis=(0, 1, 2)).item() if fmn_success else 0.0
    l_inf_AA = np.max(diff_AA, axis=(0, 1, 2)).item()



    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    # Immagine originale
    axes[0].imshow(original)
    axes[0].set_title(f"Original\nLabel: {label_original}")

    # Immagine avversaria AutoAttack
    axes[1].imshow(adv_AA)
    axes[1].set_title(f"AutoAttack\nL∞={l_inf_AA:.4f}\nPred: {label_AA}")

    # Immagine avversaria FMN
    axes[2].imshow(adv_FMN)
    axes[2].set_title(f"FMN\nL∞={l_inf_FMN:.4f}\nPred: {label_FMN}")

    # Mappa della perturbazione di AutoAttack
    axes[3].imshow(diff_AA / max(l_inf_AA, 1e-6), cmap="hot")
    axes[3].set_title("Perturb. AA")

    # Mappa della perturbazione di FMN (se fallisce mostra un'immagine grigia)
    if fmn_success:
        axes[4].imshow(diff_FMN / max(l_inf_FMN, 1e-6), cmap="hot")
    else:
        axes[4].imshow(np.full_like(diff_FMN, 0.5))  # Se FMN non cambia nulla, mostra un'immagine neutra

    axes[4].set_title("Perturb. FMN")

    for ax in axes:
        ax.axis('off')

    plt.suptitle(f"{title} - Sample {sample_idx}")

    # Salvataggio del file con nome pulito
    safe_title = title.replace(" ", "_").replace("/", "_")
    os.makedirs("../results", exist_ok=True)
    plt.savefig(f"results/AA_Success_FMN_Fail_{safe_title}_Sample_{sample_idx}.png")


    # 🔹 Salva i casi in cui FMN ha successo e AutoAttack no
    if (label_FMN != label_original) and (label_AA == label_original):
        print(f"📌 Caso speciale: FMN ha successo mentre AutoAttack no (Sample {sample_idx})")

        fig, axes = plt.subplots(1, 5, figsize=(15, 5))


        axes[0].imshow(original)
        axes[0].set_title(f"Original\nLabel: {label_original}")

        axes[1].imshow(adv_AA)
        axes[1].set_title(f"AutoAttack\nL∞={l_inf_AA:.4f}\nPred: {label_AA}")

        axes[2].imshow(adv_FMN)
        axes[2].set_title(f"FMN\nL∞={l_inf_FMN:.4f}\nPred: {label_FMN}")

        # Mappa della perturbazione di FMN (se fallisce mostra un'immagine grigia)
        if aa_success:
        # Mappa della perturbazione di AutoAttack
            axes[3].imshow(diff_AA / max(l_inf_AA, 1e-6), cmap="hot")

        else:
            axes[3].imshow(np.full_like(diff_AA, 0.5))  # Se FMN non cambia nulla, mostra un'immagine neutra

        axes[3].set_title("Perturb. AA")

        axes[4].imshow(diff_FMN / max(l_inf_FMN, 1e-6), cmap="hot")
        axes[4].set_title("Perturb. FMN")

        for ax in axes:
            ax.axis('off')

        plt.suptitle(f"FMN Success vs AutoAttack Failure - Sample {sample_idx}")
        plt.savefig(f"results/FMN_Success_AA_Fail_{safe_title}_Sample_{sample_idx}.png")



if __name__ == "__main__":

    # Caricamento dei modelli da RobustBench
    models = [load_model(name) for name in model_names if load_model(name) is not None]

    print("\n📥 Caricamento del dataset CIFAR-10...")
    # Caricamento del dataset CIFAR-10
    tr, ts = CDataLoaderCIFAR10().load()

    # Normalizzazione con backup del dataset originale
    normalizer = CNormalizerMinMax().fit(tr.X)
    ts_original = ts.deepcopy()  # Backup prima della normalizzazione
    ts.X = normalizer.transform(ts.X)

    # Ridurre a 64 campioni e adattare la forma delle immagini
    ts = ts[:n_samples, :]
    ts.X = CArray(ts.X.tondarray().reshape(-1, *input_shape))

    print("\n📊 Calcolo dell'accuratezza dei modelli su dati puliti...")
    # Calcolo delle predizioni e accuratezza iniziale dei modelli
    metric = CMetricAccuracy()
    models_preds = [clf.predict(ts.X) for clf in models]
    accuracies = [metric.performance_score(y_true=ts.Y, y_pred=y_pred) for y_pred in models_preds]

    print("\n✅ Verifica del caricamento dei modelli:")
    print("-" * 90)
    for idx, model in enumerate(models):
        if not isinstance(model, CClassifierPyTorch):
            print(f"⚠️ Errore: Il modello {model_names[idx]} non è un'istanza di CClassifierPyTorch.")
        else:
            print(f"✅ Il modello {model_names[idx]} è stato caricato correttamente.")
    print("-" * 90)

    # Stampa delle accuratezze pulite dei modelli
    print("\n📈 Accuratezza dei modelli su dati puliti:")
    print("-" * 90)
    for idx in range(len(model_names)):
        print(f"Model name: {model_names[idx]:<40} - Clean model accuracy: {(accuracies[idx] * 100):.2f} %")
    print("-" * 90)

    ### Attacco FMN ###
    print("\n⚡ Caricamento o generazione dei risultati dell'attacco FMN...")
    results_FMN_data = load_results(results_file_FMN)

    if not results_FMN_data:  # Se il caricamento non ha avuto successo, esegui l'attacco FMN
        print(f"⚠️ Il file '{results_file_FMN}' non esiste o è corrotto. Generando nuovi risultati...")
        results_FMN_data = [
            {'model_name': name,
             'result': FMN_attack(ts.X, ts.Y, model, CExplainerIntegratedGradients, len(dataset_labels))}
            for model, name in zip(models, model_names)
        ]
        save_results(results_file_FMN, results_FMN_data)

    # Accuratezza dopo attacco FMN
    print("\n📉 Accuratezza dei modelli sotto attacco FMN:")
    print("-" * 90)
    for idx in range(len(model_names)):
        accuracy = metric.performance_score(
            y_true=ts.Y,
            y_pred=results_FMN_data[idx]['result']['y_pred_adv']
        )
        print(f"Model name: {model_names[idx]:<40} - Accuracy under FMN attack: {(accuracy * 100):.2f} %")
    print("-" * 90)

    ### Attacco AutoAttack (AA) ###
    print("\n⚡ Caricamento o generazione dei risultati dell'attacco AutoAttack...")
    results_AA_data = load_results(results_file_AA)

    if not results_AA_data:  # Se il file non esiste, esegui l'attacco AA
        print(f"⚠️ Il file '{results_file_AA}' non esiste o è corrotto. Generando nuovi risultati...")
        results_AA_data = [
            {'model_name': name,
             'result': AA_attack(ts.X, ts.Y, model, CExplainerIntegratedGradients, len(dataset_labels))}
            for model, name in zip(models, model_names)
        ]
        save_results(results_file_AA, results_AA_data)


    # Accuratezza dopo attacco AutoAttack
    print("\n📉 Accuratezza dei modelli sotto attacco AutoAttack:")
    print("-" * 90)
    for result in results_AA_data:
        print(
            f"Model name: {result['model_name']:<40} - Accuracy under AA attack: {(result['accuracy_under_attack'] * 100):.2f} %")
    print("-" * 90)


    # Calcola la confidence per AutoAttack
    confidence_AA = []  # Inizializza come lista vuota


    for model_idx, model in enumerate(models):
        print(f"Calcolando la confidenza per il modello: {model_names[model_idx]}")

        x_adv_AA = results_AA_data[model_idx]['x_adv']  # Immagini avversarie AutoAttack
        scores_AA = model.predict(x_adv_AA, return_decision_function=True)[1]  # Ottieni logits

        # Verifica se il modello ha generato output validi
        if scores_AA is None or scores_AA.shape[0] == 0:
            print(f"⚠️ Errore: il modello {model_names[model_idx]} non ha generato predizioni valide!")
            confidence_AA.append(None)  # Evita errori di iterazione
            continue

        # Calcolo softmax per trasformare logits in probabilità
        conf_AA = CSoftmax().softmax(scores_AA)
        confidence_AA.append(conf_AA)

    print("✅ Confidenza per AutoAttack calcolata con successo!\n\n\n")


#################################################################################
    print("Identifying samples for which one attack succeeds while the other fails!")
    # Identificazione dei campioni con risultati discordanti
    mismatched_samples = {}
    for model_idx, model_name in enumerate(model_names):
        print(f"\nAnalizzando il modello: {model_name}")

        adv_ds_AA = results_AA_data[model_idx]['x_adv']  # Immagini avversarie AA
        adv_ds_FMN = results_FMN_data[model_idx]['result']['adv_ds'].X  # Immagini avversarie FMN

        confidence_FMN = results_FMN_data[model_idx]['result']['confidence']  # Confidenza FMN
        confidence_AA_model = confidence_AA[model_idx]  # Usa la confidence calcolata

        y_true = results_FMN_data[model_idx]['result']['adv_ds'].Y.tondarray()
        y_pred_AA = results_AA_data[model_idx]['y_pred_adv'].tondarray()  # Converti in array NumPy
        y_pred_FMN = results_FMN_data[model_idx]['result']['y_pred_adv'].tondarray()

        # Identificazione dei campioni discordanti
        differing_indices = [
            idx for idx in range(y_pred_AA.shape[0])
            if (y_pred_AA[idx] != y_pred_FMN[idx]) and
               (y_pred_AA[idx] == y_true[idx] or y_pred_FMN[idx] == y_true[idx])
        ]

        # Struttura aggiornata: salva indice + etichette reali e avversariali
        mismatched_samples[model_name] = [
            {
                "sample": idx,
                "true_label": dataset_labels[y_true[idx]],
                "adv_label_AA": dataset_labels[y_pred_AA[idx]],
                "adv_label_FMN": dataset_labels[y_pred_FMN[idx]]
            }
            for idx in differing_indices
        ]


        print(f"Campioni discordanti per {model_name}: {len(mismatched_samples[model_name])}")


    #################################################################################

    print("\n\n\n\n")

    # 🔹 Creazione di una mappatura tra model_name e model_idx
    model_name_to_idx = {name: idx for idx, name in enumerate(model_names)}

    Confidence_AA_4_sample = {}
    Confidence_FMN_4_sample = {}

    # Analisi finale e motivazioni
    for model_name, indices in mismatched_samples.items():
        model_idx = model_name_to_idx[model_name]  # Recupera l'indice corretto per la confidenza

        print(f"\nAnalisi dei risultati per il modello {model_name}")

        # Creazione delle entry nei dizionari per il modello corrente
        Confidence_AA_4_sample[model_name] = {}
        Confidence_FMN_4_sample[model_name] = {}

        for sample_data in indices:
            idx = sample_data["sample"]
            conf_AA = confidence_AA[model_idx][idx, y_pred_AA[idx]] if confidence_AA[model_idx] is not None else "N/A"
            conf_FMN = confidence_FMN[idx, y_pred_FMN[idx]]

            # Nome della classe dell'immagine originale
            label_original = sample_data["true_label"]

            # Nome della classe predetta dopo AutoAttack
            label_AA = sample_data["adv_label_AA"]

            # Nome della classe predetta dopo FMN
            label_FMN = sample_data["adv_label_FMN"]

            print(f"- Campione {idx}: \n  Confidenza AA={conf_AA}, Confidenza FMN={conf_FMN}")
            print(
                f"  Etichetta reale: '{label_original}'\n  Etichetta Avversariale FNM: '{label_FMN}'\n  Etichetta Avversariale AA: '{label_AA}' ")
            print("  Potenziali motivazioni:")

            if label_AA== label_original and label_FMN== label_original:
                print("  * ⚠️ Nessuno dei due attacchi ha avuto effetto: il modello potrebbe essere molto robusto.")
            elif label_AA!= label_original and label_FMN== label_original:
                print("  * ✅ AutoAttack ha avuto successo nel modificare la classe, mentre FMN ha fallito.")
            elif label_FMN != label_original and label_AA == label_original:
                print("  * ✅ FMN ha avuto successo nel modificare la classe, mentre AutoAttack ha fallito.")
            elif label_AA != label_FMN:
                print("  * 🔄 Entrambi gli attacchi hanno modificato la predizione, ma in direzioni diverse.")

            # Analisi delle confidenze
            if conf_AA != "N/A" and conf_AA > conf_FMN:
                print("  * AutoAttack ha generato una predizione più sicura rispetto a FMN.")
            elif conf_FMN > conf_AA:
                print("  * FMN ha generato una predizione più sicura rispetto ad AutoAttack.")
            else:
                print("  * Le confidence sono simili, il modello potrebbe essere resistente a entrambi gli attacchi.")

            # 🔹 Salva le confidence nei dizionari
            Confidence_AA_4_sample[model_name][idx] = conf_AA
            Confidence_FMN_4_sample[model_name][idx] = conf_FMN
            print()

    # 🔹 Ora le confidence per ogni campione discordante sono disponibili in:
    # Confidence_AA_4_sample[model_name][idx] e Confidence_FMN_4_sample[model_name][idx]

    print ('Confidence_AA_4_sample 13:',Confidence_AA_4_sample)
    print ('Confidence_FMN_4_sample 13:', Confidence_FMN_4_sample)
    #################################################################################################################

    # 🔹 Analisi della perturbazione L∞
    import numpy as np

    Perturbation_Linf = {}
    for model_name, indices in mismatched_samples.items():
        model_idx = model_name_to_idx[model_name]
        print(f"\n🔍 Analisi della perturbazione L∞ per il modello {model_name}")
        Perturbation_Linf[model_name] = {}
        for sample_data in indices:
            idx = sample_data["sample"]
            x_orig = ts.X[idx, :].tondarray().squeeze()
            x_adv_AA = results_AA_data[model_idx]['x_adv'][idx, :].tondarray().squeeze()
            x_adv_FMN = results_FMN_data[model_idx]['result']['adv_ds'].X[idx, :].tondarray().squeeze()
            linf_AA = np.linalg.norm(x_adv_AA - x_orig, ord=np.inf)
            linf_FMN = np.linalg.norm(x_adv_FMN - x_orig, ord=np.inf)
            Perturbation_Linf[model_name][idx] = {'AA': linf_AA, 'FMN': linf_FMN}
            print(f"- Campione {idx}: L∞ Perturbazione - AA: {linf_AA:.4f}, FMN: {linf_FMN:.4f}")


    import numpy as np
    import matplotlib.pyplot as plt

    # Numero massimo di campioni da visualizzare
    max_samples = 3

    for model_name, indices in Perturbation_Linf.items():
        model_idx = model_name_to_idx[model_name]  # Indice del modello
        print(f"\n📊 Visualizzazione per il modello: {model_name}")

        count = 0  # Contatore per limitare il numero di campioni

        for idx in list(indices.keys()):
            if count >= max_samples:
                break

            # Caricamento immagini originali e avversarie
            x_orig = ts.X[idx, :].tondarray().squeeze().reshape(input_shape)
            x_adv_AA = results_AA_data[model_idx]['x_adv'][idx, :].tondarray().squeeze().reshape(input_shape)
            x_adv_FMN = results_FMN_data[model_idx]['result']['adv_ds'].X[idx, :].tondarray().squeeze().reshape(
                input_shape)

            # Calcolo della perturbazione L∞
            linf_AA = float(Perturbation_Linf[model_name][idx]['AA'])
            linf_FMN = float(Perturbation_Linf[model_name][idx]['FMN'])

            # Recupero delle confidenze
            conf_AA = float(Confidence_AA_4_sample[model_name][idx].item()) if idx in Confidence_AA_4_sample[
                model_name] else None
            conf_FMN = float(Confidence_FMN_4_sample[model_name][idx].item()) if idx in Confidence_FMN_4_sample[
                model_name] else None

            # Calcolo delle perturbazioni
            perturbation_AA = np.abs(x_adv_AA - x_orig)
            perturbation_FMN = np.abs(x_adv_FMN - x_orig)

            # Normalizzazione per una migliore visibilità delle perturbazioni
            perturbation_AA = perturbation_AA / np.max(perturbation_AA) if np.max(
                perturbation_AA) > 0 else np.zeros_like(perturbation_AA)
            perturbation_FMN = perturbation_FMN / np.max(perturbation_FMN) if np.max(
                perturbation_FMN) > 0 else np.zeros_like(perturbation_FMN)

            # Recupero delle etichette delle classi
            label_original = dataset_labels[ts.Y[idx].item()]
            label_AA = dataset_labels[results_AA_data[model_idx]['y_pred_adv'][idx].item()]
            label_FMN = dataset_labels[results_FMN_data[model_idx]['result']['y_pred_adv'][idx].item()]

            # Verifica del successo degli attacchi
            attack_AA_success = label_AA != label_original
            attack_FMN_success = label_FMN != label_original

            # **Creazione dell'explainer Integrated Gradients**
            explainer = CExplainerIntegratedGradients(models[model_idx])  # Ora inizializzato correttamente

            # **Calcolo delle spiegazioni per le immagini avversarie (Fix del problema)**
            explain_AA = explainer.explain(CArray(x_adv_AA[None, :]),
                                           CArray([results_AA_data[model_idx]['y_pred_adv'][idx].item()]))
            explain_FMN = explainer.explain(CArray(x_adv_FMN[None, :]),
                                            CArray([results_FMN_data[model_idx]['result']['y_pred_adv'][idx].item()]))

            # Normalizzazione delle spiegazioni per la visualizzazione
            explain_AA = explain_AA.tondarray().squeeze()  # Rimuove dimensioni extra
            explain_FMN = explain_FMN.tondarray().squeeze()


            # Rimodellare le spiegazioni alla dimensione corretta
            explain_AA = explain_AA.reshape(input_shape)  # input_shape = (3, 32, 32)
            explain_FMN = explain_FMN.reshape(input_shape)


            # Se un attacco ha successo e l'altro no, plottiamo il campione
            if attack_AA_success != attack_FMN_success:
                fig, axes = plt.subplots(3, 3, figsize=(15, 12))

                # **Riga 1: Immagini originali e avversarie**
                axes[0, 0].imshow(x_orig.transpose(1, 2, 0))
                axes[0, 0].set_title(f"Originale\nClasse: {label_original}")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(x_adv_AA.transpose(1, 2, 0))
                axes[0, 1].set_title(f"Avversaria AA ({label_AA})\nConf: {conf_AA:.4f}")
                axes[0, 1].axis("off")

                axes[0, 2].imshow(x_adv_FMN.transpose(1, 2, 0))
                axes[0, 2].set_title(f"Avversaria FMN ({label_FMN})\nConf: {conf_FMN:.4f}")
                axes[0, 2].axis("off")

                # **Riga 2: Perturbazioni**
                axes[1, 0].axis("off")  # Slot vuoto per migliorare l'allineamento

                axes[1, 1].imshow(perturbation_AA.transpose(1, 2, 0), cmap="inferno")
                axes[1, 1].set_title(f"Perturbazione AA\nL∞: {linf_AA:.4f}")
                axes[1, 1].axis("off")

                axes[1, 2].imshow(perturbation_FMN.transpose(1, 2, 0), cmap="inferno")
                axes[1, 2].set_title(f"Perturbazione FMN\nL∞: {linf_FMN:.4f}")
                axes[1, 2].axis("off")

                # **Riga 3: Spiegabilità (Integrated Gradients)**
                axes[2, 0].axis("off")  # Slot vuoto per migliorare l'allineamento

                axes[2, 1].imshow(explain_AA.transpose(1, 2, 0), cmap="coolwarm")
                axes[2, 1].set_title(f"Explain AA ({label_AA})")
                axes[2, 1].axis("off")

                axes[2, 2].imshow(explain_FMN.transpose(1, 2, 0), cmap="coolwarm")
                axes[2, 2].set_title(f"Explain FMN ({label_FMN})")
                axes[2, 2].axis("off")

                plt.suptitle(f"Confronto Attacchi - {model_name} - Campione {idx}")

                # Salvataggio della figura
                plt.savefig(f'avv_{model_name}_sample{idx}.png')

                count += 1  # Aumenta il contatore per il limite di visualizzazione


'''
    ### Analisi di Explainability ###
    explainability_analysis(models, model_names, results_FMN_data, ts, dataset_labels, input_shape)


    ### Analisi della Confidence ###
    confidence_analysis(models, model_names, ts, dataset_labels, results_file_confidence="../extracted_data/data_attack_result_FMN_CONFIDENCE.pkl", num_samples=5)

    print("\n✅ Fine dell'esecuzione!")



'''