import numpy as np
import robustbench
import secml
from secml.adv.attacks.evasion import CAttackEvasionFoolbox
from secml.ml import CClassifierPyTorch
from secml.ml.classifiers.loss import CSoftmax
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.features.normalization import CNormalizerMinMax
from secml.explanation import CExplainerIntegratedGradients
from secml.figure import CFigure
import foolbox as fb
import torch
import pickle
import os
from autoattack import AutoAttack
from secml.ml.peval.metrics import CMetricAccuracy
from secml.array import CArray



print(f"RobustBench version: {robustbench.__name__}")
print(f"SecML version: {secml.__version__}")
print(f"Foolbox version: {fb.__version__}")
print(f"Numpy version: {np.__version__}")

"""## Global Variables
Contains definition of global variables
"""

# Percorsi dei file per salvare i risultati degli attacchi
results_file_AA = "extracted_data/data_autoattack_results.pkl"
results_file_FNM = 'extracted_data/data_attack_result_FNM.pkl'
results_file_confidence = 'extracted_data/data_attack_result_FNM_CONFIDENCE.pkl'

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


import torch
from autoattack import AutoAttack

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




def FNM_attack(samples, labels, model, explainer_class=None, num_classes=10):
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


import os
import pickle


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

                    attack_result = FNM_attack(
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

def explainability_analysis(models, model_names, results_FNM, ts, dataset_labels, input_shape, epsilon=8 / 255):
    """
    Esegue l'analisi di explainability per una lista di modelli attaccati con FNM.

    Parametri:
    - models (list): Lista dei modelli.
    - model_names (list): Nomi dei modelli corrispondenti.
    - results_FNM (list): Risultati dell'attacco FNM.
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
            adv_ds = results_FNM[model_id]['result'].get('adv_ds')
            y_adv = results_FNM[model_id]['result'].get('y_pred_adv')
            attributions = results_FNM[model_id]['result'].get('attributions')

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
    CONFIDENCE_results_FNM = load_results(results_file_confidence)

    if not CONFIDENCE_results_FNM:  # Se il file non esiste o è corrotto
        print(f"⚠️ Il file '{results_file_confidence}' non esiste o è corrotto. Generando nuovi risultati...")
        CONFIDENCE_results_FNM = generate_confidence_results(
            num_samples, models, model_names, dataset_labels, ts
        )
        save_results(results_file_confidence, CONFIDENCE_results_FNM)

    # Itera sui primi `num_samples` campioni
    for sample_id in range(num_samples):
        try:
            print(f"\n🔍 Generazione del plot della Confidence per il Sample n.{sample_id + 1}")

            # Creazione della figura per visualizzare la confidence per ogni modello
            fig = CFigure(width=30, height=4, fontsize=10, linewidth=2)
            label_true = ts.Y[sample_id].item()  # Etichetta vera del campione

            for model_id, model_name in enumerate(model_names):
                try:
                    attack_result = CONFIDENCE_results_FNM[sample_id][model_id]['result']

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

	### Attacco FNM ###
	print("\n⚡ Caricamento o generazione dei risultati dell'attacco FNM...")
	results_FNM = load_results(results_file_FNM)

	if not results_FNM:  # Se il caricamento non ha avuto successo, esegui l'attacco FNM
		print(f"⚠️ Il file '{results_file_FNM}' non esiste o è corrotto. Generando nuovi risultati...")
		results_FNM = [
			{'model_name': name,
			 'result': FNM_attack(ts.X, ts.Y, model, CExplainerIntegratedGradients, len(dataset_labels))}
			for model, name in zip(models, model_names)
		]
		save_results(results_file_FNM, results_FNM)

	# Accuratezza dopo attacco FNM
	print("\n📉 Accuratezza dei modelli sotto attacco FNM:")
	print("-" * 90)
	for idx in range(len(model_names)):
		accuracy = metric.performance_score(
			y_true=ts.Y,
			y_pred=results_FNM[idx]['result']['y_pred_adv']
		)
		print(f"Model name: {model_names[idx]:<40} - Accuracy under FNM attack: {(accuracy * 100):.2f} %")
	print("-" * 90)

	### Attacco AutoAttack (AA) ###
	print("\n⚡ Caricamento o generazione dei risultati dell'attacco AutoAttack...")
	results_AA = load_results(results_file_AA)

	if not results_AA:  # Se il file non esiste, esegui l'attacco AA
		print(f"⚠️ Il file '{results_file_AA}' non esiste o è corrotto. Generando nuovi risultati...")
		results_AA = [
			{'model_name': name,
			 'result': AA_attack(ts.X, ts.Y, model, CExplainerIntegratedGradients, len(dataset_labels))}
			for model, name in zip(models, model_names)
		]
		save_results(results_file_AA, results_AA)

	print(results_AA)
	# Accuratezza dopo attacco AutoAttack
	print("\n📉 Accuratezza dei modelli sotto attacco AutoAttack:")
	print("-" * 90)
	for result in results_AA:
		print(
			f"Model name: {result['model_name']:<40} - Accuracy under AA attack: {(result['accuracy_under_attack'] * 100):.2f} %")
	print("-" * 90)

	### Analisi di Explainability ###
	explainability_analysis(models, model_names, results_FNM, ts, dataset_labels, input_shape)

	### Analisi della Confidence ###
	confidence_analysis(models, model_names, ts, dataset_labels,
	                    results_file_confidence="extracted_data/data_attack_result_FNM_CONFIDENCE.pkl", num_samples=5)

	print("\n✅ Fine dell'esecuzione!")



