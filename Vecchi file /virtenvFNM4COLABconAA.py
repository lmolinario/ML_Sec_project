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

print(f"RobustBench version: {robustbench.__name__}")
print(f"SecML version: {secml.__version__}")
print(f"Foolbox version: {fb.__version__}")
print(f"Numpy version: {np.__version__}")

"""## Global Variables
Contains definition of global variables
"""

input_shape = (3, 32, 32)

model_names = [
	"Ding2020MMA",
	"Wong2020Fast",
	"Andriushchenko2020Understanding",
	"Sitawarin2020Improving",
	"Cui2023Decoupled_WRN-28-10"
]

n_samples = 64

dataset_labels = [
	'airplane', 'automobile', 'bird', 'cat', 'deer',
	'dog', 'frog', 'horse', 'ship', 'truck'
]

# Numero massimo di subplot per figura
max_subplots = 20  # 5 righe x 4 colonne
n_cols = 4  # Numero fisso di colonne
epsilon = 8 / 255  # Limite di perturbazione

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
			classifier=model, y_target=None, epsilons=epsilon,
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


def load_results(file_path):
	"""Carica i risultati da file, gestendo errori di corruzione o assenza del file."""
	if os.path.exists(file_path):
		try:
			with open(file_path, 'rb') as f:
				results = pickle.load(f)
				if isinstance(results, list):
					return results
				else:
					print(f"⚠️ Attenzione: il formato dei dati caricati da '{file_path}' non è valido.")
		except (pickle.PickleError, EOFError, FileNotFoundError, OSError) as e:
			print(f"⚠️ Errore nel caricamento di '{file_path}': {e}")

	return []  # Ritorna una lista vuota se il caricamento fallisce


def save_results(file_path, data):
	"""Salva i risultati in un file, gestendo errori di scrittura e assicurando la validità del file."""
	try:
		directory = os.path.dirname(file_path)
		if directory:  # Evita errori se il percorso non ha una directory
			os.makedirs(directory, exist_ok=True)

		with open(file_path, 'wb') as f:
			pickle.dump(data, f)
			f.flush()  # Forza la scrittura su disco
		print(f"✅ Risultati salvati in '{file_path}'.")
	except (OSError, pickle.PickleError) as e:
		print(f"⚠️ Errore durante il salvataggio in '{file_path}': {e}")


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
	fsize = 28
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


def generate_confidence_results(num_samples, models, model_names, dataset_labels, ts):
	"""Genera i risultati dell'attacco FMN per un numero limitato di campioni."""
	results = []

	for sample_id in range(num_samples):
		sample = ts.X[sample_id, :].atleast_2d()
		label = CArray(ts.Y[sample_id])
		sample_results = []

		for idx, model in enumerate(models):
			print(f"🔍 Analizzando il campione {sample_id} con il modello \"{model_names[idx]}\"...")
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

		results.append(sample_results)
		print(f"✅ Attacco completato per il campione {sample_id}.")

	return results


if __name__ == "__main__":

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
	ts.X = CArray(ts.X.tondarray().reshape(-1, *input_shape))

	"""## Fast-Minimum-Norm (FMN) attack
    Computes the accuracy of the models, just to confirm that it is working properly.
    """

	# Calcolo delle predizioni e accuratezza dei modelli
	metric = CMetricAccuracy()
	models_preds = [clf.predict(ts.X) for clf in models]
	accuracies = [metric.performance_score(y_true=ts.Y, y_pred=y_pred) for y_pred in models_preds]

	print("-" * 90)
	# Stampa delle accuratezze
	for idx in range(len(model_names)):
		print(f"Model name: {model_names[idx]:<40} - Clean model accuracy: {(accuracies[idx] * 100):.2f} %")
	print("-" * 90)

	"""##Saves or loads  attack data on the disk"""
	for idx, model in enumerate(models):
		if not isinstance(model, CClassifierPyTorch):
			print(f"Errore: Il modello {model_names[idx]} non è un'istanza di CClassifierPyTorch.")
		else:
			print(f"Il modello {model_names[idx]} è caricato correttamente come CClassifierPyTorch.")

	# 📂 Percorso file
	results_file = '../extracted_data/data_attack_result_FNM.pkl'

	# 🔄 Caricamento o generazione dei risultati
	results_FNM = load_results(results_file)

	if not results_FNM:  # Se il caricamento non ha avuto successo, genera i dati
		print(f"⚠️ Il file '{results_file}' non esiste o è corrotto. Generando nuovi risultati...")
		results_FNM = [
			{'model_name': name,
			 'result': FNM_attack(ts.X, ts.Y, model, CExplainerIntegratedGradients, len(dataset_labels))}
			for model, name in zip(models, model_names)
		]

		save_results(results_file, results_FNM)

	# Calcolo delle predizioni e accuratezza dei modelli
	metric = CMetricAccuracy()
	models_preds = [clf.predict(ts.X) for clf in models]
	accuracies = [metric.performance_score(y_true=ts.Y, y_pred=y_pred) for y_pred in models_preds]

	print("-" * 90)
	# Stampa delle accuratezze dopo l'attacco

	for idx in range(len(model_names)):
		accuracy = metric.performance_score(
			y_true=ts.Y,
			y_pred=results_FNM[idx]['result']['y_pred_adv']
		)
		print(f"Model name: {model_names[idx]:<40} - Model accuracy under attack: {(accuracy * 100):.2f} %")
	print("-" * 90)

	#############################################################################################

	import torch
	import pickle
	import os
	from autoattack import AutoAttack
	from secml.ml.peval.metrics import CMetricAccuracy
	from secml.array import CArray

	# Percorso del file per salvare i risultati
	results_file = "../extracted_data/data_autoattack_results.pkl"


	def save_results(file_path, data):
		"""Salva i risultati in un file pickle."""
		try:
			with open(file_path, 'wb') as f:
				pickle.dump(data, f)
			print(f"✅ Risultati salvati in '{file_path}'.")
		except Exception as e:
			print(f"⚠️ Errore nel salvataggio: {e}")


	def load_results(file_path):
		"""Carica i risultati da un file pickle."""
		if os.path.exists(file_path):
			try:
				with open(file_path, 'rb') as f:
					return pickle.load(f)
			except Exception as e:
				print(f"⚠️ Errore nel caricamento: {e}")
		return None  # Se il file non esiste o non è valido


	# Caricare i risultati se esistono già
	results_autoattack = load_results(results_file)

	if results_autoattack is None:  # Se i risultati non esistono, eseguire l'attacco
		print("⚠️ Nessun file di risultati trovato. Eseguo AutoAttack...")

		# Assicurarsi che almeno un modello sia stato caricato
		if not models:
			raise ValueError("Errore: Nessun modello è stato caricato correttamente.")

		# Inizializzazione della metrica di accuratezza
		metric = CMetricAccuracy()

		# Converti il dataset in tensori PyTorch per AutoAttack
		device = 'cuda' if torch.cuda.is_available() else 'cpu'

		x_test_torch = torch.tensor(ts.X.tondarray(), dtype=torch.float32).to(device)
		y_test_torch = torch.tensor(ts.Y.tondarray(), dtype=torch.long).to(device)

		# Assicurarsi che x_test_torch abbia il formato corretto [batch, channels, height, width]
		if len(x_test_torch.shape) == 2:
			x_test_torch = x_test_torch.view(-1, *input_shape)

		# Lista per salvare i risultati dell'attacco AutoAttack per ciascun modello
		results_autoattack = []

		# Esecuzione dell'attacco per ogni modello
		for idx, model in enumerate(models):
			print(f"\n🔍 Esecuzione AutoAttack su: {model_names[idx]}")

			# Estrai il modello PyTorch nativo da CClassifierPyTorch
			pytorch_model = model.model.to(device)
			pytorch_model.eval()

			# Creazione dell'attaccante AutoAttack per il modello attuale
			adversary = AutoAttack(pytorch_model, norm='Linf', eps=8 / 255)
			adversary.apgd.n_restarts = 1

			# Esecuzione dell'attacco con i tensori corretti
			x_adv_torch = adversary.run_standard_evaluation(x_test_torch, y_test_torch)

			# Conversione dei dati avversari da PyTorch Tensor a SecML CArray
			x_adv = CArray(x_adv_torch.cpu().detach().numpy())

			# Predizioni post-attacco
			y_pred_adv = model.predict(x_adv)

			# Calcolo dell'accuratezza post-attacco
			accuracy_under_attack = metric.performance_score(y_true=ts.Y, y_pred=y_pred_adv)

			# Salvataggio dei risultati
			results_autoattack.append({
				'model_name': model_names[idx],
				'x_adv': x_adv,
				'y_pred_adv': y_pred_adv,
				'accuracy_under_attack': accuracy_under_attack
			})

			print(
				f"✅ Attacco completato per {model_names[idx]} - Accuracy under attack: {(accuracy_under_attack * 100):.2f} %")

		# 🔄 Salva i risultati dopo l'esecuzione
		save_results(results_file, results_autoattack)

	else:
		print(f"✅ Risultati caricati da '{results_file}', salto l'esecuzione dell'attacco.")

	# Stampa delle accuratezze prima e dopo l'attacco
	print("\n" + "-" * 90)
	print("📊 Accuratezza dei modelli su dati puliti:")
	for idx, name in enumerate(model_names):
		accuracy_clean = metric.performance_score(y_true=ts.Y, y_pred=models[idx].predict(ts.X))
		print(f"Model name: {name:<40} - Clean accuracy: {(accuracy_clean * 100):.2f} %")
	print("-" * 90)

	print("📉 Accuratezza dei modelli sotto attacco:")
	for result in results_autoattack:
		print(
			f"Model name: {result['model_name']:<40} - Accuracy under attack: {(result['accuracy_under_attack'] * 100):.2f} %")
	print("-" * 90)

	#############################################################################################
	print("Calcolo e plot della Explainability")
	# Itera sui modelli
	for model_id in range(len(models)):
		print(f"\nVisualizzazione per il modello: {model_names[model_id]}")

		adv_ds = results_FNM[model_id]['result']['adv_ds']
		y_adv = results_FNM[model_id]['result']['y_pred_adv']
		attributions = results_FNM[model_id]['result']['attributions']

		# Reshape delle immagini in formato (n_samples, 3, 32, 32)
		adv_images = adv_ds.X.tondarray().reshape(-1, *input_shape)
		original_images = ts.X.tondarray().reshape(-1, *input_shape)

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

	print("Calcolo e plot della CONFIDENCE")
	# 📂 Percorso file
	results_file_confidence = '../extracted_data/data_attack_result_FNM_CONFIDENCE.pkl'
	num_samples_to_process = 5  # Numero di campioni da processare

	# 🔄 Caricamento o generazione dei risultati
	CONFIDENCE_results_FNM = load_results(results_file_confidence)

	if not CONFIDENCE_results_FNM:  # Se il caricamento non ha avuto successo, genera i dati
		print(f"⚠️ Il file '{results_file_confidence}' non esiste o è corrotto. Generando nuovi risultati...")
		CONFIDENCE_results_FNM = generate_confidence_results(
			num_samples_to_process, models, model_names, dataset_labels, ts
		)
		save_results(results_file_confidence, CONFIDENCE_results_FNM)

	# Creazione della figura per i primi 5 campioni
	for sample_id in range(num_samples_to_process):
		print(f"Calcolo e plot della CONFIDENCE : Sample n.{sample_id}")
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

			fig.sp.plot(itrs, scores[:, label_true], linestyle='--', c='green')
			fig.sp.plot(itrs, scores[:, label_adv], c='red')
			fig.sp.xlim(top=25, bottom=0)

			fig.sp.title(f"Confidence Sample {sample_id + 1} - Model: {model_id + 1}")
			fig.sp.legend(['Confidence True Class', 'Confidence Adv. Class'])

		fig.tight_layout()
		fig.savefig(f"results/Confidence_Sample_{sample_id + 1}.jpg")
