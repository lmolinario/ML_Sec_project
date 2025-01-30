import os
import pickle
import torch
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt
from robustbench.data import load_cifar10
from robustbench.utils import load_model

# Configurazione dispositivo
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Percorso file risultati
results_file = "results_FMN_vs_PGD.pkl"


# Funzione per generare e salvare il plot
def plot_results(models_names, clean_acc, robust_acc_fmn, robust_acc_pgd):
	plt.figure(figsize=(10, 5))
	plt.bar(models_names, clean_acc, label='Clean Accuracy', alpha=0.6)
	plt.bar(models_names, robust_acc_fmn, label='Robust Acc (FMN)', alpha=0.6)
	plt.bar(models_names, robust_acc_pgd, label='Robust Acc (PGD)', alpha=0.6)
	plt.ylabel("Accuracy")
	plt.title("Comparison of Clean and Adversarial Accuracy")
	plt.legend()
	plt.xticks(rotation=45)

	# Salvataggio del plot
	plt.savefig("comparison_accuracy.png", dpi=300, bbox_inches='tight')
	plt.close()
	print("\nGrafico delle accuratezze salvato come 'comparison_accuracy.png'.")


# Se il file dei risultati esiste, caricalo e genera il plot
if os.path.exists(results_file):
	print("\nCaricamento risultati esistenti...")
	with open(results_file, "rb") as f:
		results = pickle.load(f)

	# Estrazione dati per il plot
	models_names = list(results.keys())
	clean_acc = [results[m]['clean'] if results[m]['clean'] is not None else 0 for m in models_names]

	robust_acc_fmn = [results[m]['success_fmn'] for m in models_names]
	robust_acc_pgd = [results[m]['success_pgd'] for m in models_names]

	# Genera e salva il plot
	plot_results(models_names, clean_acc, robust_acc_fmn, robust_acc_pgd)

	# Stampa i risultati
	for name, data in results.items():
		success_fmn_mean = data["success_fmn"]
		success_pgd_mean = data["success_pgd"]
		valid_fmn_success = data["valid_fmn_success"]
		fmn_only = data["fmn_only_indices"]
		pgd_only = data["pgd_only_indices"]
		perturbations_fmn_np = np.array(data["perturbations_fmn"])

		print(f"Model: {name:<40} - Success FMN: {success_fmn_mean:.1%} - Success PGD: {success_pgd_mean:.1%}")
		print(f" - FMN ha successo con distanza < 8/255: {valid_fmn_success:.1%}")
		print(f" - FMN ha successo ma PGD no: {len(fmn_only)} campioni")
		print(f" - PGD ha successo ma FMN no: {len(pgd_only)} campioni")
		print(
			f"Model: {name} - Min Perturbation: {perturbations_fmn_np.min()} - Max Perturbation: {perturbations_fmn_np.max()}")
		print(f"Shape perturbations_fmn: {perturbations_fmn_np.shape}")

	# Termina il programma senza eseguire nuovamente gli attacchi
	exit()

# Se il file non esiste, esegui gli attacchi
print("\nNessun file di risultati trovato. Eseguo gli attacchi....")

# Caricamento dataset
x_test, y_test = load_cifar10(n_examples=64)

# Modelli da testare
model_names = [
	"Ding2020MMA",
	"Wong2020Fast",
	"Andriushchenko2020Understanding",
	"Sitawarin2020Improving",
	"Cui2023Decoupled_WRN-28-10"
]

# Caricamento modelli RobustBench
models = {name: load_model(name, dataset="cifar10", threat_model="Linf").to(device) for name in model_names}


# Attacchi
def attack_fmn(model, x_test, y_test, eps=8 / 255):
	fmodel = fb.PyTorchModel(model, bounds=(0, 1))
	attack = fb.attacks.LInfFMNAttack()
	perturbations, advs, success = attack(fmodel, x_test.to(device), y_test.to(device), epsilons=[eps])

	if isinstance(perturbations, list):
		perturbations = torch.stack(perturbations)

	# 🔥 Clippare le perturbazioni per assicurarsi che siano <= eps=8/255
	perturbations = torch.clamp(perturbations, 0, eps)

	print(
		f"FMN Perturbation Stats - Min: {perturbations.min().item()}, Max: {perturbations.max().item()}, Mean: {perturbations.mean().item()}")

	return advs, perturbations, success


def attack_pgd(model, x_test, y_test, eps=8 / 255):
	fmodel = fb.PyTorchModel(model, bounds=(0, 1))
	attack = fb.attacks.LinfPGD()
	perturbations, advs, success = attack(fmodel, x_test.to(device), y_test.to(device), epsilons=[eps])
	return advs, perturbations, success


def compute_clean_accuracy(models, x_test, y_test):
	accuracies = {}
	x_test_torch = x_test.to(device)
	y_test_torch = y_test.to(device)

	with torch.no_grad():
		for name, model in models.items():
			preds = model(x_test_torch).argmax(dim=1).cpu().numpy()
			accuracy = (preds == y_test.cpu().numpy()).mean()
			accuracies[name] = accuracy
			print(f"Model: {name:<40} - Clean Accuracy: {accuracy * 100:.2f}%")

	return accuracies


# Calcolare l'accuratezza pulita PRIMA di eseguire gli attacchi
clean_accuracies = compute_clean_accuracy(models, x_test, y_test)

# Esegui gli attacchi e salva i risultati
results = {}
for name, model in models.items():
	print(f"\nEseguendo attacchi per il modello: {name}")

	advs_fmn, perturbations_fmn, success_fmn = attack_fmn(model, x_test, y_test)
	advs_pgd, perturbations_pgd, success_pgd = attack_pgd(model, x_test, y_test)

	# Converti perturbazioni in array NumPy
	if isinstance(perturbations_fmn, list):
		perturbations_fmn = torch.stack(perturbations_fmn)
	perturbations_fmn_np = perturbations_fmn.squeeze(0).cpu().numpy()

	if isinstance(perturbations_pgd, list):
		perturbations_pgd = torch.stack(perturbations_pgd)
	perturbations_pgd_np = perturbations_pgd.squeeze(0).cpu().numpy()

	print(
		f"Model: {name} - Min Perturbation: {perturbations_fmn_np.min()} - Max Perturbation: {perturbations_fmn_np.max()}")
	print(f"Shape perturbations_fmn: {perturbations_fmn_np.shape}")

	valid_fmn_success = (perturbations_fmn_np.max(axis=(1, 2, 3)) <= (8 / 255))

	print(f"Shape valid_fmn_success: {valid_fmn_success.shape}")  # Deve essere (64,)

	success_pgd_np = success_pgd.cpu().numpy().squeeze()
	print(f"Shape success_pgd_np: {success_pgd_np.shape}")  # Deve essere (64,)

	# Identificare i campioni discordanti
	fmn_only = np.where(valid_fmn_success & ~success_pgd_np)[0]
	pgd_only = np.where(success_pgd_np & ~valid_fmn_success)[0]

	# Calcolo medie corrette
	success_fmn_mean = success_fmn.cpu().numpy().mean()
	success_pgd_mean = success_pgd.cpu().numpy().mean()

	print(f"Model: {name:<40} - Success FMN: {success_fmn_mean:.1%} - Success PGD: {success_pgd_mean:.1%}")
	print(f" - FMN ha successo con distanza < 8/255: {valid_fmn_success.mean():.1%}")
	print(f" - FMN ha successo ma PGD no: {len(fmn_only)} campioni")
	print(f" - PGD ha successo ma FMN no: {len(pgd_only)} campioni")

	# Salvataggio risultati
	results[name] = {
		"clean": clean_accuracies[name],  # ✅ Ora salva la clean accuracy!
		"success_fmn": success_fmn_mean,
		"success_pgd": success_pgd_mean,
		"valid_fmn_success": valid_fmn_success.mean(),
		"fmn_only_indices": fmn_only.tolist(),
		"pgd_only_indices": pgd_only.tolist(),
		"perturbations_fmn": perturbations_fmn_np,
		"perturbations_pgd": perturbations_pgd_np,
	}

# Salvataggio su file
with open(results_file, "wb") as f:
	pickle.dump(results, f)

# Genera e salva il plot
models_names = list(results.keys())
clean_acc = [0 for _ in models_names]  # L'accuratezza pulita non è calcolata qui
robust_acc_fmn = [results[m]['success_fmn'] for m in models_names]
robust_acc_pgd = [results[m]['success_pgd'] for m in models_names]

plot_results(models_names, clean_acc, robust_acc_fmn, robust_acc_pgd)

print(f"\nRisultati salvati in {results_file}")

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

# Caricare i risultati dal file pickle
results_file = "results_FMN_vs_PGD.pkl"

# Caricare i dati salvati
with open(results_file, "rb") as f:
	results = pickle.load(f)

# Numero di immagini da visualizzare
num_samples = 3  # Puoi cambiare questo valore

# Seleziona un modello (scegli uno tra quelli salvati)
model_name = "Andriushchenko2020Understanding"  # Cambia il nome se vuoi un altro modello

# Estrarre i dati necessari dal modello scelto
perturbations_fmn = results[model_name]["perturbations_fmn"][:num_samples]
advs_fmn = results[model_name]["advs_fmn"][:num_samples]
x_test = results[model_name]["advs_fmn"][:num_samples] - results[model_name]["perturbations_fmn"][
                                                         :num_samples]  # Ricostruisce l'originale
y_test = ["cat", "ship", "frog"]  # Placeholder, sostituiscili con le vere etichette


# Funzione per creare un heatmap della perturbazione
def explainability_map(perturbation):
	return np.abs(perturbation).sum(axis=0)  # Somma sui canali per ottenere una heatmap


# Creazione del plot
fig, axes = plt.subplots(num_samples, 4, figsize=(10, num_samples * 3))

for i in range(num_samples):
	# Immagine originale
	axes[i, 0].imshow(np.transpose(x_test[i], (1, 2, 0)))
	axes[i, 0].set_title(f"True: {y_test[i]}")
	axes[i, 0].axis("off")

	# Immagine avversariale
	axes[i, 1].imshow(np.transpose(advs_fmn[i], (1, 2, 0)))
	axes[i, 1].set_title(f"Adv: ???")  # Qui puoi mettere la predizione del modello
	axes[i, 1].axis("off")

	# Perturbazione
	pert = np.transpose(perturbations_fmn[i], (1, 2, 0))
	axes[i, 2].imshow(pert)
	axes[i, 2].set_title("Perturbation")
	axes[i, 2].axis("off")

	# Explainability Map
	exp_map = explainability_map(pert)
	axes[i, 3].imshow(exp_map, cmap="jet")
	axes[i, 3].set_title("Explain")
	axes[i, 3].axis("off")

plt.suptitle(f"Explainability for Model: {model_name}", fontsize=14)
plt.tight_layout()
plt.savefig('Explainability')

