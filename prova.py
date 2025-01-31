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







###################################################################################


# Percorso file risultati
results_file = "results_FMN_vs_PGD.pkl"

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

		print('-'*140)
		print(f"Model: {name:<40} - Success FMN: {success_fmn_mean:.1%} - Success PGD: {success_pgd_mean:.1%}")
		print(f" - FMN ha successo con distanza < 8/255: {valid_fmn_success:.1%}")
		print(f" - FMN ha successo ma PGD no: {len(fmn_only)} campioni")
		print(f" - PGD ha successo ma FMN no: {len(pgd_only)} campioni")
		print(
			f"Model: {name} - Min Perturbation: {perturbations_fmn_np.min()} - Mean Perturbation: {perturbations_fmn_np.mean()}- Max Perturbation: {perturbations_fmn_np.max()}")
		print('-'*140)

else:

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


	# Calcolare l'accuratezza pulita PRIMA di eseguire gli attacchi
	clean_accuracies = compute_clean_accuracy(models, x_test, y_test)

	# Esegui gli attacchi e salva i risultati
	results = {}
	for name, model in models.items():
		print('-'*140)
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
			f"Model: {name} - Min Perturbation: {perturbations_fmn_np.min()} - Mean Perturbation: {perturbations_fmn_np.mean()}- Max Perturbation: {perturbations_fmn_np.max()}")


		valid_fmn_success = (perturbations_fmn_np.max(axis=(1, 2, 3)) <= (8 / 255))



		success_pgd_np = success_pgd.cpu().numpy().squeeze()

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
		print(
			f"Model: {name} - Min Perturbation: {perturbations_fmn_np.min()} - Mean Perturbation: {perturbations_fmn_np.mean()}- Max Perturbation: {perturbations_fmn_np.max()}")
		print('-'*140)
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

	print(f"\nRisultati salvati in {results_file}")

# Genera e salva il plot
models_names = list(results.keys())
clean_acc = [results[m]['clean'] for m in models_names]   # L'accuratezza pulita non è calcolata qui
robust_acc_fmn = [results[m]['success_fmn'] for m in models_names]
robust_acc_pgd = [results[m]['success_pgd'] for m in models_names]
plot_results(models_names, clean_acc, robust_acc_fmn, robust_acc_pgd)
################################################################################################################################################################################################################################################


import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from robustbench.data import load_cifar10
from robustbench.utils import load_model

dataset_labels = [
	'airplane', 'automobile', 'bird', 'cat', 'deer',
	'dog', 'frog', 'horse', 'ship', 'truck'
]


def normalize_cifar10(images):
	mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(images.device)
	std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(images.device)
	return (images - mean) / std


def inspect_results(results_file):
	with open(results_file, "rb") as f:
		results = pickle.load(f)

	for model_name, data in results.items():
		print(f"\nModel: {model_name}")
		print(f"Clean Accuracy: {data.get('clean', 'N/A')}")
		print(f"Success Rate FMN: {data.get('success_fmn', 'N/A')}")
		print(f"Success Rate PGD: {data.get('success_pgd', 'N/A')}")
		print(f"Valid FMN Success Samples: {np.sum(data['valid_fmn_success'])}")
		print(f"FMN Only Indices: {data['fmn_only_indices']}")
		print(f"PGD Only Indices: {data['pgd_only_indices']}")
		if 'perturbations_fmn' in data:
			perturbations_fmn = np.array(data['perturbations_fmn'])
			print(
				f"FMN Perturbations - Min: {perturbations_fmn.min()} | Mean: {perturbations_fmn.mean()} | Max: {perturbations_fmn.max()}")


def get_adversarial_labels(model, adversarials):
	model.eval()
	adversarials_tensor = torch.tensor(adversarials, dtype=torch.float32).to(device)  # Already CHW
	adversarials_tensor = normalize_cifar10(adversarials_tensor)  # Normalizza le immagini
	with torch.no_grad():
		adv_preds = model(adversarials_tensor).argmax(dim=1).cpu().numpy()
	return adv_preds


def plot_adversarial_examples(originals, adversarials, perturbations, indices, labels, adv_labels, model_name,
                              num_samples=5):
	fig, axes = plt.subplots(num_samples, 3, figsize=(9, num_samples * 3))
	fig.suptitle(f"{model_name}: Original vs Adversarial (FMN) vs Perturbation", fontsize=14)

	for i, idx in enumerate(indices[:num_samples]):
		orig_img = np.clip(originals[idx].transpose(1, 2, 0), 0, 1)
		adv_img = np.clip(adversarials[idx].transpose(1, 2, 0), 0, 1)
		perturb_img = np.max(np.abs(adversarials[idx] - originals[idx]), axis=0) * 255  # Massimizza tra i tre canali

		original_label = dataset_labels[labels[idx]]
		adversarial_label = dataset_labels[adv_labels[idx]]

		print(f"Sample {idx}: Original Label - {original_label}, Adversarial Label - {adversarial_label}")

		axes[i, 0].imshow(orig_img)
		axes[i, 0].set_title(f"Original: {original_label}")
		axes[i, 0].axis("off")

		axes[i, 1].imshow(adv_img)
		axes[i, 1].set_title(f"Adversarial: {adversarial_label}")
		axes[i, 1].axis("off")

		axes[i, 2].imshow(perturb_img, cmap="inferno")  # Uso scala di grigi per maggiore contrasto
		axes[i, 2].set_title(f"Perturbation")
		axes[i, 2].axis("off")

	plt.tight_layout()
	plt.savefig(f'{model_name}_PLOTPROVA.png')
	plt.show()


# Carica il dataset
x_test, y_test = load_cifar10(n_examples=64)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Caricamento risultati
results_file = "results_FMN_vs_PGD.pkl"
inspect_results(results_file)

with open(results_file, "rb") as f:
	results = pickle.load(f)

# Per ogni modello, esegui il plot per 5 campioni
for model_name, data in results.items():
	valid_indices = np.where(np.array(data["valid_fmn_success"]))[0]

	# Carica il modello RobustBench corrispondente
	model = load_model(model_name, dataset="cifar10", threat_model="Linf").to(device)

	# Seleziona le immagini originali, avversariali e perturbazioni
	originals = x_test.numpy()
	adversarials = np.clip(np.array(data["perturbations_fmn"]) + originals, 0, 1)  # Already CHW
	perturbations = adversarials - originals  # Same format

	# Ottenere le predizioni del modello per le immagini avversariali
	adversarial_labels = get_adversarial_labels(model, adversarials)  # No transpose needed

	# Debug: stampa alcune predizioni
	print(f"Adversarial labels for {model_name}: {adversarial_labels[:5]}")

	# Controllo visivo sulla media dei pixel
	print(f"Mean pixel values - Original: {originals.mean():.4f}, Adversarial: {adversarials.mean():.4f}")

	# Controllo: percentuale di immagini che hanno cambiato etichetta
	changed_labels = (adversarial_labels != y_test.numpy()).sum()
	print(f"Percentage of changed labels: {(changed_labels / len(y_test)) * 100:.2f}%")

	# Mostra le immagini
	plot_adversarial_examples(originals, adversarials, perturbations, valid_indices, y_test.numpy(), adversarial_labels,
	                          model_name)
