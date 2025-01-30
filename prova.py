import os
import pickle
import torch
import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from secml.array import CArray
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.peval.metrics import CMetricAccuracy

# Configurazione dispositivo
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Percorso file risultati
results_file = "results_FMN_vs_PGD.pkl"


# Caricamento e normalizzazione del dataset CIFAR-10
def load_cifar10_data(n_samples=64):
    tr, ts = CDataLoaderCIFAR10().load()
    normalizer = CNormalizerMinMax().fit(tr.X)
    ts.X = normalizer.transform(ts.X)
    return ts[:n_samples, :]


# Caricamento modelli RobustBench
def load_robustbench_models(model_names, dataset="cifar10", threat_model="Linf"):
    return {name: load_model(model_name=name, dataset=dataset, threat_model=threat_model).to(device) for name in model_names}


# Funzione per calcolare l'accuratezza su dati puliti
def compute_clean_accuracy(models, ts):
    metric = CMetricAccuracy()
    ts_X_torch = torch.tensor(ts.X.tondarray()).float().reshape(-1, 3, 32, 32).to(device)
    ts_Y_torch = torch.tensor(ts.Y.tondarray()).long().to(device)

    accuracies = {}
    with torch.no_grad():
        for name, model in models.items():
            preds = model(ts_X_torch).argmax(dim=1).cpu().numpy()
            accuracies[name] = metric.performance_score(y_true=ts.Y, y_pred=CArray(preds))
            print(f"Model: {name:<40} - Clean Accuracy: {accuracies[name] * 100:.2f} %")
    return accuracies


# Attacco Fast-Minimum-Norm (FMN)
def attack_fmn(model, x_test, y_test, eps=8 / 255):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.LInfFMNAttack()
    perturbations, advs, success = attack(fmodel, x_test.to(device), y_test.to(device), epsilons=[eps])

    return advs.cpu().numpy(), perturbations.cpu().numpy(), success.cpu().numpy()




# Attacco PGD
def attack_pgd(model, x_test, y_test, eps=8 / 255):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.LinfPGD()
    perturbations, advs, success = attack(fmodel, x_test.to(device), y_test.to(device), epsilons=[eps])

    return advs.cpu().numpy(), perturbations.cpu().numpy(), success.cpu().numpy()



# Se il file esiste, carica i dati e visualizza le accuratezze
if os.path.exists(results_file):
    print("\nCaricamento risultati esistenti...")
    with open(results_file, "rb") as f:
        results = pickle.load(f)

    models_names = list(results.keys())
    clean_acc = [results[m]['clean'] for m in models_names]
    robust_acc_fmn = [results[m]['fmn'] for m in models_names]
    robust_acc_pgd = [results[m]['pgd'] for m in models_names]

    # Plot dei risultati
    plt.figure(figsize=(10, 5))
    plt.bar(models_names, clean_acc, label='Clean Accuracy', alpha=0.6)
    plt.bar(models_names, robust_acc_fmn, label='Robust Acc (FMN)', alpha=0.6)
    plt.bar(models_names, robust_acc_pgd, label='Robust Acc (PGD)', alpha=0.6)
    plt.ylabel("Accuracy")
    plt.title("Comparison of Clean and Adversarial Accuracy")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

else:
    print("\nNessun file di risultati trovato. Eseguo gli attacchi...")
    ts = load_cifar10_data()
    x_test, y_test = load_cifar10(n_examples=50)
    model_names = [
        "Ding2020MMA",
        "Wong2020Fast",
        "Andriushchenko2020Understanding",
        "Sitawarin2020Improving",
        "Cui2023Decoupled_WRN-28-10"
    ]
    models = load_robustbench_models(model_names)

    # Calcolo accuratezza su dati puliti
    clean_accuracies = compute_clean_accuracy(models, ts)

    # Attacchi adversarial e salvataggio risultati
    # Attacchi adversarial e salvataggio risultati
    results = {}

    for name, model in models.items():
        # Eseguire gli attacchi
        advs_fmn, perturbations_fmn, success_fmn = attack_fmn(model, x_test, y_test)
        advs_pgd, perturbations_pgd, success_pgd = attack_pgd(model, x_test, y_test)

        # Identificare i campioni discordanti
        fmn_only = np.where(success_fmn & ~success_pgd)[0]  # FMN funziona ma PGD no
        pgd_only = np.where(success_pgd & ~success_fmn)[0]  # PGD funziona ma FMN no

        print(f"Model: {name:<40} - Success FMN: {success_fmn.mean():.1%} - Success PGD: {success_pgd.mean():.1%}")
        print(f" - FMN ha successo ma PGD no: {len(fmn_only)} campioni")
        print(f" - PGD ha successo ma FMN no: {len(pgd_only)} campioni")

        # Salvataggio risultati
        results[name] = {
            "clean": clean_accuracies[name],
            "success_fmn": success_fmn.mean(),  # Percentuale di successo FMN
            "success_pgd": success_pgd.mean(),  # Percentuale di successo PGD
            "fmn_only_indices": fmn_only.tolist(),
            "pgd_only_indices": pgd_only.tolist(),
            "advs_fmn": advs_fmn,  # Tutti gli adv FMN
            "advs_pgd": advs_pgd,  # Tutti gli adv PGD
            "perturbations_fmn": perturbations_fmn,  # Tutte le perturbazioni FMN
            "perturbations_pgd": perturbations_pgd,  # Tutte le perturbazioni PGD
        }

    # Salvataggio in un file pickle per analisi successive
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nRisultati salvati in {results_file}")
