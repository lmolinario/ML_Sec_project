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
    attack = fb.attacks.LInfFMNAttack(steps=50)  # Ridurre il numero di passi
    perturbations, advs, success = attack(fmodel, x_test.to(device), y_test.to(device), epsilons=[eps])

    # Se `perturbations` è una lista, convertila in un tensore
    if isinstance(perturbations, list):
        perturbations = torch.stack(perturbations)

    # Clippare le perturbazioni al massimo valore di `eps`
    perturbations = torch.clamp(perturbations, 0, eps)

    print(f"FMN Perturbation Stats - Min: {perturbations.min().item()}, Max: {perturbations.max().item()}, Mean: {perturbations.mean().item()}")
    
    return advs, perturbations, success






# Attacco PGD
def attack_pgd(model, x_test, y_test, eps=8 / 255):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.LinfPGD()
    perturbations, advs, success = attack(fmodel, x_test.to(device), y_test.to(device), epsilons=[eps])

    return advs, perturbations, success



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

        # Converti perturbazioni in array NumPy
        perturbations_fmn_np = perturbations_fmn.cpu().numpy()

        print(f"Model: {name} - Min Perturbation: {perturbations_fmn_np.min()} - Max Perturbation: {perturbations_fmn_np.max()}")
        print(f"Shape perturbations_fmn: {perturbations_fmn_np.shape}")

        # Conta solo i campioni in cui almeno un canale ha perturbazione <= 8/255
        valid_fmn_success = (perturbations_fmn_np.max(axis=(2, 3, 4)) <= (8 / 255)).squeeze()

        # Converti successi PGD in array compatibile
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



        # **Salvataggio risultati**
        results[name] = {
            "clean": clean_accuracies[name],
            "success_fmn": success_fmn_mean,
            "success_pgd": success_pgd_mean,
            "valid_fmn_success": valid_fmn_success.mean(),
            "fmn_only_indices": fmn_only.tolist(),
            "pgd_only_indices": pgd_only.tolist(),
            "advs_fmn": advs_fmn.cpu().numpy() if isinstance(advs_fmn, torch.Tensor) else np.array([a.cpu().numpy() for a in advs_fmn]),
            "advs_pgd": advs_pgd.cpu().numpy() if isinstance(advs_pgd, torch.Tensor) else np.array([a.cpu().numpy() for a in advs_pgd]),
            "perturbations_fmn": perturbations_fmn_np,
            "perturbations_pgd": np.array([p.cpu().numpy() for p in perturbations_pgd]),  # ✅ CORRETTO
        }


    # Salvataggio in un file pickle per analisi successive
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nRisultati salvati in {results_file}")
