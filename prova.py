import torch
import foolbox as fb
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.peval.metrics import CMetricAccuracy
from secml.array import CArray
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.features.normalization import CNormalizerMinMax

# Configurazione dispositivo
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Caricamento e normalizzazione del dataset CIFAR-10
def load_cifar10_data(n_samples=64):
    tr, ts = CDataLoaderCIFAR10().load()
    normalizer = CNormalizerMinMax().fit(tr.X)
    ts.X = normalizer.transform(ts.X)
    ts = ts[:n_samples, :]
    return ts

# Caricamento modelli RobustBench
def load_robustbench_models(model_names, dataset="cifar10", threat_model="Linf"):
    return [load_model(model_name=name, dataset=dataset, threat_model=threat_model).to(device) for name in model_names]

# Calcolo accuratezza su dati puliti
def compute_clean_accuracy(models, ts):
    metric = CMetricAccuracy()

    # Assicurati che i dati abbiano la shape corretta (Batch, Channels, Height, Width)
    ts_X_torch = torch.tensor(ts.X.tondarray()).float().reshape(-1, 3, 32, 32).to(device)
    ts_Y_torch = torch.tensor(ts.Y.tondarray()).long().to(device)

    with torch.no_grad():
        models_preds = [clf(ts_X_torch).argmax(dim=1).cpu().numpy() for clf in models]

    accuracies = [metric.performance_score(y_true=ts.Y, y_pred=CArray(y_pred)) for y_pred in models_preds]

    print("-" * 90)
    for idx, name in enumerate(model_names):
        print(f"Model: {name:<40} - Clean Accuracy: {accuracies[idx] * 100:.2f} %")
    print("-" * 90)
    return accuracies


# Attacco Fast-Minimum-Norm (FMN)
def attack_fmn(model, x_test, y_test, eps=8/255):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.LInfFMNAttack()  # Nome corretto dell'attacco
    perturbations, advs, success = attack(fmodel, x_test.to(device), y_test.to(device), epsilons=[eps])
    return 1 - success.float().mean()

# Attacco PGD
def attack_pgd(model, x_test, y_test, eps=8/255):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    perturbations, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to(device), y_test.to(device), epsilons=[eps])
    return 1 - success.float().mean()

# Esecuzione pipeline
if __name__ == "__main__":
    # Carica dati e modelli
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

    # Accuratezza su dati puliti
    compute_clean_accuracy(models, ts)

    print("\nEvaluating adversarial robustness...")
    for model, name in zip(models, model_names):
        robust_acc_fmn = attack_fmn(model, x_test, y_test)
        robust_acc_pgd = attack_pgd(model, x_test, y_test)
        print(f"Model: {name:<40} - Robust Acc (FMN): {robust_acc_fmn:.1%} - Robust Acc (PGD): {robust_acc_pgd:.1%}")
