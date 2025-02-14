import os  # Module for interacting with the operating system
import pickle  # Module for saving and loading objects

import matplotlib.pyplot as plt  # Matplotlib for plotting
import numpy as np  # NumPy for numerical computations

import robustbench  # Library for loading pre-trained robust models

from secml.array import CArray  # SecML's custom array class
from secml.data.loader import CDataLoaderCIFAR10  # CIFAR-10 dataset loader
from secml.explanation import CExplainerIntegratedGradients  # Explainability module using Integrated Gradients
from secml.figure import CFigure  # Module for visualization and plotting
from secml.ml import CClassifierPyTorch  # Wrapper for PyTorch classifiers in SecML
from secml.ml.classifiers.loss import CSoftmax  # Softmax function for classification
from secml.ml.features.normalization import CNormalizerMinMax  # Min-Max normalization utility
from secml.ml.peval.metrics import CMetricAccuracy  # Accuracy metric computation
from attack.AttackStrategy import AttackContext,FMNAttackStrategy,AutoAttackStrategy
import misc.logo # project logo

"""## Global Variables
Contains definition of global variables
"""

# File paths for saving attack results
results_file_AA = "extracted_data/data_attack_result_AA.pkl"
results_file_FMN = 'extracted_data/data_attack_result_FMN.pkl'
results_file_confidence = 'extracted_data/data_attack_result_FMN_CONFIDENCE.pkl'

# Input shape (channels, height, width) for deep learning models
input_shape = (3, 32, 32)

# List of selected models for analysis from RobustBench
model_names = [
    "Ding2020MMA",                    # MMA model by Ding (2020)
    "Wong2020Fast",                   # Fast model by Wong (2020)
    "Andriushchenko2020Understanding", # Model by Andriushchenko (2020)
    "Sitawarin2020Improving",         # Improved model by Sitawarin (2020)
    "Cui2023Decoupled_WRN-28-10"      # Wide ResNet 28-10 model by Cui (2023)
]

# Number of samples to use in testing
n_samples = 64

# Class labels of the CIFAR-10 dataset
dataset_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Visualization configuration
max_subplots = 20  # Maximum number of subplots per figure (5 rows x 4 columns)
n_cols = 4         # Fixed number of columns in plots


def load_model(model_name):
    """
    Loads a pre-trained model from RobustBench and wraps it in CClassifierPyTorch
    for integration with SecML.

    Parameters:
    - model_name (str): Name of the model to load.

    Returns:
    - CClassifierPyTorch: Model wrapped in a classifier compatible with SecML.
    - None: If the model loading fails.
    """
    try:
        # Load the model from RobustBench (CIFAR-10 dataset, Linf attack)
        model = robustbench.utils.load_model(
            model_name=model_name, dataset='cifar10', threat_model='Linf'
        )

        # Wraps the model in a PyTorch classifier compatible with SecML
        return CClassifierPyTorch(
            model=model,
            input_shape=input_shape,  # Input shape: (3, 32, 32)
            pretrained=True,  # Use pre-trained weights
            pretrained_classes=CArray(range(10)),  # Output classes (0-9 for CIFAR-10)
            preprocess=None  # No additional preprocessing
        )

    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None



def load_results(file_path):
    """
    Loads results from a pickle file, handling possible file corruption or absence.

    Parameters:
    - file_path (str): Path to the file containing saved data.

    Returns:
    - list: Loaded results if the file is valid; otherwise, an empty list.
    """
    if os.path.exists(file_path):  # Check if the file exists
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)

                # Verify that the data format is correct
                if isinstance(results, list):
                    return results
                else:
                    print(f"Warning: The data format loaded from '{file_path}' is not valid.")

        except (pickle.PickleError, EOFError, FileNotFoundError, OSError) as e:
            print(f"Error loading '{file_path}': {e}")

    return []  # Return an empty list if loading fails


def save_results(file_path, data):
    """
    Saves results to a pickle file, handling write errors and ensuring file validity.

    Parameters:
    - file_path (str): Path to the file where data will be saved.
    - data (list/dict): Data to be saved in the file.

    Returns:
    - None
    """
    try:
        # Create the directory if it does not exist
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Save the data using Pickle
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            f.flush()  # Force writing to disk

        print(f"✅ Results successfully saved in '{file_path}'.")

    except (OSError, pickle.PickleError) as e:
        print(f"Error saving to '{file_path}': {e}")


def convert_image(image):
    """
    Converts an image from CArray or NumPy format to (H, W, C).
    """
    try:
        # If the image is a CArray, convert it to a NumPy array
        if hasattr(image, "tondarray"):
            image = image.tondarray()

        # If the image is a 1D array of size 3072, reshape it
        if image.shape == (1, 3072) or image.shape == (3072,):
            image = image.reshape(3, 32, 32)

        # Check the dimensions again
        if image.shape != input_shape:
            raise ValueError(f"Incorrect dimensions: expected {input_shape}, found {image.shape}")

        # Convert the image from (C, H, W) to (H, W, C)
        return image.transpose(1, 2, 0)

    except Exception as e:
        print(f"Error in image conversion: {e}")
        return None  # Return None in case of an error



def show_image(fig, local_idx, img, img_adv, expl, label, pred):
    """
    Displays a sequence of four images in a figure: original, adversarial, perturbation, and explanation.

    Parameters:
    - fig (CFigure): SecML figure for visualization.
    - local_idx (int): Local index for the subplot.
    - img (CArray/ndarray): Original image.
    - img_adv (CArray/ndarray): Adversarial image generated by the attack.
    - expl (CArray/ndarray): Model explanation map.
    - label (str): Original class label.
    - pred (str): Predicted class label after the attack.

    Returns:
    - None: The function modifies the figure in-place.
    """

    fsize = 28  # Image title font size

    try:
        # Compute normalized perturbation between 0 and 1
        diff_img = img_adv - img
        diff_img -= diff_img.min()
        diff_img /= max(diff_img.max(), 1e-6)  # Prevents division by zero

        # Original image
        fig.subplot(3, 4, local_idx + 1)  # `local_idx` starts from 0
        fig.sp.imshow(convert_image(img))
        fig.sp.title(f"True: {label}", fontsize=fsize)
        fig.sp.xticks([])
        fig.sp.yticks([])

        # Adversarial image
        fig.subplot(3, 4, local_idx + 2)
        fig.sp.imshow(convert_image(img_adv))
        fig.sp.title(f'Adv: {pred}', fontsize=fsize)
        fig.sp.xticks([])
        fig.sp.yticks([])

        # Perturbation
        fig.subplot(3, 4, local_idx + 3)
        fig.sp.imshow(convert_image(diff_img))
        fig.sp.title('Perturbation', fontsize=fsize)
        fig.sp.xticks([])
        fig.sp.yticks([])

        # Explanation (normalized for better visualization)
        expl = convert_image(expl)
        r, g, b = np.fabs(expl[:, :, 0]), np.fabs(expl[:, :, 1]), np.fabs(expl[:, :, 2])
        expl = np.maximum(np.maximum(r, g), b)

        fig.subplot(3, 4, local_idx + 4)
        fig.sp.imshow(expl, cmap='seismic')
        fig.sp.title('Explain', fontsize=fsize)
        fig.sp.xticks([])
        fig.sp.yticks([])

    except Exception as e:
        print(f"Error during image visualization: {e}")



def generate_confidence_results(num_samples, models, model_names, dataset_labels, ts):
    """
    Generates FMN attack results for a limited number of samples.

    Parameters:
    - num_samples (int): Number of samples to process.
    - models (list): List of CClassifierPyTorch models to attack.
    - model_names (list): Corresponding model names.
    - dataset_labels (list): Class labels of the dataset.
    - ts (CDataset): Test dataset containing images and labels.

    Returns:
    - list: A list containing attack results for each model and sample.
    """

    results = []  # List to store the results

    try:
        for sample_id in range(num_samples):
            sample = ts.X[sample_id, :].atleast_2d()  # Extract the sample as a 2D matrix
            label = CArray(ts.Y[sample_id])  # Label of the current sample
            sample_results = []  # Attack results for each model

            # Iterate over each model to execute the FMN attack
            for idx, model in enumerate(models):
                try:
                    print(f"Analyzing sample {sample_id} with model \"{model_names[idx]}\"...")

                    attack_context = AttackContext(FMNAttackStrategy())  # Seleziona la strategia FMN
                    attack_result = attack_context.execute_attack(
                        samples=sample,
                        labels=label,
                        model=model,
                        model_name=model_names[idx],
                        explainer_class=CExplainerIntegratedGradients,
                        num_classes=len(dataset_labels)
                    )

                    # Save the results for the current model
                    sample_results.append({
                        'model_name': model_names[idx],
                        'sample_id': sample_id,
                        'result': attack_result
                    })

                except Exception as e:
                    print(f"Error during attack on model \"{model_names[idx]}\" for sample {sample_id}: {e}")

            results.append(sample_results)  # Add the sample's results to the main list
            print(f"Attack completed for sample {sample_id}.")

    except Exception as e:
        print(f"General error in generating FMN attack results: {e}")

    return results  # Return the list of results

def explainability_analysis(models, model_names, results_FMN, ts, dataset_labels, input_shape, epsilon=8 / 255):
    """
    Performs explainability analysis for a list of models attacked with FMN.

    Parameters:
    - models (list): List of models.
    - model_names (list): Corresponding model names.
    - results_FMN (list): FMN attack results.
    - ts (CDataset): Original dataset.
    - dataset_labels (list): Dataset class labels.
    - input_shape (tuple): Input shape of images (C, H, W).
    - epsilon (float, optional): Threshold for sample selection (default 8/255).

    Returns:
    - None: Saves explainability results as images.
    """

    print("\n'nStarting Explainability Analysis\n")

    try:
        for model_id, model_name in enumerate(model_names):
            print("-" * 90)
            print(f"Analyzing model: {model_name}")

            # Extract attack-related data
            adv_ds = results_FMN[model_id]['result'].get('adv_ds')
            y_adv = results_FMN[model_id]['result'].get('y_pred_adv')
            attributions = results_FMN[model_id]['result'].get('attributions')

            if adv_ds is None or y_adv is None or attributions is None:
                print(f"Missing data for model {model_name}. Skipping to the next one.")
                continue

            # Reshape images into the correct format (batch, C, H, W)
            adv_images = adv_ds.X.tondarray().reshape(-1, *input_shape)
            original_images = ts.X.tondarray().reshape(-1, *input_shape)

            # Compute L∞ distance between original and adversarial images
            distances = np.abs(adv_images - original_images).max(axis=(1, 2, 3))

            # Select valid samples: successful attack and perturbation < epsilon
            selected_indices = [
                idx for idx in range(ts.X.shape[0])
                if (distances[idx] < epsilon and y_adv[idx] != ts.Y[idx])
            ]

            print(f"Valid samples for model {model_name}: {len(selected_indices)}")

            valid_indices = []
            for idx in selected_indices:
                img = original_images[idx]
                img_adv = adv_images[idx]
                diff_img = img_adv - img

                # Avoid division by zero and check if the perturbation is significant
                if diff_img.max() > 1e-6:
                    valid_indices.append(idx)

                if len(valid_indices) == 3:  # Select the first 3 valid samples
                    break

            print(f"Selected {len(valid_indices)} samples for visualization.")

            if valid_indices:
                # Set up the figure for visualization
                n_rows = len(valid_indices)
                fig = CFigure(height=n_rows * 6, width=18)
                fig.title(f"Explainability for Model: {model_name}", fontsize=32)

                for ydx, idx in enumerate(valid_indices):
                    img = original_images[idx]
                    img_adv = adv_images[idx]
                    expl = attributions[idx][y_adv[idx].item(), :]

                    # Normalize perturbation for better visualization
                    diff_img = img_adv - img
                    diff_img /= diff_img.max()

                    local_idx = ydx * 4  # Subplot position

                    # Display images in the figure
                    show_image(
                        fig,
                        local_idx,
                        img,
                        img_adv,
                        expl,
                        dataset_labels[ts.Y[idx].item()],
                        dataset_labels[y_adv[idx].item()]
                    )

                # Finalize and save the figure
                fig.tight_layout(rect=[0, 0.003, 1, 0.94])
                fig.savefig(f"results/Explainability_model_{model_name}.jpg")
                print(f"Explainability image saved for model {model_name}.")
                print("-" * 90)

    except Exception as e:
        print(f"Error during explainability analysis: {e}")

def confidence_analysis(models, model_names, ts, dataset_labels, results_file_confidence, num_samples=5):
    """
    Computes and generates confidence plots for the first N samples.

    Parameters:
    - models (list): List of models.
    - model_names (list): Corresponding model names.
    - ts (CDataset): Original dataset.
    - dataset_labels (list): Class labels of the dataset.
    - results_file_confidence (str): File path to save/load the results.
    - num_samples (int, optional): Number of samples to analyze (default: 5).

    Returns:
    - None: Saves the confidence plots.
    """

    print("\nConfidence analysis for attacked samples...")

    # Load or generate results
    confidence_results_FMN = load_results(results_file_confidence)

    if not confidence_results_FMN:  # If the file does not exist or is corrupted
        print(f"The file '{results_file_confidence}' does not exist or is corrupted. Generating new results...")
        confidence_results_FMN = generate_confidence_results(
            num_samples, models, model_names, dataset_labels, ts
        )
        save_results(results_file_confidence, confidence_results_FMN)

    # Iterate over the first `num_samples` samples
    for sample_id in range(num_samples):
        try:
            print(f"\nGenerating Confidence plot for Sample No. {sample_id + 1}")

            # Create a figure to visualize confidence for each model
            fig = CFigure(width=30, height=4, fontsize=10, linewidth=2)
            label_true = ts.Y[sample_id].item()  # True label of the sample

            for model_id, model_name in enumerate(model_names):
                try:
                    attack_result = confidence_results_FMN[sample_id][model_id]['result']

                    # Extract the sequence of adversarial images generated during the attack
                    x_seq = attack_result['x_seq']
                    n_iter = x_seq.shape[0]
                    itrs = CArray.arange(n_iter)  # Assign iterations for the X-axis

                    # Compute confidence scores for the true and adversarial classes
                    scores = models[model_id].predict(x_seq, return_decision_function=True)[1]
                    scores = CSoftmax().softmax(scores)  # Normalize scores with softmax

                    # Determine the adversarial class (predicted after the attack)
                    label_adv = attack_result['y_pred_adv'][0].item()

                    # Create a subplot for each model
                    fig.subplot(1, len(models), model_id + 1)

                    # Label for the first subplot
                    if model_id == 0:
                        fig.sp.ylabel('Confidence')

                    fig.sp.xlabel('Iteration')

                    # Plot confidence for the true and adversarial class
                    fig.sp.plot(itrs, scores[:, label_true], linestyle='--', c='green')  # True class
                    fig.sp.plot(itrs, scores[:, label_adv], c='red')  # Adversarial class
                    fig.sp.xlim(top=25, bottom=0)

                    # Subplot title with classification info
                    fig.sp.title(f"Sample {sample_id + 1} - Model: {model_name}\n"
                                 f"True Label: {dataset_labels[label_true]} ({label_true})\n"
                                 f"Adv Label: {dataset_labels[label_adv]} ({label_adv})")

                    fig.sp.legend(['Confidence True Class', 'Confidence Adv. Class'])

                except Exception as e:
                    print(f"Error processing confidence for model {model_name}, Sample {sample_id + 1}: {e}")

            print(f"Generation completed for Sample No. {sample_id + 1}")

            # Save the plot
            fig.tight_layout()
            output_path = f"results/Confidence_Sample_{sample_id + 1}.jpg"
            fig.savefig(output_path)
            print(f"Image saved at {output_path}")

        except Exception as e:
            print(f"General error in generating confidence for Sample {sample_id + 1}: {e}")



def plot_comparison(original, adv_AA, adv_FMN, title, sample_idx, dataset_labels, y_true, y_pred_AA, y_pred_FMN, conf_AA, conf_FMN):
    """
    Displays original images, images attacked by AutoAttack and FMN,
    and the generated perturbations. Additionally, saves cases where
    FMN succeeds, but AutoAttack does not.
    """

    # Reshape images into (H, W, C) format
    original = original.reshape(3, 32, 32).transpose(1, 2, 0)
    adv_AA = adv_AA.reshape(3, 32, 32).transpose(1, 2, 0)
    adv_FMN = adv_FMN.reshape(3, 32, 32).transpose(1, 2, 0)

    diff_AA = np.abs(adv_AA - original)  # AutoAttack perturbation
    diff_FMN = np.abs(adv_FMN - original)  # FMN perturbation

    # Get the class name of the original image and the predicted labels after attacks
    label_original = dataset_labels[y_true.Y[sample_idx].item()]
    label_AA = dataset_labels[y_pred_AA[sample_idx].item()]
    label_FMN = dataset_labels[y_pred_FMN[sample_idx].item()]

    # If confidence is "N/A", set a default value of 0.0
    conf_AA = conf_AA.item() if conf_AA != "N/A" else 0.0
    conf_FMN = conf_FMN.item() if conf_FMN != "N/A" else 0.0

    # Check if the attacks were successful
    fmn_success = label_FMN != label_original
    aa_success = label_AA != label_original

    print(f"Sample {sample_idx}: FMN Success = {fmn_success}, AA Success = {aa_success}")
    print(f"   Confidence AA: {conf_AA:.4f}, Confidence FMN: {conf_FMN:.4f}")

    # Compute L∞ perturbations
    l_inf_AA = np.max(diff_AA)
    l_inf_FMN = np.max(diff_FMN) if fmn_success else 0.0

    # Function to plot and save the results
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

        # Save the plot
        os.makedirs("results", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        plt.savefig(f"results/{filename}_{safe_title}_Sample_{sample_idx}.png")
        plt.close(fig)  # Close the figure to save memory

    # Case 1: AA succeeds, FMN fails
    if aa_success and not fmn_success:
        save_plot("AA Success - FMN Fail", "AA_Success_FMN_Fail")

    # Case 2: Both succeed
    if aa_success and fmn_success:
        save_plot("AA Success - FMN Success", "AA_Success_FMN_Success")

    # Case 3: FMN succeeds, AA fails
    if fmn_success and not aa_success:
        print(f"Special case: FMN succeeds while AutoAttack fails (Sample {sample_idx})")
        save_plot("FMN Success - AA Fail", "FMN_Success_AA_Fail")



def analyze_discrepant_samples(mismatched_samples, model_names, confidence_AA, confidence_FMN, y_pred_AA, y_pred_FMN):
    """
    Analyzes the samples where AutoAttack and FMN produce different results and prints relevant information.

    Args:
        mismatched_samples (dict): Dictionary containing discrepant samples for each model.
        model_names (list): List of model names.
        confidence_AA (dict): Dictionary of confidence scores for AutoAttack.
        confidence_FMN (dict): Dictionary of confidence scores for FMN.
        y_pred_AA (dict): Model predictions under AutoAttack.
        y_pred_FMN (dict): Model predictions under FMN.
    """
    model_name_to_idx = {name: idx for idx, name in enumerate(model_names)}

    Confidence_AA_4_sample = {}
    Confidence_FMN_4_sample = {}

    for model_name, indices in mismatched_samples.items():
        model_idx = model_name_to_idx[model_name]

        print(f"\nAnalysis of discrepant samples for model {model_name}")
        print("-" * 90)
        Confidence_AA_4_sample[model_name] = {}
        Confidence_FMN_4_sample[model_name] = {}

        for sample_data in indices:
            idx = sample_data["sample"]
            conf_AA = confidence_AA[model_idx][idx, y_pred_AA[idx]] if confidence_AA[
                                                                           model_idx] is not None else "N/A"
            conf_FMN = confidence_FMN[idx, y_pred_FMN[idx]]

            label_original = sample_data["true_label"]
            label_AA = sample_data["adv_label_AA"]
            label_FMN = sample_data["adv_label_FMN"]

            print(f"- Sample {idx}: \n  Confidence AA={conf_AA.item()}, Confidence FMN={conf_FMN.item()}")
            print(
                f"  True label: '{label_original}'\n  Adversarial Label FNM: '{label_FMN}'\n  Adversarial Label AA: '{label_AA}' "
            )
            print("  Possible explanations:")

            if label_AA == label_original and label_FMN == label_original:
                print("  * Neither attack was effective: the model may be very robust.")
            elif label_AA != label_original and label_FMN == label_original:
                print("  * AutoAttack successfully modified the class, while FMN failed.")
            elif label_FMN != label_original and label_AA == label_original:
                print("  * FMN successfully modified the class, while AutoAttack failed.")
            elif label_AA != label_FMN:
                print("  * Both attacks changed the prediction, but in different directions.")

            if conf_AA != "N/A" and conf_AA > conf_FMN:
                print("  * AutoAttack generated a more confident prediction compared to FMN.")
            elif conf_FMN > conf_AA:
                print("  * FMN generated a more confident prediction compared to AutoAttack.")
            else:
                print(
                    "  * Confidence scores are similar, the model may be resistant to both attacks.")

            Confidence_AA_4_sample[model_name][idx] = conf_AA
            Confidence_FMN_4_sample[model_name][idx] = conf_FMN
            print("-" * 90)
            print()

    return Confidence_AA_4_sample, Confidence_FMN_4_sample

def plot_perturbations(models, model_name_to_idx, dataset_labels, ts, results_AA_data, results_FMN_data,
                       Perturbation_Linf, Confidence_AA_4_sample, Confidence_FMN_4_sample, input_shape,
                       max_samples=3):
    """
    Generates and saves perturbation plots for models, comparing AA and FMN attacks.
    """
    print("-" * 90)
    for model_name, indices in Perturbation_Linf.items():
        model_idx = model_name_to_idx[model_name]  # Model index
        print(f"\n📊 Visualization for model: {model_name}\n")
        print("-" * 90)
        count = 0  # Counter to limit the number of samples

        for idx in list(indices.keys()):
            if count >= max_samples:
                break

            # Load original and adversarial images
            x_orig = ts.X[idx, :].tondarray().squeeze().reshape(input_shape)
            x_adv_AA = results_AA_data[model_idx]['x_adv'][idx, :].tondarray().squeeze().reshape(input_shape)
            x_adv_FMN = results_FMN_data[model_idx]['result']['adv_ds'].X[idx, :].tondarray().squeeze().reshape(
                input_shape)

            # Compute L∞ perturbation
            linf_AA = float(Perturbation_Linf[model_name][idx]['AA'])
            linf_FMN = float(Perturbation_Linf[model_name][idx]['FMN'])

            # Retrieve confidence scores
            conf_AA = float(Confidence_AA_4_sample[model_name][idx].item()) if idx in Confidence_AA_4_sample[
                model_name] else None
            conf_FMN = float(Confidence_FMN_4_sample[model_name][idx].item()) if idx in Confidence_FMN_4_sample[
                model_name] else None

            # Compute perturbations
            perturbation_AA = np.abs(x_adv_AA - x_orig)
            perturbation_FMN = np.abs(x_adv_FMN - x_orig)

            # Normalize perturbations for better visualization
            perturbation_AA = perturbation_AA / np.max(perturbation_AA) if np.max(
                perturbation_AA) > 0 else np.zeros_like(perturbation_AA)
            perturbation_FMN = perturbation_FMN / np.max(perturbation_FMN) if np.max(
                perturbation_FMN) > 0 else np.zeros_like(perturbation_FMN)

            # Retrieve class labels
            label_original = dataset_labels[ts.Y[idx].item()]
            label_AA = dataset_labels[results_AA_data[model_idx]['y_pred_adv'][idx].item()]
            label_FMN = dataset_labels[results_FMN_data[model_idx]['result']['y_pred_adv'][idx].item()]

            # Check attack success
            attack_AA_success = label_AA != label_original
            attack_FMN_success = label_FMN != label_original

            # **Create Integrated Gradients explainer**
            explainer = CExplainerIntegratedGradients(models[model_idx])  # Correctly initialized now

            # **Compute explanations for adversarial images (Fixing the issue)**
            explain_AA = explainer.explain(CArray(x_adv_AA[None, :]),
                                           CArray([results_AA_data[model_idx]['y_pred_adv'][idx].item()]))
            explain_FMN = explainer.explain(CArray(x_adv_FMN[None, :]),
                                            CArray([results_FMN_data[model_idx]['result']['y_pred_adv'][idx].item()]))

            # Normalize explanations for visualization
            explain_AA = explain_AA.tondarray().squeeze()  # Removes extra dimensions
            explain_FMN = explain_FMN.tondarray().squeeze()

            # Reshape explanations to correct dimensions
            explain_AA = explain_AA.reshape(input_shape)  # input_shape = (3, 32, 32)
            explain_FMN = explain_FMN.reshape(input_shape)

            # If one attack succeeds and the other fails, plot the sample
            if attack_AA_success != attack_FMN_success:
                fig, axes = plt.subplots(3, 3, figsize=(15, 12))

                # **Row 1: Original and adversarial images**
                axes[0, 0].imshow(x_orig.transpose(1, 2, 0))
                axes[0, 0].set_title(f"Original\nClass: {label_original}")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(x_adv_AA.transpose(1, 2, 0))
                axes[0, 1].set_title(f"Adversarial AA ({label_AA})\nConf: {conf_AA:.4f}")
                axes[0, 1].axis("off")

                axes[0, 2].imshow(x_adv_FMN.transpose(1, 2, 0))
                axes[0, 2].set_title(f"Adversarial FMN ({label_FMN})\nConf: {conf_FMN:.4f}")
                axes[0, 2].axis("off")

                # **Row 2: Perturbations**
                axes[1, 0].axis("off")  # Empty slot for better alignment

                axes[1, 1].imshow(perturbation_AA.transpose(1, 2, 0), cmap="inferno")
                axes[1, 1].set_title(f"Perturbation AA\nL∞: {linf_AA:.4f}")
                axes[1, 1].axis("off")

                axes[1, 2].imshow(perturbation_FMN.transpose(1, 2, 0), cmap="inferno")
                axes[1, 2].set_title(f"Perturbation FMN\nL∞: {linf_FMN:.4f}")
                axes[1, 2].axis("off")

                # **Row 3: Explainability (Integrated Gradients)**
                axes[2, 0].axis("off")  # Empty slot for better alignment

                axes[2, 1].imshow(explain_AA.transpose(1, 2, 0), cmap="coolwarm")
                axes[2, 1].set_title(f"Explain AA ({label_AA})")
                axes[2, 1].axis("off")

                axes[2, 2].imshow(explain_FMN.transpose(1, 2, 0), cmap="coolwarm")
                axes[2, 2].set_title(f"Explain FMN ({label_FMN})")
                axes[2, 2].axis("off")

                plt.suptitle(f"Attack Comparison - {model_name} - Sample {idx}")

                # Save the figure
                output_path = f"results/Adv_{model_name}_sample{idx}.png"
                fig.savefig(output_path)
                print(f"Image saved at {output_path}")
                print("-" * 90)
                count += 1  # Increase counter for display limit

if __name__ == "__main__":
    # Loading models from RobustBench
    models = [load_model(name) for name in model_names if load_model(name) is not None]

    print("\nLoading CIFAR-10 dataset...")
    # Loading the CIFAR-10 dataset
    tr, ts = CDataLoaderCIFAR10().load()

    # Normalizing the dataset with a backup of the original
    normalizer = CNormalizerMinMax().fit(tr.X)
    ts_original = ts.deepcopy()  # Backup before normalization
    ts.X = normalizer.transform(ts.X)

    # Reduce to 64 samples and adjust the image shape
    ts = ts[:n_samples, :]
    ts.X = CArray(ts.X.tondarray().reshape(-1, *input_shape))

    print("\nComputing model accuracy on clean data...")
    # Compute initial model accuracy
    metric = CMetricAccuracy()
    models_preds = [clf.predict(ts.X) for clf in models]
    accuracies = [metric.performance_score(y_true=ts.Y, y_pred=y_pred) for y_pred in models_preds]

    print("\nVerifying model loading:")
    print("-" * 90)
    for idx, model in enumerate(models):
        if not isinstance(model, CClassifierPyTorch):
            print(f"Error: Model {model_names[idx]} is not an instance of CClassifierPyTorch.")
        else:
            print(f"Model {model_names[idx]} loaded successfully.")
    print("-" * 90)

    # Print initial model accuracy
    print("\nModel accuracy on clean data:")
    print("-" * 90)
    for idx in range(len(model_names)):
        print(f"Model name: {model_names[idx]:<40} - Clean model accuracy: {(accuracies[idx] * 100):.2f} %")
    print("-" * 90)

    ### **Executing FMN Attack with Strategy Pattern** ###
    print("\nLoading or generating FMN attack results...")
    results_FMN_data = load_results(results_file_FMN)

    if not results_FMN_data:
        print(f"The file '{results_file_FMN}' does not exist or is corrupted. Generating new results...")

        attack_context = AttackContext(FMNAttackStrategy())  # Select FMN strategy
        results_FMN_data = [
            {'model_name': name,
             'result': attack_context.execute_attack(ts.X, ts.Y, model, name, CExplainerIntegratedGradients,
                                                     len(dataset_labels))}
            for model, name in zip(models, model_names)
        ]
        save_results(results_file_FMN, results_FMN_data)

    # Accuracy after FMN Attack
    print("\nModel accuracy under FMN attack:")
    print("-" * 90)
    for idx in range(len(model_names)):
        accuracy = metric.performance_score(
            y_true=ts.Y,
            y_pred=results_FMN_data[idx]['result']['y_pred_adv']
        )
        print(f"Model name: {model_names[idx]:<40} - Accuracy under FMN attack: {(accuracy * 100):.2f} %")
    print("-" * 90)

    ### **Executing AutoAttack with Strategy Pattern** ###
    print("\nLoading or generating AutoAttack results...")
    results_AA_data = load_results(results_file_AA)

    if not results_AA_data:
        print(f"The file '{results_file_AA}' does not exist or is corrupted. Generating new results...")

        attack_context.set_strategy(AutoAttackStrategy())  # Switch strategy to AutoAttack
        results_AA_data = [
            {'model_name': name,
             'result': attack_context.execute_attack(ts.X, ts.Y, model, name, CExplainerIntegratedGradients,
                                                     len(dataset_labels))}
            for model, name in zip(models, model_names)
        ]
        save_results(results_file_AA, results_AA_data)

    # Accuracy after AutoAttack
    print("\nModel accuracy under AutoAttack:")
    print("-" * 90)
    for result in results_AA_data:
        print(
            f"Model name: {result['model_name']:<40} - Accuracy under AA attack: {(result['accuracy_under_attack'] * 100):.2f} %")
    print("-" * 90)

    # Compute confidence for AutoAttack
    confidence_AA = []  # Initialize as an empty list

    print('\nStarting confidence calculation...\n')
    print("-" * 90)
    for model_idx, model in enumerate(models):
        print(f"Computing confidence for model: {model_names[model_idx]}")

        x_adv_AA = results_AA_data[model_idx]['x_adv']  # Adversarial images from AutoAttack
        scores_AA = model.predict(x_adv_AA, return_decision_function=True)[1]  # Get logits

        # Verify if the model generated valid outputs
        if scores_AA is None or scores_AA.shape[0] == 0:
            print(f"Error: Model {model_names[model_idx]} did not generate valid predictions!")
            confidence_AA.append(None)  # Avoid iteration errors
            continue

        # Compute softmax to transform logits into probabilities
        conf_AA = CSoftmax().softmax(scores_AA)
        confidence_AA.append(conf_AA)
    print("-" * 90)
    print("\nConfidence for AutoAttack successfully computed!\n")
    print("-" * 90)

    #################################################################################
    print("\nIdentifying samples where one attack succeeds while the other fails!\n")
    # Identifying samples with differing attack results
    mismatched_samples = {}
    print("-" * 90)
    for model_idx, model_name in enumerate(model_names):
        adv_ds_AA = results_AA_data[model_idx]['x_adv']  # Adversarial images from AutoAttack
        adv_ds_FMN = results_FMN_data[model_idx]['result']['adv_ds'].X  # Adversarial images from FMN

        confidence_FMN = results_FMN_data[model_idx]['result']['confidence']  # Confidence scores for FMN
        confidence_AA_model = confidence_AA[model_idx]  # Use precomputed confidence scores

        y_true = results_FMN_data[model_idx]['result']['adv_ds'].Y.tondarray()
        y_pred_AA = results_AA_data[model_idx]['y_pred_adv'].tondarray()  # Convert to NumPy array
        y_pred_FMN = results_FMN_data[model_idx]['result']['y_pred_adv'].tondarray()

        # Identifying samples where attack results differ
        differing_indices = [
            idx for idx in range(y_pred_AA.shape[0])
            if (y_pred_AA[idx] != y_pred_FMN[idx]) and
               (y_pred_AA[idx] == y_true[idx] or y_pred_FMN[idx] == y_true[idx])
        ]

        # Updated structure: store index + true and adversarial labels
        mismatched_samples[model_name] = [
            {
                "sample": idx,
                "true_label": dataset_labels[y_true[idx]],
                "adv_label_AA": dataset_labels[y_pred_AA[idx]],
                "adv_label_FMN": dataset_labels[y_pred_FMN[idx]]
            }
            for idx in differing_indices
        ]
        print(f"Analyzing model: {model_name:<40} - Mismatched samples: {len(mismatched_samples[model_name])}")
    print("-" * 90)

    #################################################################################

    Confidence_AA_4_sample, Confidence_FMN_4_sample = analyze_discrepant_samples(
        mismatched_samples, model_names, confidence_AA, confidence_FMN, y_pred_AA, y_pred_FMN
    )

    #################################################################################################################

    # L∞ Perturbation Analysis
    # Create a mapping between model_name and model_idx
    model_name_to_idx = {name: idx for idx, name in enumerate(model_names)}

    Perturbation_Linf = {}
    for model_name, indices in mismatched_samples.items():
        model_idx = model_name_to_idx[model_name]
        print("-" * 90)
        print(f"\nL∞ Perturbation Analysis for model {model_name}")
        Perturbation_Linf[model_name] = {}
        for sample_data in indices:
            idx = sample_data["sample"]
            x_orig = ts.X[idx, :].tondarray().squeeze()
            x_adv_AA = results_AA_data[model_idx]['x_adv'][idx, :].tondarray().squeeze()
            x_adv_FMN = results_FMN_data[model_idx]['result']['adv_ds'].X[idx, :].tondarray().squeeze()
            linf_AA = np.linalg.norm(x_adv_AA - x_orig, ord=np.inf)
            linf_FMN = np.linalg.norm(x_adv_FMN - x_orig, ord=np.inf)
            Perturbation_Linf[model_name][idx] = {'AA': linf_AA, 'FMN': linf_FMN}
            print(f"- Sample {idx}: L∞ Perturbation - AA: {linf_AA:.4f}, FMN: {linf_FMN:.4f}")
        print("-" * 90)

    # Call the function
    plot_perturbations(
        models=models,
        model_name_to_idx=model_name_to_idx,
        dataset_labels=dataset_labels,
        ts=ts,
        results_AA_data=results_AA_data,
        results_FMN_data=results_FMN_data,
        Perturbation_Linf=Perturbation_Linf,
        Confidence_AA_4_sample=Confidence_AA_4_sample,
        Confidence_FMN_4_sample=Confidence_FMN_4_sample,
        input_shape=input_shape,
        max_samples=3  # Show only 3 examples per model
    )

    ### Explainability Analysis ###
    explainability_analysis(models, model_names, results_FMN_data, ts, dataset_labels, input_shape)

    ### Confidence Analysis ###
    confidence_analysis(models, model_names, ts, dataset_labels,
                        results_file_confidence="extracted_data/data_attack_result_FMN_CONFIDENCE.pkl", num_samples=5)

    print("\n✅ Execution completed!")




