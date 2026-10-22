import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
import os

from matplotlib.ticker import MaxNLocator

from utils.intervention_utils import compute_intervention_order, modify_concepts
from datasets.concept_dataset import ConceptDataset
from models.full_models import joint_model, concepts_model, main_model
from evaluators.evaluator_cbm import CBMEvaluator
from utils.model_utils import generate_concept_logits


def read_concept_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    base = os.path.dirname(os.path.abspath(csv_path))

    def resolve(p):
        if pd.isna(p) or p == "":
            return p  # good images have no mask
        return os.path.normpath(os.path.join(base, p))

    for col in ("image_path", "mask_path"):
        if col in df.columns:
            df[col] = df[col].apply(resolve)
    return df

def simulate_concept_intervention(dataset_path: str,
                                  category: str,
                                  use_gen_anomalies: bool,
                                  contaminate: bool,
                                  n_per_type: int,
                                  save_dir: str,
                                  model_type: str,
                                  device: torch.device,
                                  batch_size: int = 8):

    # construct save path
    if use_gen_anomalies:
        file_name = "unsupervised_model.pth"
        if contaminate:
            file_name = f"semisupervised_{n_per_type}_model.pth"
    else:
        file_name = "supervised_model.pth"

    save_dir = os.path.join(save_dir, category, "cbm", model_type)
    if model_type == "joint":
        save_path = os.path.join(save_dir, file_name)
        save_path_concepts = None
    elif model_type in ["sequential", "independent"]:
        save_path = os.path.join(save_dir, file_name)
        save_path_concepts = os.path.join(save_dir, f"concepts_{file_name}.pth")

    if use_gen_anomalies:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset_gen_to_real.csv")
    else:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset.csv")
    dataframe = read_concept_csv(dataframe_path)
    gt_test_df = dataframe[dataframe["split"] == "test"]

    state_dict = torch.load(save_path, weights_only=False)
    if model_type in ["independent", "sequential"]:
        state_dict_concepts = torch.load(save_path_concepts, weights_only=False)

    test_dataset = ConceptDataset(dataframe, "test", use_attr=True)
    num_attr = len(test_dataset.attr_cols)
    attr_cols = test_dataset.attr_cols

    f1_scores = []

    # step 1: extract predicted concepts
    print(f"Extracting predicted concepts...")

    if model_type in ["sequential", "independent"]:
        concept_model = concepts_model(num_attr=num_attr, freeze_parameters=True, model_state_dict=state_dict_concepts, mode="test")

    else:
        concept_model, main_task_model = joint_model(num_attr=num_attr, freeze_parameters=True, use_relu=True, model_state_dict=state_dict, mode="test", concept_intervention=True)

    concept_model.to(device)

    pred_df = generate_concept_logits(concept_model, dataframe, save_path=None, device=device)
    pred_df_test = pred_df[pred_df["split"] == "test"]

    # compute 5th and 95th percentile over the training distribution
    logits_array = pred_df[attr_cols][pred_df["split"] == "train"].values
    ptl_5 = np.percentile(logits_array, 5, axis=0)
    ptl_95 = np.percentile(logits_array, 95, axis=0)
    ptl_5, ptl_95 = dict(zip(attr_cols, ptl_5)), dict(zip(attr_cols, ptl_95))

    # step 2: modify concepts based on intervention order
    print(f"Computing intervention order...")
    intervention_order = compute_intervention_order(pred_df_test, attr_cols)

    for i in range(num_attr):
        print(f"\nInervenening on the first {i + 1} concept(s)...")
        modified_df = modify_concepts(intervention_order, gt_test_df, pred_df_test, attr_cols, ptl_5, ptl_95, i + 1)

        # step 3: perform inference over the main task using the new df
        new_test_dataset = ConceptDataset(modified_df, "test", use_attr=True, load_image=False)
        attr_cols = new_test_dataset.attr_cols
        new_test_dataloader = torch.utils.data.DataLoader(new_test_dataset, batch_size, shuffle=False)

        print(f"Performing inference of {model_type} model using the new concepts...")
        if model_type in ["sequential", "independent"]:
            main_task_model = main_model(num_attr=num_attr, model_state_dict=state_dict)

        main_task_model.to(device)

        main_evaluator = CBMEvaluator(main_task_model, num_attr, attr_cols, new_test_dataloader, device, main_only=True)
        auc_main, f1_main = main_evaluator.evaluate()

        f1_scores.append(f1_main)

    print("\nF1 score by number of intervened concepts:")
    for n_concepts, f1 in enumerate(f1_scores, start=1):
        print(f"  {n_concepts:>3} concept(s): F1 = {f1:.4f}")

    return f1_scores



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--use_gen_anomalies", action="store_true", help="Unsupervised or semi-supervised scenario")
    parser.add_argument("--contaminate", action="store_true", help="Semi-supervised scenario")
    parser.add_argument("--n_per_type", type=int, default=0, help="How many original images to add to the generated dataset")
    parser.add_argument("--model_type", type=str, default = "joint", help="Which model to train, e.g. 'independent', 'joint', ...")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the models in")
    parser.add_argument("--category", type=str, help="Which category to train/test")
    parser.add_argument("--device", type=str, default = "cuda", help="Where to run the script")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size to use")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    f1_scores = simulate_concept_intervention(args.dataset_path, args.category, args.use_gen_anomalies, args.contaminate, args.n_per_type,
                                              args.save_dir, args.model_type, device, args.batch_size)


if __name__ == "__main__":
    main()
