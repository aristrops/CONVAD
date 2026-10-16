import argparse
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import rankdata

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve

from datasets.concept_dataset import ConceptDataset
from models.full_models import joint_model
from evaluators.evaluator_cbm import CBMEvaluator
from evaluators.evaluator_stfpm import STFPMEvaluator
from models.model_backbones import BackboneModelFeatures

#---------HELPERS----------------

#compute best f1 threshold on held-out set
def best_f1_threshold(scores: np.ndarray, y_true: np.ndarray):
    prec, rec, thresholds = precision_recall_curve(y_true, scores)
    denom = prec[:-1] + rec[:-1]
    f1 = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / denom, 0.0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def compute_model_size_mb(*model_paths: str) -> dict:
    seen   = {}   # path -> size in MB
    result = {}

    for path in model_paths:
        if path in seen:
            result[path] = seen[path]
            continue
        try:
            size_mb = os.path.getsize(path) / (1024 ** 2)
        except FileNotFoundError:
            print(f"  [WARNING] File not found for size check: {path}")
            size_mb = 0.0
        seen[path]   = size_mb
        result[path] = size_mb

    result["total_mb"] = sum(seen.values())
    return result


def build_path(save_dir, branch_name, use_gen_anomalies, contaminate, n_per_type, model_type = "joint"):
    save_dir = os.path.join(save_dir, branch_name, model_type)

    if use_gen_anomalies:
        file_name = "unsupervised_model.pth"
        if contaminate:
            file_name = f"semisupervised_{n_per_type}_model.pth"
    else:
        file_name = "supervised_model.pth"

    model_path = os.path.join(save_dir, file_name)

    return model_path


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


def make_dataloaders(dataframe: pd.DataFrame, split: str, batch_size: int):
    stfpm_dataset = ConceptDataset(
        dataframe,
        split=split,
        use_attr=False,
        load_mask=True,
    )
    stfpm_dataloader = torch.utils.data.DataLoader(
        stfpm_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    cbm_dataset = ConceptDataset(
        dataframe,
        split,
        use_attr=True,
        load_mask=False,
    )
    cbm_dataloader = torch.utils.data.DataLoader(
        cbm_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    num_attr = len(cbm_dataset.attr_cols)
    attr_cols = cbm_dataset.attr_cols

    return stfpm_dataloader, cbm_dataloader, num_attr, attr_cols


def get_labels(dataframe: pd.DataFrame, split: str) -> np.ndarray:
    return (
        dataframe[dataframe["split"] == split]["label_index"]
        .to_numpy()
        .astype(int)
    )


 #------CONCEPT BRANCH-------------
def test_cbm(dataloader: str,
             cbm_path: str,
             num_attr: int,
             attr_cols: list,
             device: torch.device):

    state_dict = torch.load(cbm_path, weights_only=False)

    model = joint_model(num_attr=num_attr, use_relu=True, use_sigmoid=False, model_state_dict=state_dict, mode="test")
    model.to(device)

    evaluator = CBMEvaluator(model, num_attr, attr_cols, dataloader, device)
    auc_main, f1_main, auc_attr, f1_attr, main_preds, image_scores = evaluator.evaluate(return_scores=True)

    return auc_main, auc_attr, image_scores


 #--------VISUAL BRANCH-----------
def test_stfpm(dataloader,
               teacher_path: str,
               student_path: str,
               device: torch.device):

    state_dict = torch.load(teacher_path, weights_only=False)
    state_dict = {
        k.replace("first_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("first_model.")
    }
    state_dict_student = torch.load(student_path)
 
    teacher_model = BackboneModelFeatures(pretrained=True)
    teacher_model.load_state_dict(state_dict, strict=False)
 
    student_model = BackboneModelFeatures(pretrained=True)
    student_model.load_state_dict(state_dict_student, strict=False)
 
    evaluator = STFPMEvaluator(teacher_model, student_model, dataloader, device)
    image_auc, image_f1, image_preds, pixel_auc, image_scores = evaluator.evaluate(return_scores=True)
 
    return image_auc, pixel_auc, image_scores

 
#---------COMBINED EVALUATION---------------
def evaluate_combined_performance(category: str,
                                  dataset_path: str,
                                  use_gen_anomalies: bool,
                                  contaminate: bool,
                                  n_per_type: int,
                                  save_dir: str,
                                  device: torch.device,
                                  batch_size: int = 8):

    save_dir = os.path.join(save_dir, category)
    cbm_path = teacher_path = build_path(save_dir, "cbm", use_gen_anomalies, contaminate, n_per_type)
    student_path = build_path(save_dir, "stfpm", use_gen_anomalies, contaminate, n_per_type)

    size_info = compute_model_size_mb(cbm_path, student_path)

    print(f"\n── Model sizes ──")
    print(f"  Concept / Teacher : {size_info[cbm_path]:.2f} MB  (shared, counted once)")
    print(f"  Student       : {size_info[student_path]:.2f} MB")
    print(f"  Total         : {size_info['total_mb']:.2f} MB")

    if use_gen_anomalies:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset_gen_to_real.csv")
    else:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset.csv")

    dataframe = read_concept_csv(dataframe_path)

    print("\nBuilding validation dataloaders...")
    stfpm_val_dl, cbm_val_dl, num_attr, attr_cols = make_dataloaders(dataframe, "val", batch_size)
 
    y_val = get_labels(dataframe, "val")
    print(f" Val images: {len(y_val)}  (anomalous: {y_val.sum()})")

    print("\nBuilding test dataloaders...")
    stfpm_test_dl, cbm_test_dl, num_attr, attr_cols = make_dataloaders(dataframe, "test", batch_size)
 
    y_true = get_labels(dataframe, "test")
    print(f" Test images: {len(y_true)}  (anomalous: {y_true.sum()})")

    print("\nEvaluating STFPM model on validation set")
    _, _, scores_stfpm_val = test_stfpm(stfpm_val_dl, teacher_path, student_path, device)

    print("\nEvaluating CBM model on validation set...")
    _, _, scores_cbm_val = test_cbm(cbm_val_dl, cbm_path, num_attr, attr_cols, device)

    print("\nEvaluating STFPM model on TEST set…")
    auc_stfpm, pixel_auc_stfpm, scores_stfpm = test_stfpm(stfpm_test_dl, teacher_path, student_path, device)

    print("\nEvaluating CBM model on TEST set…")
    auc_cbm, concept_auc_cbm, scores_cbm = test_cbm(cbm_test_dl, cbm_path, num_attr, attr_cols, device)
 
    #rank-normalize (val)
    n_val = len(scores_stfpm_val)
    scores_stfpm_val_rank = rankdata(scores_stfpm_val, method="average") / n_val
    scores_cbm_val_rank   = rankdata(scores_cbm_val,   method="average") / n_val
 
    #rank_normalize (test)
    n_test = len(scores_stfpm)
    scores_stfpm_rank = rankdata(scores_stfpm, method="average") / n_test
    scores_cbm_rank   = rankdata(scores_cbm,   method="average") / n_test
 
    #compute thresholds on validation
    thr_cbm,   f1_cbm_val   = best_f1_threshold(scores_cbm_val_rank,   y_val)
    thr_stfpm, f1_stfpm_val = best_f1_threshold(scores_stfpm_val_rank, y_val)

    #binary predictions on test set using val-calibrated thresholds
    pred_cbm_test   = (scores_cbm_rank   >= thr_cbm).astype(int)
    pred_stfpm_test = (scores_stfpm_rank >= thr_stfpm).astype(int)

    stfpm_only_anomaly = (pred_stfpm_test == 1) & (pred_cbm_test == 0)
 
    combined_scores = scores_cbm_rank.copy()
    combined_scores[stfpm_only_anomaly] = 1.0

    agree = pred_cbm_test == pred_stfpm_test
    cbm_only_anom = (pred_cbm_test == 1) & (pred_stfpm_test == 0)
 
    print(f"\n── Branch agreement on test set")
    print(f"Agree (both normal)   : "
          f"{((pred_cbm_test==0)&(pred_stfpm_test==0)).sum():>4d}  "
          f"({((pred_cbm_test==0)&(pred_stfpm_test==0)).mean()*100:.1f}%)")
    print(f"Agree (both anomaly)  : "
          f"{((pred_cbm_test==1)&(pred_stfpm_test==1)).sum():>4d}  "
          f"({((pred_cbm_test==1)&(pred_stfpm_test==1)).mean()*100:.1f}%)")
    print(f"CBM-only anomaly      : "
          f"{cbm_only_anom.sum():>4d}  ({cbm_only_anom.mean()*100:.1f}%)")
    print(f"STFPM-only anomaly    : "
          f"{stfpm_only_anomaly.sum():>4d}  ({stfpm_only_anomaly.mean()*100:.1f}%)")
 
    print(f"\nCBM dominance breakdown by ground-truth class")
    for lbl, name in [(0, "Normal"), (1, "Anomalous")]:
        mask = y_true == lbl
        if mask.sum() == 0:
            continue
        n_cbm_driven = (~stfpm_only_anomaly[mask]).sum()
        pct          = n_cbm_driven / mask.sum() * 100
        print(f"{name:10s}: CBM drives score in "
              f"{n_cbm_driven}/{mask.sum()} images ({pct:.1f}%)")

    print(f"\n── Score statistics ─────────────────────────────────────────")
    print(f"  STFPM raw   : [{scores_stfpm.min():.4f}, {scores_stfpm.max():.4f}]")
    print(f"  CBM raw     : [{scores_cbm.min():.4f},   {scores_cbm.max():.4f}]")
    print(f"  STFPM rank  : [{scores_stfpm_rank.min():.4f}, {scores_stfpm_rank.max():.4f}]")
    print(f"  CBM rank    : [{scores_cbm_rank.min():.4f},   {scores_cbm_rank.max():.4f}]")
    print(f"  Combined    : [{combined_scores.min():.4f},   {combined_scores.max():.4f}]")

    auc_combined = roc_auc_score(y_true, combined_scores)
 
    print(f"\n── STFPM branch ──────────────────────────────────────────────")
    print(f"  Image AUC  : {auc_stfpm:.4f}")
    print(f"  Pixel AUC  : {pixel_auc_stfpm:.4f}")
 
    print(f"\n── CBM branch ────────────────────────────────────────────────")
    print(f"  Image AUC  : {auc_cbm:.4f}")
    print(f"  Concept AUC: {concept_auc_cbm:.4f}")
 
    print(f"\n── Combined (CBM-prioritised fusion) ─────────────────────────")
    print(f"  Image AUC  : {auc_combined:.4f}")
 
    return {
        "category":           category,
        "stfpm_auc":          auc_stfpm,
        "stfpm_pixel_auc":    pixel_auc_stfpm,
        "cbm_auc":            auc_cbm,
        "cbm_concept_auc":    concept_auc_cbm,
        "combined_auc":       auc_combined,
        "pct_agree":          agree.mean() * 100,
        "pct_cbm_only_anom":  cbm_only_anom.mean() * 100,
        "pct_stfpm_only_anom": stfpm_only_anomaly.mean() * 100,
    }
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--use_gen_anomalies", action="store_true", help="Unsupervised or semi-supervised scenario: use generated anomalies to train")
    parser.add_argument("--contaminate", action="store_true",
                        help="Semi-supervised scenario: contaminate the generated dataset with some original anomalous samples")
    parser.add_argument("--n_per_type", type=int, default=0,
                        help="How many original images to add to the generated dataset")
    parser.add_argument("--category", type=str, help="Which category to train/test")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the models in")
    parser.add_argument("--device",     type=str, default="cpu", help="Torch device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed",       type=int, default=0)

    args = parser.parse_args()
 
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    results = evaluate_combined_performance(args.category, args.dataset_path, args.use_gen_anomalies, args.contaminate, args.n_per_type, args.save_dir, device)

 
if __name__ == "__main__":
    main()
 







