import torch
import time
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

from utils.metrics import AverageMeter, binary_accuracy, compute_image_f1
from sklearn.metrics import f1_score, roc_auc_score

class CBMEvaluator:
    def __init__(self,
                 model,
                 num_attr: int,
                 attr_cols, 
                 test_dataloader,
                 device: torch.device, 
                 bottleneck: bool = False,
                 concepts: bool = False,
                 main_only: bool = False):

        self.model = model
        self.num_attr = num_attr
        self.attr_cols = attr_cols
        self.test_dataloader = test_dataloader
        self.device = device
        self.bottleneck = bottleneck
        self.concepts = concepts
        self.main_only = main_only
    
    #-------Function to evaluate model performance on test set---------
    def evaluate(self, return_scores = False):
        self.model.eval()

        accuracy_meter_main = AverageMeter()
        accuracy_meter_attr = AverageMeter()

        if not self.bottleneck:
            all_main_targets, all_main_probs = [], []
        else:
            all_main_targets = all_main_probs = None

        if not self.main_only:
            all_attr_probs, all_attr_targets = [], []
        else:
            all_attr_probs = all_attr_targets = None
            accuracy_meter_attr = None
        
        total_inference_time = 0.0
        total_instances = 0
        
        for sample in self.test_dataloader:
            if self.main_only:
                inputs, labels = sample
                inputs = inputs.to(self.device)
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs).t().float()
            else:
                inputs, concepts, labels = sample
                inputs, concepts = inputs.to(self.device), concepts.to(self.device)

            with torch.no_grad():
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                predictions = self.model(inputs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
            
            batch_time = end - start
            total_inference_time += batch_time
            total_instances += inputs.size(0)

            #compute main task accuracy
            if not self.bottleneck:
                logits_main = predictions[0].squeeze(1) if isinstance(predictions, list) else predictions.squeeze(1)
                probs_main = torch.sigmoid(logits_main)

                all_main_targets.append(labels.cpu().int())
                all_main_probs.append(probs_main.cpu())

                accuracy_main = binary_accuracy(probs_main, labels)
                accuracy_meter_main.update(accuracy_main.item(), inputs.size(0))
            
            #if attributes exist, compute attribute accuracy
            if not self.main_only:
                if not self.bottleneck:
                    attr_logits = torch.cat(predictions[1:], dim = 1)
                else:
                    attr_logits = torch.cat(predictions, dim = 1)
                
                probs_attr = torch.sigmoid(attr_logits)

                all_attr_probs.append(probs_attr.cpu())
                all_attr_targets.append(concepts.cpu().int())

                accuracy_attr = binary_accuracy(probs_attr.cpu(), concepts.cpu())
                accuracy_meter_attr.update(accuracy_attr.numpy(), inputs.size(0))
        
        if not self.bottleneck and all_main_probs is not None:
            all_main_targets = torch.cat(all_main_targets).numpy()
            all_main_probs = torch.cat(all_main_probs).numpy()
            image_scores = all_main_probs

            #compute main F1 score
            f1_main, best_thresh_main = compute_image_f1(all_main_targets, all_main_probs)
            auc_main = roc_auc_score(all_main_targets, all_main_probs)

            #predict label according to the best threshold
            main_preds = (all_main_probs >= best_thresh_main).astype(int)

        else:
            f1_main = 0
            auc_main = 0
            image_scores = np.array([])

        if all_attr_probs:
            all_attr_probs = torch.cat(all_attr_probs).numpy()
            all_attr_targets = torch.cat(all_attr_targets).numpy()

            #compute AUC
            aucs = []
            for i in range(all_attr_targets.shape[1]):
                try:
                    auc = roc_auc_score(all_attr_targets[:, i], all_attr_probs[:, i])
                    aucs.append(auc)
                except ValueError:
                    aucs.append(np.nan)

            mean_auc = np.nanmean(aucs)

            #compute F1 score
            f1_attr_scores = []
            thresholds_attr = []
            for i in range(all_attr_targets.shape[1]):
                f1_i, thr_i = compute_image_f1(all_attr_targets[:, i], all_attr_probs[:, i])
                f1_attr_scores.append(f1_i)
                thresholds_attr.append(thr_i)

            f1_attr = np.nanmean(f1_attr_scores)

            # #compute attribute-specific f1-score
            # f1_scores_attr = []
            # for i, attr_name in enumerate(self.attr_cols):
            #     f1 = f1_score(all_attr_targets[:, i], all_attr_preds[:, i])
            #     f1_scores_attr.append((attr_name, f1))

            # f1_scores_attr = sorted(f1_scores_attr, key=lambda x: (np.isnan(x[1]), x[1]))

            # # print("\nPer-Attribute F1 Score:")
            # # for name, score in f1_scores_attr:
            # #     print(f"{name}: F1 Score = {score:.4f}")

        else:
            mean_auc = 0
            f1_attr = 0
        
        #avg_inference_time = total_inference_time / total_instances

        if self.bottleneck:
            print(f"Concept AUC = {mean_auc:.4f}, Concept F1 = {f1_attr:.4f}")
        else:
            print(f"\nMain Task AUC = {auc_main:.4f}, Main Task F1 = {f1_main:.4f} (at threshold  = {best_thresh_main:.4f})")
            if self.concepts:
                print(f"Concept AUC = {mean_auc:.4f}, Concept F1 = {f1_attr:.4f}")
        
        #print(f"\nAverage inference time per instance: {avg_inference_time*1000:.4f} ms")

        if self.bottleneck:
            return mean_auc, f1_attr
        else:
            if self.main_only:
                return auc_main, f1_main
            if return_scores:
                return auc_main, f1_main, mean_auc, f1_attr, main_preds, image_scores
        return auc_main, f1_main, mean_auc, f1_attr, main_preds
    
    
    #-----Function to perform inference on a single image-------
    def inference(self, image_path):

        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        #parse results
        if not self.bottleneck:
            if isinstance(predictions, list):
                main_logits = predictions[0].squeeze(1)
                attr_logits = torch.cat(predictions[1:], dim=1) if len(predictions) > 1 else None
            else:
                main_logits = predictions.squeeze(1)
                attr_logits = None
        else:
            main_logits = None
            attr_logits = torch.cat(predictions, dim=1)

        #compute predicted label
        main_pred = None
        main_prob = None
        if main_logits is not None:
            main_prob = torch.sigmoid(main_logits)
            main_pred = (main_prob >= 0.5).int()

        #compute predicted concepts
        attr_probs, attr_preds = [], []
        if attr_logits is not None:
            attr_probs = torch.sigmoid(attr_logits).squeeze(0).cpu().numpy()
            attr_preds = (attr_probs >= 0.5).astype(int)

        if main_pred is not None:
            print(f"Main Task Prediction: {'Anomalous' if main_pred else 'Normal'}")
        if len(self.attr_cols) > 0 and len(attr_preds) > 0:
            print("\nConcept Predictions:")
            for name, pred in zip(self.attr_cols, attr_preds):
                print(f"  - {name}: {'Present' if pred else 'Absent'}")




