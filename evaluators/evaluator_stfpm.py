import numpy as np
import matplotlib.pyplot as plt
import torch
import os

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter

from utils.metrics import compute_pixel_f1, min_max_norm, compute_pixel_pro, compute_pixel_pr, compute_image_f1

class STFPMEvaluator:
    def __init__(self,
                teacher_model,
                student_model,
                test_dataloader,
                device: torch.device):

        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.test_dataloader = test_dataloader
        self.device = device

    def unnormalize_image(self, img_tensor, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        for t, m, s in zip(img_tensor, mean, std):
            t.mul_(s).add_(m)
        return img_tensor
    
    #-------Function to visualize the predicted heatmap---------
    def visualize(self, image, mask, save_path = None):

        self.teacher_model.eval()
        self.student_model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0).to(self.device)
        else:
            image = image.to(self.device)

        mask = mask.unsqueeze(0)

        with torch.no_grad():
            t_features = self.teacher_model(image)
            s_features = self.student_model(image)

        score_map = 1.

        for j in range(len(t_features)):
            t_features[j] = F.normalize(t_features[j], dim = 1)
            s_features[j] = F.normalize(s_features[j], dim = 1)

            sm = torch.sum((t_features[j] - s_features[j])**2, 1, keepdim = True)
            sm = F.interpolate(sm, size = (64, 64), mode = "bilinear", align_corners = False)

            score_map = score_map * sm
        
        score_map = score_map.squeeze(0).squeeze(0).cpu().numpy()
        print("Shape of score map:", score_map.shape)
        norm_score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)

        #resize score map
        original_size = image.shape[-2:]
        score_map_resized = F.interpolate(torch.tensor(norm_score_map).unsqueeze(0).unsqueeze(0), 
                                            size = original_size, mode = "bilinear", align_corners=False).squeeze().numpy()
        
        #convert tensors to image
        unnorm_tensor = self.unnormalize_image(image.cpu().squeeze().clone())
        input_image = unnorm_tensor.permute(1, 2, 0).numpy()
        input_image = np.clip(input_image, 0, 1)

        smoothed_mask = gaussian_filter(mask.float(), sigma=2).squeeze(0).squeeze(0)
        gt_mask = F.interpolate(torch.tensor(smoothed_mask).unsqueeze(0).unsqueeze(0), 
                                            size = original_size, mode = "bilinear", align_corners=False).squeeze().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(input_image)
        axs[0].set_title("Input Image")
        axs[1].imshow(input_image)
        axs[1].imshow(score_map_resized, cmap="jet", alpha=0.5)
        axs[1].set_title("Predicted Heatmap")
        axs[2].imshow(input_image)
        axs[2].imshow(gt_mask, cmap="jet", alpha = 0.5)
        axs[2].set_title("Ground Truth Heatmap")

        for ax in axs:
            ax.axis("off")
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

        plt.close()
        
    #-------Function to evaluate model performance on test set---------
    def evaluate(self, return_scores = False):

        self.teacher_model.eval()
        self.student_model.eval()

        loss_map = np.zeros((len(self.test_dataloader.dataset), 64, 64))
        gt_masks = []

        i = 0

        for image, label, mask in self.test_dataloader:
            image = image.to(self.device)
            with torch.no_grad():
                t_features = self.teacher_model(image)
                s_features = self.student_model(image)

            score_map = 1.

            for j in range(len(t_features)):
                t_features[j] = F.normalize(t_features[j], dim = 1)
                s_features[j] = F.normalize(s_features[j], dim = 1)

                sm = torch.sum((t_features[j] - s_features[j])**2, 1, keepdim = True)
                sm = F.interpolate(sm, size = (64, 64), mode = "bilinear", align_corners = False)

                score_map = score_map * sm
            
            loss_map[i: i + image.size(0)] = score_map.squeeze().cpu().data.numpy()

            #collect and resize GT masks
            resized_masks = F.interpolate(mask.float(), size = (64, 64), mode = "nearest")
            gt_masks.append(resized_masks.squeeze(1).cpu().numpy())

            i += image.size(0)
        
        gt_masks = np.concatenate(gt_masks, axis = 0)

        # ---------------- Pixel-level metrics ----------------
        y_pred = loss_map
        pred_masks = min_max_norm(y_pred)

        if isinstance(gt_masks, np.ndarray):
            y_true = (gt_masks > 0.5).astype(int)
        else:  
            y_true = (gt_masks > 0.5).int()
        
        pixel_auc = roc_auc_score(y_true.flatten(), pred_masks.flatten())
        pixel_f1_score = compute_pixel_f1(y_true, y_pred)
        pixel_pr = compute_pixel_pr(pred_masks, y_true)
        pixel_pro = compute_pixel_pro(pred_masks, y_true)

        # ---------------- Image-level metrics ----------------
        image_labels = (np.sum(gt_masks, axis=(1, 2)) > 0).astype(int)
        image_scores = np.max(pred_masks.reshape(pred_masks.shape[0], -1), axis=1) #max across the predicted heatmap
        image_auc = roc_auc_score(image_labels, image_scores)
        image_f1_max, best_thresh = compute_image_f1(image_labels, image_scores)

        #predict labels using the best threshold
        image_preds = (image_scores >= best_thresh).astype(int)

        print(f"Pixel AUC = {pixel_auc:.4f}, Pixel F1 = {pixel_f1_score:.4f}, "
              f"Pixel PRO = {pixel_pro:.4f}, Pixel PR = {pixel_pr:.4f}")
        print(f"Image AUC = {image_auc:.4f}, Image F1-max = {image_f1_max:.4f} (at threshold = {best_thresh:.4f})")

        if return_scores:
            return image_auc, image_f1_max, image_preds, pixel_auc, image_scores
        return image_auc, image_f1_max, image_preds
