import torch
import torch.nn.functional as F

from MedSAM.network import MedSAM
from segment_anything import SamPredictor

from typing import Optional, Dict
import numpy as np
from skimage import transform  # Import transform for resizing


class Predictor:
    """
    Wrapper for MedSAM model
    """

    def __init__(self,
                 path: str,
                 model_type: str = 'vit_b',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 verbose: bool = False):

        self.verbose = verbose
        self.device = device
        self.model_type = model_type
        self.build_model(path)
        self.sam_predictor = SamPredictor(self.model.sam)

    def build_model(self, path):
        """
        Build the model
        """
        self.model = MedSAM(checkpoint=path,
                            model_type=self.model_type,
                            device=self.device)

    def medsam_inference(self, img_embed, box_1024, H, W):
        box_torch = torch.as_tensor(box_1024,
                                   dtype=torch.float,
                                   device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None,:]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.model.sam.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = self.model.sam.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.model.sam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(low_res_pred,
                                     size=(H, W),
                                     mode="bilinear",
                                     align_corners=False)  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.detach().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg

    def predict(self,
                prompts: Dict[str, any],
                img_features: Optional[torch.Tensor] = None,
                multimask_mode: bool = False):
        """
        Make predictions!

        Returns:
            mask (torch.Tensor): H x W
            img_features (torch.Tensor): B x 1 x H x W (for SAM models)
            low_res_mask (torch.Tensor): B x 1 x H x W logits
        """
        if self.verbose:
            print("point_coords", prompts.get("point_coords", None))
            print("point_labels", prompts.get("point_labels", None))
            print("box", prompts.get("box", None))
            print("img", prompts.get("img").shape, prompts.get("img").min(),
                  prompts.get("img").max())
            if prompts.get("scribble") is not None:
                print("scribble", prompts.get("scribble", None).shape,
                      prompts.get("scribble").min(),
                      prompts.get("scribble").max())

        # Prepare image and prompts for MedSAM
        image = prompts.get('img').squeeze(1).cpu().numpy()
        original_shape = prompts.get('original_shape')
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image, image, image],
                             axis=-1)  #convert to RGB
        elif image.ndim == 3 and image.shape[
                0] == 1:  #check if single channel, and convert to RGB
            image = np.concatenate([image, image, image],
                                   axis=0)  #convert to RGB
        elif image.ndim == 3 and image.shape[
                2] == 1:  #check if single channel, and convert to RGB
            image = np.concatenate([image, image, image],
                                   axis=2)  #convert to RGB
        image = np.transpose(image, (1, 2, 0)).astype(
            np.uint8)  #transpose after channel manipulation

        # ---  Changes start here ---

        # Resize image to 1024x1024
        img_1024 = transform.resize(image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Get image features using image_encoder
        with torch.no_grad():
            img_features = self.model.sam.image_encoder(
                img_1024_tensor)  # (B, 256, 64, 64)

        # --- Changes end here ---

        if prompts.get('point_coords') is not None:
            points = prompts.get('point_coords').squeeze().cpu().numpy()
            labels = prompts.get('point_labels').squeeze().cpu().numpy()
            prompts['point_coords'] = points
            prompts['point_labels'] = labels

        box_prompt = prompts.get('box')
        if box_prompt is not None:
            box_prompt = box_prompt.squeeze().cpu().numpy()
            # transfer box_np t0 1024x1024 scale
            H, W = original_shape
            box_1024 = box_prompt / np.array([W, H, W, H]) * 1024
            box_1024 = box_1024.reshape(1, 4)
            medsam_mask = self.medsam_inference(img_features, box_1024,
                                               original_shape[0],
                                               original_shape[1])
        else:
            print("No box prompt provided.")
            return None, img_features, None

        return medsam_mask, img_features, medsam_mask