import os
import sys
sys.path.append('/home/mendax/project/CLIPPose')
import json
import torch
import numpy as np
from PIL import Image
from clippose import load

class CLIPPoseSimilarity:
    def __init__(self, model_name="ViT-B/32", checkpoint_path=None, device=None):
        """
        Initialize the CLIPPose similarity class.

        Args:
            model_name (str): Name of the CLIPPose model.
            checkpoint_path (str): Path to the checkpoint file.
            device (str): Device to use (e.g., "cuda:0" or "cpu").
        """
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = load(model_name, device=self.device, jit=False)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def preprocess_images(self, images):
        """
        Preprocess a list of images.

        Args:
            images (list of PIL.Image): List of images to preprocess.

        Returns:
            torch.Tensor: Preprocessed images as a tensor.
        """
        return torch.tensor(np.stack([self.preprocess(image) for image in images])).to(self.device)

    def calculate_similarity(self, m, refer_images, test_images):
        """
        Calculate similarity between test images and reference images for a given object.

        Args:
            m (int): Number of top similar references to select.
            refer_images (list of PIL.Image): List of reference images.
            test_images (list of PIL.Image): List of test images.

        Returns:
            torch.Tensor: Indices of the top m similar references for each test image.
        """
        # Preprocess reference and test images
        refers_input = self.preprocess_images(refer_images)
        image_input = self.preprocess_images(test_images)

        # Compute features
        with torch.no_grad():
            refer_features = self.model.encode_image(refers_input).float()
            image_features = self.model.encode_image(image_input).float()

        # Normalize features and calculate similarity
        refer_features_norm = refer_features / refer_features.norm(dim=-1, keepdim=True)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = image_features_norm @ refer_features_norm.T

        # Select top m similar references
        similaritys, top_m_indices = torch.topk(similarity, m, dim=1, largest=True, sorted=True)

        return top_m_indices, similaritys

if __name__ == "__main__":
    m = 10
    checkpoint_path = "checkpoint/model_20.pt"
    obj_id = 1

    # Example reference and test images
    refer_images = []
    refer_dirs = f"/home/mendax/project/CLIPPose/refers/obj_{obj_id}"
    for filename in [filename for filename in os.listdir(refer_dirs) if filename.endswith(".png") or filename.endswith(".jpg")]:
        refer = Image.open(os.path.join(refer_dirs, filename)).convert("RGB")
        refer_images.append(refer)

    test_images = [Image.open(f"/home/mendax/project/CLIPPose/refers/obj_1/001_000200_000000.png")]

    clippose_similarity = CLIPPoseSimilarity(checkpoint_path=checkpoint_path)
    top_m_indices = clippose_similarity.calculate_similarity(m, refer_images, test_images)
    print(f"Top {m} similar references indices:\n{top_m_indices}")
