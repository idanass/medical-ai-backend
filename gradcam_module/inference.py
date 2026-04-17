import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image
import torchxrayvision as xrv
import torchvision.transforms as transforms
import skimage.io


class GradCAMInference:
    """
    Full pipeline: image path → predictions + Grad-CAM heatmaps.
    """

    def __init__(self, weights="densenet121-res224-nih"):
        self.model = xrv.models.DenseNet(weights=weights)
        self.model.eval()
        self.pathologies = [p for p in self.model.pathologies if p.strip() != ""]
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        layer = self.model.features.denseblock4

        def fwd(module, input, output):
            self.activations = output.detach()

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer.register_forward_hook(fwd)
        layer.register_full_backward_hook(bwd)

    def preprocess_bytes(self, image_bytes: bytes) -> tuple:
        """
        Accepts raw image bytes (from API upload).
        Returns (img_tensor, orig_img_normalized)
        """
        import numpy as np
        from PIL import Image
        import io

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
        img = np.array(pil_img).astype(np.float32)
        img = xrv.datasets.normalize(img, 255)
        img = img[None, ...]  # (1, H, W)

        transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        img = transform(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1,1,224,224)

        orig = img_tensor.squeeze().cpu().numpy()
        orig = (orig - orig.min()) / (orig.max() - orig.min())

        return img_tensor, orig

    def _generate_cam(self, img_tensor: torch.Tensor, path_idx: int) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(img_tensor.requires_grad_(True))
        out[0, path_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return cam.squeeze().numpy()

    def _overlay_to_base64(self, cam: np.ndarray, orig_img: np.ndarray) -> str:
        heatmap = cm.jet(cam)[:, :, :3]
        orig_rgb = np.stack([orig_img] * 3, axis=-1)
        overlay = (0.45 * orig_rgb + 0.55 * heatmap)
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(overlay_uint8).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def predict(self, image_bytes: bytes) -> dict:
        """
        Main method for API.
        Input  : raw image bytes
        Output : { pathology: { score, cam_base64 } }
        """
        img_tensor, orig = self.preprocess_bytes(image_bytes)

        with torch.no_grad():
            preds = self.model(img_tensor).squeeze()

        named_indices = [
            (i, p) for i, p in enumerate(self.model.pathologies)
            if p.strip() != ""
        ]

        results = {}
        for path_idx, path_name in named_indices:
            cam = self._generate_cam(img_tensor, path_idx)
            b64 = self._overlay_to_base64(cam, orig)
            results[path_name] = {
                "score": round(preds[path_idx].item(), 4),
                "cam_base64": b64
            }

        return results