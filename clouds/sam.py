from torch import nn
import torch
from segment_anything import SamAutomaticMaskGenerator


class SAM(nn.Module):
    def __init__(self,
        *,
        mobile: bool,
        size_threshold: int,
        erosion: bool,
        erosion_size: int,
        num_points: int,
        selection_mode: str,
        rm_intersection: bool,
        refinement: bool,
        ):

        super().__init__()
        self.mobile = mobile
        self.sam_refinement = refinement
        self.sam_size_threshold = size_threshold
        self.sam_erosion = erosion
        self.sam_erosion_size = erosion_size
        self.sam_num_points = num_points
        self.sam_selection_mode = selection_mode
        self.sam_rm_intersection = rm_intersection

        if self.mobile:
                from mobile_sam import sam_model_registry, SamPredictor
                from mobile_sam.utils.transforms import ResizeLongestSide

                self.sam_preprocessor = ResizeLongestSide(1024)
                self.sam = sam_model_registry["vit_t"](
                    checkpoint="./weights/mobile_sam.pt"
                )
        else:
            from segment_anything import sam_model_registry, SamPredictor
            from segment_anything.utils.transforms import ResizeLongestSide

            self.sam_preprocessor = ResizeLongestSide(1024)
            self.sam = sam_model_registry["vit_h"](
                checkpoint="./weights/sam_vit_h_4b8939.pth"
            )

        self.sam_predictor = SamPredictor(self.sam)
        self.sam_mask_generator = SamAutomaticMaskGenerator(self.sam)

    def forward(self, x):
        """
        Define the forward pass for your inference model.

        Args:
            x: Input data or image tensor.

        Returns:
            output: Model's output tensor after the forward pass.
        """

        return x

    def set_torch_image(self, image, size):
        with torch.no_grad():
            self.sam_predictor.set_torch_image(image, size)

    def predict_torch(self, point_coords, point_labels, multimask_output, mask_input=None):
        # self.sam.eval()
        with torch.no_grad():
            return self.sam_predictor.predict_torch(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
                mask_input=mask_input,
            )

    def apply_image(self, image):
        with torch.no_grad():
            return self.sam_preprocessor.apply_image(image)

    def generate_mask(self, image):
        masks = self.sam_mask_generator.generate(image)
        return masks

    def apply_coords(self, coords, size):
        with torch.no_grad():
            return self.sam_preprocessor.apply_coords(coords,size)

