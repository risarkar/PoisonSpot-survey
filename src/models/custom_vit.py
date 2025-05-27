import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_vit import ViT
class CustomViT(ViT):
    """
    Vision-Transformer that auto-resizes positional embeddings
    and returns features if requested.
    """
    def __init__(self, *args, **kwargs):
        super(CustomViT, self).__init__(*args, **kwargs)
        self._resize_positional_embeddings()

    def _resize_positional_embeddings(self):
        num_patches = (224 // 16) ** 2  # 224x224 image with 16x16 patches
        seq_length = num_patches + 1  # +1 for the class token
        pos_embedding = self.positional_embedding.pos_embedding

        if seq_length != pos_embedding.size(1):
            print(f"Resizing positional embeddings from {pos_embedding.size(1)} to {seq_length}")
            self.positional_embedding.pos_embedding = nn.Parameter(
                F.interpolate(pos_embedding.unsqueeze(0), size=(seq_length, pos_embedding.size(2)), mode='nearest').squeeze(0)
            )

    def forward(self, x, return_features=False):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d 
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x[:, 0]) # b,d
            if return_features:
                return x
            x = self.fc(x)  # b,num_classes
        return x 