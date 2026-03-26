import torch
import torch.nn as nn
from transformers import AutoModel
from config import Config


# -----------------------------
# Gradient Reversal Layer
# -----------------------------
class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradientReversal.apply(x, lambda_)


# -----------------------------
# Model
# -----------------------------
class PersonaDisentangleModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Load BERT
        self.encoder = AutoModel.from_pretrained(Config.model_name)

        hidden = Config.hidden_size

        # Projection layers
        self.persona_proj = nn.Linear(hidden, Config.persona_dim)
        self.content_proj = nn.Linear(hidden, Config.content_dim)

        # Classifiers
        self.persona_classifier = nn.Linear(Config.persona_dim, Config.num_classes)
        self.adv_classifier = nn.Linear(Config.content_dim, Config.num_classes)

        # -------- Freeze lower BERT layers --------
        # Do this AFTER encoder is created
        for name, param in self.encoder.named_parameters():
            if "encoder.layer." in name:
                layer_id = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_id < 8:
                    param.requires_grad = False


    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0]

        z_p = self.persona_proj(cls_embedding)
        z_c = self.content_proj(cls_embedding)

        persona_logits = self.persona_classifier(z_p)

        z_c_rev = grad_reverse(z_c, Config.lambda_adv)
        adv_logits = self.adv_classifier(z_c_rev)

        return z_p, z_c, persona_logits, adv_logits