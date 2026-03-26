import torch
import torch.nn as nn
from config import Config


class PersonaLoss:

    def __init__(self):
        self.ce = nn.CrossEntropyLoss()

    def orthogonality_loss(self, z_p, z_c):

        # normalize vectors
        z_p_norm = torch.nn.functional.normalize(z_p, dim=1)
        z_c_norm = torch.nn.functional.normalize(z_c, dim=1)

        dot_product = (z_p_norm * z_c_norm).sum(dim=1)

        loss = torch.mean(dot_product ** 2)

        return loss

    def compute(self, z_p, z_c, persona_logits, adv_logits, labels):

        cls_loss = self.ce(persona_logits, labels)

        adv_loss = self.ce(adv_logits, labels)

        orth_loss = self.orthogonality_loss(z_p, z_c)

        total_loss = (
            cls_loss
            + Config.lambda_adv * adv_loss
            + Config.lambda_orth * orth_loss
        )

        return total_loss, cls_loss, adv_loss, orth_loss