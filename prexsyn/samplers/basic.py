import torch

from prexsyn.data.struct import PropertyRepr, SynthesisRepr
from prexsyn.models.outputs.synthesis import Prediction
from prexsyn.models.prexsyn import PrexSyn
from prexsyn_engine.featurizer.synthesis import PostfixNotationTokenDef

from .builder import SynthesisReprBuilder


class BasicSampler:
    def __init__(
        self,
        model: PrexSyn,
        token_def: PostfixNotationTokenDef,
        t_types: float = 1.0,
        t_bb: float = 1.0,
        t_rxn: float = 1.0,
        max_length: int = 16,
    ) -> None:
        super().__init__()
        self.model = model
        self.token_def = token_def
        self.t_types = t_types
        self.t_bb = t_bb
        self.t_rxn = t_rxn
        self.max_length = max_length

    def _create_builder(self, batch_size: int) -> SynthesisReprBuilder:
        return SynthesisReprBuilder(
            batch_size=batch_size,
            device=self.model.device,
            bb_token=self.token_def.BB,
            rxn_token=self.token_def.RXN,
            pad_token=self.token_def.PAD,
            start_token=self.token_def.START,
            end_token=self.token_def.END,
        )

    def _sample_from_logits(self, prediction: Prediction, ended: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_shape = prediction.type_logits.shape[:-1]
        ended = ended.flatten()

        type_logits = prediction.type_logits.flatten(0, -2) / self.t_types
        next_type = torch.multinomial(torch.softmax(type_logits, dim=-1), num_samples=1).squeeze(-1)
        next_type = torch.where(ended, self.token_def.PAD, next_type)

        next_bb = torch.zeros_like(next_type)
        bb_pos = (next_type == self.token_def.BB).nonzero(as_tuple=True)[0]
        if bb_pos.numel() > 0:
            bb_logits = prediction.bb_logits.flatten(0, -2)[bb_pos] / self.t_bb
            next_bb_subset = torch.multinomial(torch.softmax(bb_logits, dim=-1), num_samples=1).squeeze(-1)
            next_bb[bb_pos] = next_bb_subset

        next_rxn = torch.zeros_like(next_type)
        rxn_pos = (next_type == self.token_def.RXN).nonzero(as_tuple=True)[0]
        if rxn_pos.numel() > 0:
            rxn_logits = prediction.rxn_logits.flatten(0, -2)[rxn_pos] / self.t_rxn
            next_rxn_subset = torch.multinomial(torch.softmax(rxn_logits, dim=-1), num_samples=1).squeeze(-1)
            next_rxn[rxn_pos] = next_rxn_subset

        return {
            "token_types": next_type.reshape(batch_shape),
            "bb_indices": next_bb.reshape(batch_shape),
            "rxn_indices": next_rxn.reshape(batch_shape),
        }

    def sample(self, property_repr: PropertyRepr) -> SynthesisRepr:
        with torch.no_grad():
            e_property = self.model.embed_properties(property_repr)
            batch_size = e_property.batch_size
            builder = self._create_builder(batch_size)
            for _ in range(self.max_length):
                e_synthesis = self.model.embed_synthesis(builder.get())
                h_syn = self.model.encode(e_property, e_synthesis)

                prediction = self.model.predict(h_syn[..., -1:, :])
                next = self._sample_from_logits(prediction, ended=builder.ended)
                builder.append(**next)
        return builder.get()
