from typing import Optional

import torch
from transformers.utils import logging


logger = logging.get_logger(__name__)


@torch.no_grad()
def gaudi_BlipForQuestionAnswering_generate(
    self,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """
    Copied from BlipForQuestionAnswering.generate: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/blip/modeling_blip.py#L1246
    The only differences are:
        - torch.full add dtype=torch.int64, or else the default type is torch.float32. lead to coredump in embeding layer
    """

    vision_outputs = self.vision_model(pixel_values=pixel_values)

    image_embeds = vision_outputs[0]

    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

    if isinstance(input_ids, list):
        input_ids = torch.LongTensor(input_ids)

    question_outputs = self.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
        return_dict=False,
    )

    question_embeds = question_outputs[0]

    question_attention_mask = torch.ones(question_embeds.size()[:-1], dtype=torch.long).to(question_embeds.device)

    bos_ids = torch.full(
        (question_embeds.size(0), 1),
        fill_value=self.decoder_start_token_id,
        device=question_embeds.device,
        dtype=torch.int64,
    )

    outputs = self.text_decoder.generate(
        input_ids=bos_ids,
        eos_token_id=self.config.text_config.sep_token_id,
        pad_token_id=self.config.text_config.pad_token_id,
        encoder_hidden_states=question_embeds,
        encoder_attention_mask=question_attention_mask,
        **generate_kwargs,
    )

    return outputs
