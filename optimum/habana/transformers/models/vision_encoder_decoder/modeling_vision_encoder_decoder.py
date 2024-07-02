from transformers.utils import logging


logger = logging.get_logger(__name__)


def gaudi_VisionEncoderDecoderModel_prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
):
    """
    Copied from VideoEncoderDecoderModel.prepare_inputs_for_generation: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L645
    The only differences are:
    - add token idx support
    """
    decoder_inputs = self.decoder.prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=kwargs.get("decoder_attention_mask", None),
        token_idx=kwargs.get("token_idx", None),
    )
    decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
    input_dict = {
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "decoder_input_ids": decoder_inputs["input_ids"],
        "encoder_outputs": encoder_outputs,
        "past_key_values": decoder_inputs["past_key_values"],
        "use_cache": use_cache,
        "decoder_position_ids": decoder_inputs.get("position_ids", None),
        "decoder_token_idx": decoder_inputs.get("token_idx", None),
    }
    return input_dict
