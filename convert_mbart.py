import argparse
import logging
import os
import copy

from transformers import AutoTokenizer, MBartTokenizer, MBartForConditionalGeneration
# from transformers.models.bart.modeling_bart import shift_tokens_right
from longformer_encoder_decoder import LongformerEncoderDecoderForConditionalGeneration, LongformerSelfAttentionForMBart, LongformerEncoderDecoderConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_long_model(
    save_model_to,
    base_model,
    tokenizer_name_or_path,
    attention_window,
    max_pos
):
    model = MBartForConditionalGeneration.from_pretrained(base_model)
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_pos)
    config = LongformerEncoderDecoderConfig.from_pretrained(base_model)
    model.config = config

    # in BART attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here
    config.attention_probs_dropout_prob = config.attention_dropout
    config.architectures = ['LongformerEncoderDecoderForConditionalGeneration', ]

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape
    assert current_max_pos == config.max_position_embeddings + 2

    config.max_encoder_position_embeddings = max_pos
    config.max_decoder_position_embeddings = config.max_position_embeddings
    del config.max_position_embeddings
    max_pos += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
    assert max_pos >= current_max_pos

    # allocate a larger position embedding matrix for the encoder
    new_encoder_pos_embed = model.model.encoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_encoder_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[2:]
        k += step
    model.model.encoder.embed_positions.weight.data = new_encoder_pos_embed

    # allocate a larger position embedding matrix for the decoder
    # new_decoder_pos_embed = model.model.decoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # # copy position embeddings over and over to initialize the new position embeddings
    # k = 2
    # step = current_max_pos - 2
    # while k < max_pos - 1:
    #     new_decoder_pos_embed[k:(k + step)] = model.model.decoder.embed_positions.weight[2:]
    #     k += step
    # model.model.decoder.embed_positions.weight.data = new_decoder_pos_embed

    # replace the `modeling_bart.SelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers

    for i, layer in enumerate(model.model.encoder.layers):
        longformer_self_attn_for_bart = LongformerSelfAttentionForMBart(config, layer_id=i)

        longformer_self_attn_for_bart.longformer_self_attn.query = layer.self_attn.q_proj
        longformer_self_attn_for_bart.longformer_self_attn.key = layer.self_attn.k_proj
        longformer_self_attn_for_bart.longformer_self_attn.value = layer.self_attn.v_proj

        longformer_self_attn_for_bart.longformer_self_attn.query_global = copy.deepcopy(layer.self_attn.q_proj)
        longformer_self_attn_for_bart.longformer_self_attn.key_global = copy.deepcopy(layer.self_attn.k_proj)
        longformer_self_attn_for_bart.longformer_self_attn.value_global = copy.deepcopy(layer.self_attn.v_proj)

        longformer_self_attn_for_bart.output = layer.self_attn.out_proj

        layer.self_attn = longformer_self_attn_for_bart

    # save model
    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Convert BART to LongBART. Replaces BART encoder's SelfAttnetion with LongformerSelfAttention")
    parser.add_argument(
        '--base_model',
        type=str,
        default="ARTeLab/mbart-summarization-fanpage",
        help='The name or path of the base model you want to convert'
    )
    parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        default="ARTeLab/mbart-summarization-fanpage",
        help='The name or path of the tokenizer'
    )
    parser.add_argument(
        '--save_model_to',
        type=str,
        default="model", 
        help='The path to save the converted model'
    )
    parser.add_argument(
        '--attention_window',
        type=int,
        default=128,
        help='attention window size for longformer self attention (one sided)'
    )
    parser.add_argument(
        '--max_pos',
        type=int,
        default=4096,
        help='maximum encoder positions'
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    create_long_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        attention_window=args.attention_window,
        max_pos=args.max_pos
    ) 

if __name__ == "__main__":
    main()