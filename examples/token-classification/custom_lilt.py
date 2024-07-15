import torch
import torch.nn as nn
import math

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode)

class CustomLiltSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.layout_query = nn.Linear(
            config.hidden_size // config.channel_shrink_ratio, self.all_head_size // config.channel_shrink_ratio
        )
        self.layout_key = nn.Linear(
            config.hidden_size // config.channel_shrink_ratio, self.all_head_size // config.channel_shrink_ratio
        )
        self.layout_value = nn.Linear(
            config.hidden_size // config.channel_shrink_ratio, self.all_head_size // config.channel_shrink_ratio
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.channel_shrink_ratio = config.channel_shrink_ratio
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None

    def transpose_for_scores(self, x, r=1):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size // r)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        layout_inputs,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        use_fused_sdpa=False,  # Add use_fused_sdpa parameter for Fused SDPA
    ):
        layout_value_layer = self.transpose_for_scores(self.layout_value(layout_inputs), r=self.channel_shrink_ratio)
        layout_key_layer = self.transpose_for_scores(self.layout_key(layout_inputs), r=self.channel_shrink_ratio)
        layout_query_layer = self.transpose_for_scores(self.layout_query(layout_inputs), r=self.channel_shrink_ratio)

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.fused_scaled_dot_product_attention and use_fused_sdpa:
            import habana_frameworks.torch.hpu as ht

            use_recompute = not self.training
            with ht.sdp_kernel(enable_recompute=use_recompute):
                attention_scores = self.fused_scaled_dot_product_attention(
                    query_layer, key_layer, value_layer, attention_mask, self.dropout, False, 1.0, "fast"
                )
                layout_attention_scores = self.fused_scaled_dot_product_attention(
                    layout_query_layer, layout_key_layer, layout_value_layer, attention_mask, self.dropout, False, 1.0, "fast"
                )
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            layout_attention_scores = torch.matmul(layout_query_layer, layout_key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        tmp_attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        tmp_layout_attention_scores = layout_attention_scores / math.sqrt(
            self.attention_head_size // self.channel_shrink_ratio
        )
        attention_scores = tmp_attention_scores + tmp_layout_attention_scores
        layout_attention_scores = tmp_layout_attention_scores + tmp_attention_scores

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            layout_attention_scores = layout_attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        layout_attention_probs = nn.Softmax(dim=-1)(layout_attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        layout_attention_probs = self.dropout(layout_attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            layout_attention_probs = layout_attention_probs * head_mask

        layout_context_layer = torch.matmul(layout_attention_probs, layout_value_layer)

        layout_context_layer = layout_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = layout_context_layer.size()[:-2] + (self.all_head_size // self.channel_shrink_ratio,)
        layout_context_layer = layout_context_layer.view(*new_context_layer_shape)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            ((context_layer, layout_context_layer), attention_probs)
            if output_attentions
            else ((context_layer, layout_context_layer),)
        )

        return outputs
