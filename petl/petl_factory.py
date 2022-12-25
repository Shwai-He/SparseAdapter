import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import RobertaConfig, BertConfig

def init_lisa_params(module):
    std = 1e-20
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def init_bias_mlp(module):
    std = 1e-2
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()


def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def init_zero_weights(module):
    if isinstance(module, nn.Embedding):
        nn.init.constant_(module.weight, 0.0)

class PrefixForAdapter(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        if isinstance(config, (RobertaConfig, BertConfig)):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn if not (args.attn_mode == 'adapter' and args.attn_option == 'concat') \
                        else config.mid_dim
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = torch.arange(self.attn_bn).long()
        self.wte = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte_enc = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte2 = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # if args.lisa_option == "cross_attn":
        #     self.apply(init_lisa_params)

    def forward(self, bsz, nsamples=1, device="gpu"):
        old_bsz = bsz
        bsz, seqlen = bsz * nsamples, self.mid_dim
        result = []
        for i in range(self.match_n_layer):
            temp_dict = {'self': {
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(device) #bsz, attn_bn
                                  },
                         }
            temp_dict['encoder_decoder'] = {
                                            "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(device)
                                            }
            # at generation time, this is expanded automatically to the beam size
            temp_dict['encoder'] = {
                                    "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(device)
                                    }
            result.append(temp_dict)
        return result


class Prefix(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        if isinstance(config, (RobertaConfig, BertConfig)):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn if not (args.attn_mode == 'adapter' and args.attn_option == 'concat') \
                        else config.mid_dim
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = torch.arange(self.attn_bn).long()
        self.wte = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte_enc = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte2 = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # if args.lisa_option == "cross_attn":
        #     self.apply(init_lisa_params)

    def forward(self, bsz, nsamples=1, device="gpu"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(device)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_value": key_val[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device) #bsz, attn_bn
                                  },
                         }

            key_val2 = past_key_values2[i]
            temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_value": key_val2[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device)
                                            }
            key_val_enc = past_key_values_enc[i]
            # at generation time, this is expanded automatically to the beam size
            temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_value": key_val_enc[1].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device)
                                    }
            result.append(temp_dict)
        return result


class PrefixCrossAttn(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        if isinstance(config, RobertaConfig):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = torch.arange(self.attn_bn).long()
        self.wte = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte2 = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # if args.lisa_option == "cross_attn":
        #     self.apply(init_lisa_params)

    def forward(self, bsz, nsamples=1, device="gpu"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'encoder_decoder': {"prev_key": key_val[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_value": key_val[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device) #bsz, attn_bn
                                  },
                         }
            key_val2 = past_key_values2[i]
            temp_dict['self'] = {"prev_key": key_val2[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                        "prev_value": key_val2[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                        "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device)
                                        }
            result.append(temp_dict)
        return result


class PrefixDirectInit(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        if isinstance(config, RobertaConfig):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)
        self.input_tokens = torch.arange(self.attn_bn).long()
        self.encoder_attn_key = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                for _ in range(self.match_n_layer)])
        self.encoder_attn_value = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                               for _ in range(self.match_n_layer)])
        self.decoder_self_attn_key = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                     for _ in range(self.match_n_layer)])
        self.decoder_self_attn_value = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                    for _ in range(self.match_n_layer)])

        self.decoder_cross_attn_key = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                      for _ in range(self.match_n_layer)])
        self.decoder_cross_attn_value = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                     for _ in range(self.match_n_layer)])

        # fixme: choose a favorable init method
        self.apply(init_bert_weights)

    def _shape(self, x, bsz):
        y = x.view(bsz, self.attn_bn, self.match_n_head, self.match_n_embd)
        y = y.permute([0, 2, 1, 3])
        y = y.contiguous().view(bsz * self.match_n_head, -1, self.match_n_embd)
        return y

    def forward(self, bsz, nsamples=1, device="cuda"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(device)

        result = []
        for i, (enc_attn_k, enc_attn_v, dec_self_attn_k, dec_self_attn_v, dec_xattn_k, dec_xattn_v) in \
                enumerate(zip(self.encoder_attn_key, self.encoder_attn_value, self.decoder_self_attn_key,
                              self.decoder_self_attn_value, self.decoder_cross_attn_key, self.decoder_cross_attn_value)):
            temp_dict = {'self': {"prev_key": self._shape(dec_self_attn_k(input_tokens), bsz),
                                  "prev_value": self._shape(dec_self_attn_v(input_tokens), bsz),
                                  "prev_key_padding_mask": torch.zeros(bsz, self.attn_bn).to(device) #bsz, attn_bn
                                  },
                         'encoder_decoder': {"prev_key": self._shape(dec_xattn_k(input_tokens), bsz),
                                  "prev_value": self._shape(dec_xattn_v(input_tokens), bsz),
                                  "prev_key_padding_mask": torch.zeros(bsz, self.attn_bn).to(device)  #bsz, attn_bn
                                  },
                         'encoder': {"prev_key": self._shape(enc_attn_k(input_tokens_enc), old_bsz),
                                  "prev_value": self._shape(enc_attn_v(input_tokens_enc), old_bsz),
                                  "prev_key_padding_mask": torch.zeros(old_bsz, self.attn_bn).to(device) #bsz, attn_bn
                                  },
                        }
            result.append(temp_dict)
        return result


class MLP_Bias(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        if isinstance(config, RobertaConfig):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.src_len = config.max_source_length + 2
        self.tgt_len = config.max_target_length + 2
        self.tgt_input_tokens = torch.arange(self.tgt_len).long()
        self.src_input_tokens = torch.arange(self.src_len).long()

        self.wte = nn.Embedding(self.tgt_len, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * self.n_embd))

        self.wte_enc = nn.Embedding(self.src_len, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * self.n_embd))

        self.wte2 = nn.Embedding(self.tgt_len, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * self.n_embd))

        self.apply(init_bias_mlp)
        # # initialization
        # nn.init.constant_(self.wte.weight, 0.0)
        # nn.init.constant_(self.wte_enc.weight, 0.0)
        # nn.init.constant_(self.wte2.weight, 0.0)

    def forward(self, bsz, nsamples=1, device="cuda"):
        temp_control = self.wte(self.tgt_input_tokens.to(device))
        past_key_values = self.control_trans(temp_control)  # tgt_len, layer*emb
        past_key_values = past_key_values.view(self.tgt_len, self.match_n_layer, self.n_embd)
        past_key_values = self.dropout(past_key_values)

        temp_control2 = self.wte2(self.tgt_input_tokens.to(device))
        past_key_values2 = self.control_trans2(temp_control2)  # tgt_len, layer*emb
        past_key_values2 = past_key_values2.view(self.tgt_len, self.match_n_layer, self.n_embd)
        past_key_values2 = self.dropout(past_key_values2)

        temp_control_enc = self.wte_enc(self.src_input_tokens.to(device))
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # src_len, layer*emb
        past_key_values_enc = past_key_values_enc.view(self.src_len, self.match_n_layer, self.n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)

        result = []
        for ii in range(self.match_n_layer):
            temp_dict = {"encoder": past_key_values_enc[:, ii, :],
                         "self": past_key_values[:, ii, :],
                         "encoder_decoder": past_key_values2[:, ii, :]}
            result.append(temp_dict)
        return result


class Bias(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.match_n_layer = config.num_hidden_layers
        self.n_embd = config.d_model

        # Option 1: a simple version, no transformations, each attention layer has its own bias parameters
        self.encoder_attn_bias = nn.ModuleList([nn.Embedding(args.max_source_length + 2, self.n_embd)
                                                for _ in range(self.match_n_layer)])
        self.decoder_self_attn_bias = nn.ModuleList([nn.Embedding(args.max_target_length + 2, self.n_embd)
                                                     for _ in range(self.match_n_layer)])

        self.decoder_cross_attn_bias = nn.ModuleList([nn.Embedding(args.max_target_length + 2, self.n_embd)
                                                      for _ in range(self.match_n_layer)])
        for embed in self.encoder_attn_bias:
            assert isinstance(embed, nn.Embedding)
            nn.init.constant_(embed.weight, 0.0)
        for embed in self.decoder_self_attn_bias:
            assert isinstance(embed, nn.Embedding)
            nn.init.constant_(embed.weight, 0.0)
        for embed in self.decoder_cross_attn_bias:
            assert isinstance(embed, nn.Embedding)
            nn.init.constant_(embed.weight, 0.0)

    def forward(self, bsz, nsamples=1, device="cuda"):
        result = []
        max_src_len = self.args.max_source_length + 2
        max_tgt_len = self.args.max_target_length + 2

        src_positions = torch.arange(0, max_src_len, dtype=torch.long, device=device)
        tgt_positions = torch.arange(0, max_tgt_len, dtype=torch.long, device=device)
        for ii in range(self.match_n_layer):
            temp_dict = {"encoder": self.encoder_attn_bias[ii].forward(src_positions),
                         "self": self.decoder_self_attn_bias[ii].forward(tgt_positions),
                         "encoder_decoder": self.decoder_cross_attn_bias[ii].forward(tgt_positions)}
            result.append(temp_dict)
        return


class Adapter_Layer(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.config = config
        self.n_embd = config.hidden_size if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.register_buffer('down_mask', torch.ones(self.down_proj.weight.shape))
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.register_buffer('up_mask', torch.ones(self.up_proj.weight.shape))
        self.non_linear_func = nn.ReLU()

        self.dropout = dropout
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = torch.matmul(x, (self.down_proj.weight * self.down_mask).transpose(-2, -1)) + self.down_proj.bias
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = torch.matmul(down, (self.up_proj.weight * self.up_mask).transpose(-2, -1)) + self.up_proj.bias
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

    def sparseFunction(self, x, s, activation=torch.relu, f=torch.sigmoid):
        return torch.sign(x) * activation(torch.abs(x) - f(100*s - 99*s.detach()))

    def reverseSigmoid(self, x):
        return - torch.log(1 / x - 1)


def softmax_gating(logits_1, logits_2):
    # the last two dimensions of logits is (T, S)
    max_logits = torch.max(torch.cat([logits_1, logits_2], dim=-1), dim=-1, keepdim=True)[0]
    logits_1 = logits_1 - max_logits
    logits_2 = logits_2 - max_logits
    exp_logits_1 = logits_1.exp()
    exp_logits_2 = logits_2.exp()
    s = torch.sum(exp_logits_1, dim=-1) + torch.sum(exp_logits_2, dim=-1)
    w1 = torch.sum(exp_logits_1, dim=-1) / s
    w2 = torch.sum(exp_logits_2, dim=-1) / s  # bsz x num_heads, tgt_len

    return w1.unsqueeze(-1), w2.unsqueeze(-1)



def adapter_func(x, down_w, up_w, layernorm, training, dropout=0.0, add_residual=True, layernorm_option="in"):
    residual = x

    if layernorm is not None and layernorm_option == "in":
        x = layernorm(x)
    # print("x", x.size())
    # print(x)
    # input()
    # print("down w", down_w.size())
    # print(down_w)
    # input()
    down = x @ down_w
    # print("down", down.size())
    # print(down)
    # input()
    down = nn.functional.relu(down)
    down = nn.functional.dropout(down, p=dropout, training=training)
    up = down @ up_w

    if layernorm is not None and layernorm_option == "out":
        up = layernorm(x)
    if add_residual:
        output = up + residual
    else:
        output = up
    return output

# copied from LoRA: https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            lora_init: str="bert",
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.lora_init = lora_init
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.ef_lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.ef_lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.register_buffer('A_mask', torch.ones(self.ef_lora_A.shape))
            self.register_buffer('B_mask', torch.ones(self.ef_lora_B.shape))
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'ef_lora_A'):
            if self.lora_init == "bert":
                nn.init.normal_(self.ef_lora_A, std=0.02)
                nn.init.normal_(self.ef_lora_B, std=0.02)
            elif self.lora_init == "lora":
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.ef_lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.ef_lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T((self.ef_lora_B * self.B_mask) @ (self.ef_lora_A * self.A_mask)) * self.scaling
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T((self.ef_lora_B * self.B_mask) @ (self.ef_lora_A * self.A_mask)) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ (self.ef_lora_A * self.A_mask).T @ (self.ef_lora_B * self.B_mask).T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
