from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer,LlamaForCausalLM
import os
import time
import warnings
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

def norm(input_emb):
    input_emb=input_emb- input_emb.mean(1, keepdim=True).detach()
    input_emb=input_emb/torch.sqrt(
        torch.var(input_emb, dim=1, keepdim=True, unbiased=False) + 1e-5)
   
    return input_emb
class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)  
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  
                x = F.relu(x)
                x = self.dropout(x)  
        return x
warnings.filterwarnings('ignore')


class CoAttention(nn.Module):
    def __init__(self, dim_kv=512, dim_q=12, seq_len=24):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, dim_q)  # project k to (b, 24, 12)
        self.k_proj = nn.Linear(dim_kv, dim_q)  # project k to (b, 5, 12)
        # self.v_proj = nn.Linear(dim_kv, dim_q)  # project v to (b, 5, 12)
        self.linear = nn.Linear(seq_len, dim_q)

    def forward(self, q, k, v, mask=None):
        # q: (b, 24, 12), k: (b, 5, 512), v: (b, 5, 512)
        b, q_len, dim_q = q.shape
        _, kv_len, dim_kv = k.shape

        # Project k, v -> (b, 5, 12)
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        # v_proj = self.v_proj(v)
        self.q_proj_emb = q_proj
        self.k_proj_emb = k_proj

        # Co-attention
        qk = torch.bmm(q_proj, k_proj.transpose(1, 2))  # (b, 24, 24)
        a_q = F.softmax(qk, dim=1)
        a_k = F.softmax(qk, dim=2)
        self.attn_map = qk
        c_q = torch.bmm(a_q, k_proj)  # (b, 24, 12)
        c_k = torch.bmm(a_k.transpose(1,2), torch.cat((q, c_q), 2)) # (b, 5, dim_q*2)
        attn = self.linear(c_k.transpose(1,2))  # (b, dim_q, 12)
        # print(attn.shape)
            
        # 논문 co-attention?? 
        # attn_scores = torch.bmm(q_proj, k_proj.transpose(1,2)) # (b, 24, 5)
        # attn_probs = F.softmax(attn_scores, dim=-1) # (b, 24, 5)
        # attn = torch.bmm(attn_probs, v_proj) # (b, 24, 12)
        # # print(q_proj.shape, k_proj.shape, v_proj.shape)
     
        return attn


class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, kv_dim, n_heads, output_dim, pred_len):
        super().__init__()
        assert query_dim % n_heads == 0, "query_dim must be divisible by n_heads"
        assert kv_dim % n_heads == 0, "kv_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim_q = query_dim // n_heads
        self.head_dim_kv = kv_dim // n_heads

        # Q, K, V projection
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)

        # Output projection
        # self.norm1 = nn.LayerNorm(query_dim)
        # self.norm2 = nn.LayerNorm(query_dim)
        # self.act1 = nn.LeakyReLU()
        # self.dropout = nn.Dropout(0.1)
        # self.out_proj = nn.Linear(query_dim, query_dim)
        # # self.out_proj2 = nn.Linear(query_dim, output_dim)
        # self.mlp = nn.Sequential(nn.Linear(query_dim, pred_len), nn.LeakyReLU())
        # self.film_net = nn.Sequential(
        #     nn.Linear(query_dim, query_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(query_dim//2, pred_len*2),  # 48 for scale + 48 for shift
        # )
        # self.pred_len = pred_len

    def forward(self, queries, keys, values, mask=None):
        B, L_q, _ = queries.shape
        B, L_kv, _ = keys.shape

        # Project Q, K, V
        Q = self.q_proj(queries)  # (B, L_q, D_q)
        K = self.k_proj(keys)     # (B, L_kv, D_q)
        V = self.v_proj(values)   # (B, L_kv, D_q)
        self.proj_q_emb = Q

        # Split into heads
        Q = Q.view(B, L_q, self.n_heads, self.head_dim_q).transpose(1, 2)  # (B, H, L_q, D_hq)
        K = K.view(B, L_kv, self.n_heads, self.head_dim_q).transpose(1, 2) # (B, H, L_kv, D_hq)
        V = V.view(B, L_kv, self.n_heads, self.head_dim_q).transpose(1, 2) # (B, H, L_kv, D_hq)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim_q ** 0.5)  # (B, H, L_q, L_kv)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L_q, L_kv)
        self.attn_map = attn_weights.mean(dim=1)  # Average attention map across heads

        attn_output = torch.matmul(attn_weights, V)  # (B, H, L_q, D_hq)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, L_q, H, D_hq)
        attn_output = attn_output.view(B, L_q, -1)  # (B, L_q, D_q)
    
        # print(f"attn_output, {attn_output.shape}")
        attn_output = queries + attn_output  # Residual connection
        
        
        
        # ## 여기서 부터는 output 처리
        # y = attn_output = self.norm1(attn_output)
        # y = self.dropout(self.act1(y))
        # attn_output = self.norm2(attn_output+y)
        
        # attn_output = self.out_proj(attn_output)
        # # print(f"attn_output, {attn_output.shape}") # attn_output, torch.Size([32, head, q_dim])
        

        # # Final projection
        # attn_out = attn_output.mean(dim=1)
        # film_params = self.film_net(attn_out)
        
        # scale, shift = film_params[:,:self.pred_len], film_params[:,self.pred_len:]
        # scale = scale.unsqueeze(-1)
        # shift = shift.unsqueeze(-1)
        # # residual = F.interpolate(attn_output.transpose(1,2), size=self.pred_len, mode="linear").transpose(1,2)
        # residual = self.mlp(attn_output)
        # residual = residual.permute(0,2,1)[:,:,:1]
        # # print("attn_output mean:", attn_output.abs().mean().item(), "residual mean:", residual.abs().mean().item())

        return attn_output

# def kl_normal(mu_q, logvar_q, mu_p, logvar_p):
#     # closed-form KL(N(q)||N(p))
#     # sum over 차원, mean over 배치
#     var_q = logvar_q.exp()
#     var_p = logvar_p.exp()
#     kld = 0.5 * ((var_q / var_p) + ((mu_p - mu_q).pow(2) / var_p) - 1 + (logvar_p - logvar_q))
#     return kld.sum(-1).mean()

def kl_normal(mu_q, logvar_q, mu_p, logvar_p):
    # KL(q || p)
    # 0.5 * sum( log(sigma_p^2/sigma_q^2) + (sigma_q^2 + (mu_q-mu_p)^2)/sigma_p^2 - 1 )
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    term1 = (logvar_p - logvar_q)
    term2 = (var_q + (mu_q - mu_p).pow(2)) / var_p
    return 0.5 * torch.sum(term1 + term2 - 1, dim=-1).mean()

class PrivatePriorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # [mu, logvar]
        )

    def forward(self, x):
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar
    
# Approximate mutual information minimization via covariance penalty
def latent_mi_loss(z):
    # z: [B, D]
    B, D = z.size()
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.t() @ z_centered) / (B - 1)
    # zero diagonal
    diag = torch.diag(torch.diag(cov))
    off_diag = cov - diag
    # penalty is sum of squared off-diagonal
    return (off_diag ** 2).sum()

class CALayer(nn.Module):
    def __init__(self,embedding_dim=13,embedding_seq = 26,d_model=512, seq_len=24,text_embedding_dim=12, pred_len=6, prior_hidden_dim=64):
        super(CALayer, self).__init__()
        self.text_embedding_dim = text_embedding_dim
        self.pred_len = pred_len
        self.flag = False

        self.pre_emb = nn.Sequential(nn.Conv1d(embedding_seq, seq_len, 1, 1), nn.Linear(embedding_dim, d_model))

        # self.mu_ts = nn.Linear(d_model, text_embedding_dim)
        # self.logvar_ts = nn.Linear(d_model, text_embedding_dim)
        # self.mu_tf = nn.Linear(text_embedding_dim, text_embedding_dim)
        # self.logvar_tf = nn.Linear(text_embedding_dim, text_embedding_dim)
        # self.mu_tp = nn.Linear(text_embedding_dim, text_embedding_dim)
        # self.logvar_tp = nn.Linear(text_embedding_dim, text_embedding_dim)

        # posterior parameter layers (q) for TF and TP
        self.mu_tf = nn.Linear(text_embedding_dim, text_embedding_dim)
        self.logvar_tf = nn.Linear(text_embedding_dim, text_embedding_dim)
        self.mu_tp = nn.Linear(text_embedding_dim, text_embedding_dim)
        self.logvar_tp = nn.Linear(text_embedding_dim, text_embedding_dim)

        # Private Prior Networks (p) for TF and TP
        self.prior_tf = PrivatePriorNetwork(input_dim=d_model,
                                            hidden_dim=prior_hidden_dim,
                                            latent_dim=text_embedding_dim)
        self.prior_tp = PrivatePriorNetwork(input_dim=text_embedding_dim,
                                            hidden_dim=prior_hidden_dim,
                                            latent_dim=text_embedding_dim)
        
        self.coattn = CoAttention(512, self.text_embedding_dim, seq_len)
        self.coattn2 = CoAttention(self.text_embedding_dim, self.text_embedding_dim, seq_len) 
        self.ca1 = CrossAttentionLayer(
            query_dim=self.text_embedding_dim,
            kv_dim=self.text_embedding_dim,
            n_heads=4,
            output_dim=self.text_embedding_dim,
            pred_len=self.pred_len
        )
        self.ca2 = CrossAttentionLayer(
            query_dim=self.text_embedding_dim,
            kv_dim=self.text_embedding_dim,
            n_heads=4,
            output_dim=self.text_embedding_dim,
            pred_len=self.pred_len
        )
        self.ca_fusion = nn.Sequential(nn.Conv1d(seq_len, 1, 1, 1), nn.LeakyReLU())
        self.linear = nn.Linear(self.text_embedding_dim*2, self.pred_len)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, prompt_emb, preds_prompt_emb, encoder_emb):
        prompt_emb = F.normalize(prompt_emb, p=2, dim=2)
        preds_prompt_emb = F.normalize(preds_prompt_emb, p=2, dim=2)
        encoder_emb = F.normalize(encoder_emb, p=2, dim=1)

        encoder_emb = self.pre_emb(encoder_emb)

        coattn_out = self.coattn(prompt_emb, encoder_emb, encoder_emb)
        # ca1 = self.ca1(preds_prompt_emb, coattn_out, coattn_out)
        # ## t-sne embedding
        # ca2 = self.ca2(coattn_out, preds_prompt_emb, preds_prompt_emb)
        
        # self.tsne_fact = self.coattn.q_proj_emb
        # self.tsne_ts = self.coattn.k_proj_emb
        # self.tsne_pred = self.ca1.proj_q_emb
        coattn_out2 = self.coattn2(preds_prompt_emb, coattn_out, coattn_out)
        
        # self.fact_ts_attn = self.coattn.attn_map
        # self.pred_co_attn = self.ca1.attn_map
        # self.co_pred_attn = self.ca2.attn_map

        # --- KL TF: align TS -> Fact latent distribution ---
        # compute TS context for prior
        h_ts = encoder_emb.mean(dim=1)
        # posterior q(z_fact | fact)
        h_f = prompt_emb.mean(dim=1)
        mu_f, logv_f = self.mu_tf(h_f), self.logvar_tf(h_f)
        z_f = self.reparameterize(mu_f, logv_f)
        # prior p(z_fact | ts)
        mu_f_prior, logv_f_prior = self.prior_tf(h_ts)
        L_TF = kl_normal(mu_f, logv_f, mu_f_prior, logv_f_prior)

        # --- KL TP: align CoAtt -> Pred latent distribution ---
        # posterior q(z_pred | pred)
        h_p = preds_prompt_emb.mean(dim=1)
        mu_p, logv_p = self.mu_tp(h_p), self.logvar_tp(h_p)
        z_p = self.reparameterize(mu_p, logv_p)
        # prior p(z_pred | coattn_out)
        h_c = coattn_out.mean(dim=1)
        mu_p_prior, logv_p_prior = self.prior_tp(h_c)
        L_TP = kl_normal(mu_p, logv_p, mu_p_prior, logv_p_prior)

        # --- MI Regularization for latent disentanglement ---
        MI_loss_f = latent_mi_loss(z_f)
        MI_loss_p = latent_mi_loss(z_p)

        fus = self.ca_fusion(torch.cat((coattn_out2, coattn_out2), dim=-1))
        out = self.linear(self.dropout(fus)).transpose(1, 2)
        out = self.norm(out)
        return out, L_TF, L_TP, MI_loss_f, MI_loss_p
    
    def get_tsne_emb(self):
        return self.tsne_fact, self.tsne_ts, self.tsne_pred

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        args.task_name = 'long_term_forecast'
        super(Exp_Long_Term_Forecast, self).__init__(args)
        configs=args
        self.text_path=configs.text_path
        self.prompt_weight=configs.prompt_weight
        self.prior_weight = configs.prior_weight

        self.nce_weight = configs.nce_weight
        self.nce_tau = getattr(args, 'nce_tau', 0.07)

        self.attribute="final_sum"
        self.type_tag=configs.type_tag
        self.text_len=configs.text_len
        self.d_llm = configs.llm_dim
        self.pred_len=configs.pred_len
        self.text_embedding_dim = configs.text_emb
        self.pool_type=configs.pool_type
        self.use_fullmodel=configs.use_fullmodel
        self.hug_token=configs.huggingface_token
        mlp_sizes=[self.d_llm,int(self.d_llm/8),self.text_embedding_dim]
        self.Doc2Vec=False
        if mlp_sizes is not None:
            # self.mlp = MLP(mlp_sizes,dropout_rate=0.3)
            self.mlp = nn.Sequential(
                nn.Linear(mlp_sizes[0], mlp_sizes[1]),
                nn.ReLU(),
                nn.Linear(mlp_sizes[1], mlp_sizes[2]),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

            # print number of parameters of self.model
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f'Number of parameters in TS model: {num_params}')
            # print number of parameters of self.mlp
            num_params_mlp = sum(p.numel() for p in self.mlp.parameters())
            print(f'Number of parameters in MLP: {num_params_mlp}')
            print(f'Total number of parameters: {num_params + num_params_mlp}')
        else:
            self.mlp = None
        if self.args.model == 'iTransformer':
            self.ca_layer = CALayer(embedding_dim = self.args.d_model, embedding_seq = 5, d_model = self.args.d_model, seq_len = self.args.seq_len, text_embedding_dim = self.text_embedding_dim, pred_len = self.pred_len).to(self.device)
        elif self.args.model == 'PatchTST':
            self.ca_layer = CALayer(embedding_dim = self.args.d_model, embedding_seq = self.args.enc_in *4, d_model = self.args.d_model, seq_len = self.args.seq_len, text_embedding_dim = self.text_embedding_dim, pred_len = self.pred_len).to(self.device)
        elif self.args.model == 'Crossformer':
            self.ca_layer = CALayer(embedding_dim = self.args.d_model, embedding_seq = self.args.enc_in, d_model = self.args.d_model, seq_len = self.args.seq_len, text_embedding_dim = self.text_embedding_dim, pred_len = self.pred_len).to(self.device)
        elif self.args.model == 'DLinear':
            self.ca_layer = CALayer(embedding_dim = self.args.pred_len, embedding_seq = self.args.enc_in, d_model = self.args.d_model, seq_len = self.args.seq_len, text_embedding_dim = self.text_embedding_dim, pred_len = self.pred_len).to(self.device)
        elif self.args.model == 'FiLM':
            self.ca_layer = CALayer(embedding_dim = self.args.enc_in, embedding_seq = self.args.seq_len, d_model = self.args.d_model, seq_len = self.args.seq_len, text_embedding_dim = self.text_embedding_dim, pred_len = self.pred_len).to(self.device)
        elif self.args.model == 'Informer': 
            self.ca_layer = CALayer(embedding_dim = self.args.d_model, embedding_seq = 13, d_model = self.args.d_model, seq_len = self.args.seq_len, text_embedding_dim = self.text_embedding_dim, pred_len = self.pred_len).to(self.device)
        elif self.args.model in ['FEDformer','Autoformer','Informer','Transformer']:
            self.ca_layer = CALayer(embedding_dim = self.args.d_model, embedding_seq = self.args.seq_len, d_model = self.args.d_model, seq_len = self.args.seq_len, text_embedding_dim = self.text_embedding_dim, pred_len = self.pred_len).to(self.device)


        self.language_to_time_series_projection = nn.Sequential(
            nn.Linear(self.d_llm, 12),
            nn.ReLU()
        ).cuda()

        if configs.llm_model == 'Doc2Vec':
            print('Cannot using Doc2Vec')
            print("Training Doc2Vec model")
            raise Exception('Doc2Vec model is not supported')
        else:
            if configs.llm_model == 'LLAMA2':
                self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
                self.llama_config.num_hidden_layers = configs.llm_layers
                self.llama_config.output_attentions = True
                self.llama_config.output_hidden_states = True
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                # except EnvironmentError:  # downloads model from HF is not already done
                except:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaModel.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                # except EnvironmentError:  # downloads the tokenizer from HF if not already done
                except:  # downloads model from HF is not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'LLAMA3':
                # Automatically load the configuration, model, and tokenizer for LLaMA-3-8B
                llama3_path = "meta-llama/Meta-Llama-3-8B-Instruct"
                cache_path = "./"

                # Load the configuration with custom adjustments
                self.config =  LlamaConfig.from_pretrained(llama3_path,token=self.hug_token,cache_dir=cache_path)

                self.config.num_hidden_layers = configs.llm_layers
                self.config.output_attentions = True
                self.config.output_hidden_states = True

                self.llm_model  = LlamaModel.from_pretrained(
                    llama3_path,
                    config=self.config,
                    token=self.hug_token,cache_dir=cache_path
                )
                self.tokenizer = AutoTokenizer.from_pretrained(llama3_path,use_auth_token=self.hug_token,cache_dir=cache_path)
            elif configs.llm_model == 'GPT2':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2M':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-medium')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )
                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2L':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-large')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )
                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2XL':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-xl')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'BERT':
                self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

                self.bert_config.num_hidden_layers = configs.llm_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True
                try:
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.bert_config,
                    )
                # except EnvironmentError:  # downloads model from HF is not already done
                except:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.bert_config,
                    )

                try:
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                # except EnvironmentError:  # downloads the tokenizer from HF if not already done
                except:  # downloads model from HF is not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            
            else:
                raise Exception('LLM model is not defined')

            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model=self.llm_model.to(self.device)
        if args.init_method == 'uniform':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.uniform_(self.weight1.weight)
            nn.init.uniform_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        elif args.init_method == 'normal':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.normal_(self.weight1.weight)
            nn.init.normal_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        else:
            raise ValueError('Unsupported initialization method')
        
        self.mlp=self.mlp.to(self.device)
        self.learning_rate2=1e-2
        self.learning_rate3=1e-4

    def _info_nce_loss(self, z_ts, z_txt, tau=None):
        if tau is None:
            tau = self.nce_tau
        z_ts = F.normalize(z_ts, dim=-1)
        z_txt = F.normalize(z_txt, dim=-1)

        logits = torch.matmul(z_ts, z_txt.t()) / tau  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i2t + loss_t2i)
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            # self.mlp = nn.DataParallel(model, device_ids=self.args.device_ids)
            # self.ca_layer = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.llm_model, self.tokenizer)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    def _select_optimizer_mlp(self):
        model_optim = optim.Adam(self.mlp.parameters(), lr=self.args.learning_rate2)
        return model_optim
    def _select_optimizer_ca(self):
        model_optim = optim.Adam([
            {"params":self.ca_layer.parameters(), "lr":1e-4}
        ], lr = self.learning_rate3)
        return model_optim
    def _select_optimizer_weight(self):
        model_optim = optim.Adam([{'params': self.weight1.parameters()},
                              {'params': self.weight2.parameters()}], lr=self.args.learning_rate_weight)
        return model_optim
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def run_one_batch(self, batch, data_provider, text_embedding=None, preds_text_embedding=None, prior_y=None, training=False, test=False, return_align_feats=False, return_tsne_emb=False):
        batch_x, batch_y, batch_x_mark, batch_y_mark, index = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        if prior_y is None:
            prior_y = torch.from_numpy(data_provider.get_prior_y(index)).float().to(self.device)

        if text_embedding is None:
            text_embedding = data_provider.get_text_embeddings(index)
            preds_text_embedding = data_provider.get_preds_text_embeddings(index)

        prompt_emb = self.mlp(text_embedding)
        preds_prompt_emb = self.mlp(preds_text_embedding)
        # print(text_embedding.shape)

        # Decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

        # batch_x is [bsz, seq_len, num_vars], prompt_emb is [bsz, seq_len, text_embedding_dim]. concatenate them in the last dimension
        # batch_x = torch.cat([batch_x, prompt_emb], dim=-1).detach()
        batch_x = batch_x.detach()

        # dec_inp is [bsz, label_len + pred_len, num_vars], where only label_len is the true data, the rest is 0
        # text_dec_inp = torch.zeros((self.args.batch_size, self.args.pred_len, self.text_embedding_dim)).to(self.device)
        # text_dec_inp = torch.cat([prompt_emb[:, :self.args.label_len, :], text_dec_inp], dim=1).float().to(self.device)
        # dec_inp = torch.cat([dec_inp, text_dec_inp], dim=-1).detach()
        dec_inp = dec_inp.detach()
        
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model.module(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple): outputs = outputs[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(outputs, tuple): outputs = outputs[0]

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        outputs = outputs[:, :, 0].unsqueeze(-1)  # (B, T, 1)

        # CA
        L_TF = L_TP = MI_loss_f = MI_loss_p = outputs.new_tensor(0.0).to(self.device)
        if self.prompt_weight > 0:
            encoder_emb = self.model.get_encoder_embedding()
            # fus = self.ca_layer(prompt_emb, preds_prompt_emb, encoder_emb)
            fus, L_TF, L_TP, MI_loss_f, MI_loss_p = self.ca_layer(prompt_emb, preds_prompt_emb, encoder_emb)
            # tsne_fact, tsne_ts, tsne_pred = self.ca_layer.get_tsne_emb()
            outputs = outputs + fus

        m = self.model.module if hasattr(self.model, 'module') else self.model
        raw_enc = m.get_encoder_embedding()  # (B, enc_in, d_model)

        if self.prior_weight > 0:
            outputs = (1 - self.prior_weight) * outputs + self.prior_weight * prior_y

        true = batch_y[:, -self.args.pred_len:, f_dim:]

        z_ts = z_txt = None
        if return_align_feats:
            z_txt = prompt_emb.mean(dim=1)  # (B, d)
            #enc_proj = self.ca_layer.coattn.k_proj(encoder_emb)  # (B, L_ts, d)
            proc_enc = self.ca_layer.pre_emb(raw_enc)          # (B, seq_len, d_model)
            enc_proj = self.ca_layer.coattn.k_proj(proc_enc)   # (B, seq_len, text_dim)
            z_ts = enc_proj.mean(dim=1)  # (B, d)

        if test==True:
            true = true.detach()
            outputs= outputs.detach()
            if data_provider.scale and self.args.inverse:
                outputs = data_provider.inverse_transform(outputs.squeeze(0)).reshape(outputs.shape)
                true = data_provider.inverse_transform(true.squeeze(0)).reshape(true.shape)

        if return_align_feats:
            return outputs, true, L_TF, L_TP, MI_loss_f, MI_loss_p, z_ts, z_txt
        # if return_tsne_emb:
        #     return outputs, true, L_TF, L_TP, MI_loss_f, MI_loss_p, tsne_fact, tsne_ts, tsne_pred
        return outputs, true, L_TF, L_TP, MI_loss_f, MI_loss_p
        

    
    def vali(self, vali_data, vali_loader, criterion, all_metric=False):
        total_loss = []
        if all_metric:
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            total_mspe = []
        self.model.eval()
        self.mlp.eval()
        self.ca_layer.eval()
        
        all_tsne_fact = []
        all_tsne_ts = []
        all_tsne_pred = []
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                pred, true, L_TF, L_TP, MI_loss_f, MI_loss_p= self.run_one_batch(batch, vali_data)
                # tsne_fact = tsne_fact.detach().cpu()
                # tsne_ts = tsne_ts.detach().cpu()
                # tsne_pred = tsne_pred.detach().cpu()

                pred = pred.detach().cpu()
                true = true.detach().cpu()

                # all_tsne_fact.append(tsne_fact)
                # all_tsne_ts.append(tsne_ts)
                # all_tsne_pred.append(tsne_pred)
                # pred_co_attn = self.ca_layer.pred_co_attn
                # co_pred_attn = self.ca_layer.co_pred_attn
                # fact_ts_attn = self.ca_layer.fact_ts_attn

                loss = criterion(pred, true)
                if all_metric:
                    mae, mse, rmse, mape, mspe = metric(np.array(pred), np.array(true))
                    total_mae.append(mae)
                    total_mse.append(mse)
                    total_rmse.append(rmse)
                    total_mape.append(mape)
                    total_mspe.append(mspe)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        self.mlp.train()
        self.ca_layer.train()
        
        # tsne_facts = torch.cat(all_tsne_fact, dim=0)
        # tsne_tss = torch.cat(all_tsne_ts, dim=0)
        # tsne_preds = torch.cat(all_tsne_pred, dim=0)
        # print(tsne_facts.shape, tsne_tss.shape, tsne_preds.shape)
        # self.tsne_img = self.get_tsne_plot_image(tsne_facts, tsne_tss, tsne_preds)
        # self.pred_co_attn_img = self.get_attention_map_images(fact_ts_attn)
        
        
        if all_metric:
            total_mae = np.average(total_mae)
            total_mse = np.average(total_mse)
            total_rmse = np.average(total_rmse)
            total_mape = np.average(total_mape)
            total_mspe = np.average(total_mspe)
            return total_loss, total_mae, total_mse, total_rmse, total_mape, total_mspe
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_optim_mlp = self._select_optimizer_mlp()
        model_optiom_ca = self._select_optimizer_ca()

        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.mlp.train()
            self.ca_layer.train()
            epoch_time = time.time()
            self.epoch = epoch + 1
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                model_optim_mlp.zero_grad()
                model_optiom_ca.zero_grad()
                # pred, true = self.run_one_batch(batch, train_data, training=True)
                pred, true, L_TF, L_TP, MI_loss_f, MI_loss_p, z_ts, z_txt = self.run_one_batch(batch, train_data, training=True, return_align_feats=True)

                loss = criterion(pred, true)
                loss = loss + (L_TF + L_TP) * self.args.kl_weight
                # loss = loss + 0.1 * (MI_loss_f + MI_loss_p)
                loss_nce = self._info_nce_loss(z_ts, z_txt)
                loss = loss + self.nce_weight * loss_nce
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | loss nce: {2:.7f}".format(i + 1, epoch + 1, loss.item(), loss_nce.item()) )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # for name, param in self.ca_layer.named_parameters():
                    #     if param.grad is not None:
                    #         # print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
                    #         continue
                    #     else:
                    #         print(f"{name}: No gradient")
                    model_optim.step()
                    model_optim_mlp.step()
                    model_optiom_ca.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae, _, _, _, _ = self.vali(test_data, test_loader, criterion, all_metric=True)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss (MSE): {4:.7f} Test MAE: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, test_mae))
            flag = early_stopping(vali_loss, self.model, os.path.join(path, "checkpoint_model.pth"))
            flag = early_stopping(vali_loss, self.mlp, os.path.join(path, "checkpoint_mlp.pth"))
            flag = early_stopping(vali_loss, self.ca_layer, os.path.join(path, "checkpoint_ca_layer.pth"))
            if flag == True:
                save_path = f"./tsne/{self.args.data_path[:-4]}_{self.args.model}_{self.args.pred_len}/"
                os.makedirs(save_path, exist_ok=True)
                # self.tsne_img.save(save_path + f'tsne_plot_{self.epoch}.png')
                # self.pred_co_attn_img[-1].save(save_path + f'pred_co_attn_{self.epoch}.png')
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint_model.pth')))
        self.mlp.load_state_dict(torch.load(os.path.join(path, 'checkpoint_mlp.pth')))
        self.ca_layer.load_state_dict(torch.load(os.path.join(path, 'checkpoint_ca_layer.pth')))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint_model.pth')))
            self.mlp.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint_mlp.pth')))
            self.ca_layer.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint_ca_layer.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.mlp.eval()
        self.ca_layer.eval()
        
        all_tsne_fact = []
        all_tsne_ts = []
        all_tsne_pred = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_text = test_data.get_text(batch[-1])  # index
                batch_text_flattened = batch_text_flattened = batch_text.reshape(-1).tolist()
                batch_preds_text_flattened = test_data.get_preds_text(batch[-1]).reshape(-1).tolist()

                if self.Doc2Vec==False:
                    tokenized_output = self.tokenizer(
                        batch_text_flattened,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256
                    )
                    language_max_len = tokenized_output['input_ids'].shape[1]
                    input_ids = tokenized_output['input_ids'].view(self.args.batch_size, self.args.seq_len, language_max_len).to(self.device)
                    attn_mask = tokenized_output['attention_mask'].view(self.args.batch_size, self.args.seq_len, language_max_len).to(self.device)
                    prompt_embeddings = self.llm_model.get_input_embeddings()(input_ids)
                    
                    tokenized_output2 = self.tokenizer(
                        batch_preds_text_flattened,
                        return_tensors="pt",    
                        padding=True,
                        truncation=True,
                        max_length=256
                    )
                    language_max_len2 = tokenized_output2['input_ids'].shape[1]
                    input_ids2 = tokenized_output2['input_ids'].view(self.args.batch_size, self.args.seq_len, language_max_len2).to(self.device)
                    attn_mask2 = tokenized_output2['attention_mask'].view(self.args.batch_size, self.args.seq_len, language_max_len2).to(self.device)
                    preds_prompt_embeddings = self.llm_model.get_input_embeddings()(input_ids2)

                else:
                    prompt = batch_text
                    prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(self.device)
                if self.use_fullmodel:
                    prompt_emb =self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
                    preds_prompt_emb = self.llm_model(inputs_embeds=preds_prompt_embeddings).last_hidden_state
                else:
                    prompt_emb=prompt_embeddings
                    preds_prompt_emb = preds_prompt_embeddings

                if self.Doc2Vec == False:
                    # Expand attn_mask to match prompt_emb dimensions
                    expanded_mask = attn_mask.unsqueeze(-1).expand_as(prompt_emb)
                    expanded_mask2 = attn_mask2.unsqueeze(-1).expand_as(preds_prompt_emb)

                    if self.pool_type == "avg":
                        # Mask the embeddings by setting padded tokens to 0
                        masked_emb = prompt_emb * expanded_mask
                        valid_counts = expanded_mask.sum(dim=2, keepdim=True).clamp(min=1)
                        pooled_emb = masked_emb.sum(dim=2) / valid_counts.squeeze(2)
                        prompt_emb = pooled_emb
                        
                        masked_emb = preds_prompt_emb * expanded_mask2
                        valid_counts = expanded_mask2.sum(dim=2, keepdim=True).clamp(min=1)
                        pooled_emb = masked_emb.sum(dim=2) / valid_counts.squeeze(2)
                        preds_prompt_emb = pooled_emb

                    elif self.pool_type == "max":
                        # Mask the embeddings by setting padded tokens to a very small value
                        masked_emb = prompt_emb.masked_fill(expanded_mask == 0, float('-inf'))
                        pooled_emb, _ = masked_emb.max(dim=2)
                        prompt_emb = pooled_emb

                    elif self.pool_type == "min":
                        # Mask the embeddings by setting padded tokens to a very large value
                        masked_emb = prompt_emb.masked_fill(expanded_mask == 0, float('inf'))
                        pooled_emb, _ = masked_emb.min(dim=2)
                        prompt_emb = pooled_emb
                else:
                    prompt_emb = prompt_emb
                    preds_prompt_emb = preds_prompt_emb
                # prompt_emb = self.mlp(prompt_emb)

                prior_y = torch.from_numpy(test_data.get_prior_y(batch[-1])).float().to(self.device)

                pred, true, L_TF, L_TP, MI_loss_f, MI_loss_p= self.run_one_batch(batch, test_data, text_embedding=prompt_emb, preds_text_embedding=preds_prompt_emb,prior_y=prior_y, test=True)

                pred = pred.detach().cpu().numpy()
                true = true.detach().cpu().numpy()
                # tsne_fact = tsne_fact.detach().cpu()
                # tsne_ts = tsne_ts.detach().cpu()
                # tsne_pred = tsne_pred.detach().cpu()

                preds.append(pred)
                trues.append(true)
                # all_tsne_fact.append(tsne_fact)
                # all_tsne_ts.append(tsne_ts)
                # all_tsne_pred.append(tsne_pred)
                if i % 20 == 0:
                    input = batch[0].detach().cpu().numpy() # batch_x
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        
        dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open(self.args.save_name, 'a')
        # f.write(setting + "  \n")
        f.write(f'{self.args.model}\t{self.args.data_path[:-4]}\t{self.args.llm_model}\t{self.args.seq_len}\t{self.args.label_len}\t{self.args.pred_len}\t{self.args.text_emb}\t{self.args.prior_weight}\t{self.args.prompt_weight}\t{self.args.nce_weight}\t{self.args.kl_weight}\t')
        f.write('{}\t{}\t{}\t{}\t{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        # f.write('\n')
        f.close()
        
        # tsne_facts = torch.cat(all_tsne_fact, dim=0)
        # tsne_tss = torch.cat(all_tsne_ts, dim=0)
        # tsne_preds = torch.cat(all_tsne_pred, dim=0)
        # tsne_img = self.get_tsne_plot_image(tsne_facts, tsne_tss, tsne_preds, labels=None)
        # tsne_img.save(f'tsne_plot_{self.epoch}_test.png')
        
        

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mse
    def get_tsne_plot_image(self,z_f, z_t, z_p, labels=None):
        # Step 1: stack all data
        def to_numpy_2d(x):
            x = x.mean(dim=1)
            if isinstance(x, torch.Tensor):
                x = F.normalize(x, dim=-1)
                x = x.detach().cpu().numpy()
            
            return x.reshape(x.shape[0], -1)  # (B, S, D) → (B×S, D)

        z_f = to_numpy_2d(z_f)
        z_t = to_numpy_2d(z_t)
        z_p = to_numpy_2d(z_p)
    
        all_data = np.concatenate([z_f, z_p, z_t], axis=0)  # shape: (N_total, D)

        # Step 2: create group labels: 0=fact, 1=ts, 2=pred
        labels = (
            [0] * len(z_f) +
            [1] * len(z_p) +
            [2] * len(z_t)
        )
        labels = np.array(labels)

        # Step 3: run t-SNE on the full set
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        all_tsne = tsne.fit_transform(all_data)

        # Step 4: plot single scatter with group color
        colors = ['blue', 'green', 'red']
        names = ['Text(fact)', 'Text(predict)', 'TS']

        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(3):
            idx = labels == i
            ax.scatter(all_tsne[idx, 0], all_tsne[idx, 1], c=colors[i], label=names[i], alpha=0.6, s=15)

        # ax.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1.02, 0.5))
        # ax.set_title("Single t-SNE: text(fact), text(predict), ts",fontsize=20, pad=30)
        ax.axis('off')

        # Convert to PIL image
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)

        return Image.fromarray(img)
    def get_attention_map_images(self, attn_tensor, vmin=None, vmax=None, cmap='viridis'):
        """
        Convert (B, L, D) attention maps into PIL images and return as a list.
        
        Args:
            attn_tensor: torch.Tensor or np.ndarray of shape (B, L, D)
            vmin, vmax: min/max values for colormap normalization (optional)
            cmap: colormap used for visualization (default: viridis)
            
        Returns:
            List[PIL.Image.Image]: attention map images for each batch
        """
        if isinstance(attn_tensor, torch.Tensor):
            attn_tensor = attn_tensor.detach().cpu().numpy()
        
        B, L, D = attn_tensor.shape
        images = []

        for b in range(B):
            fig, ax = plt.subplots(figsize=(6, 6))
            # norm_map = (attn_tensor[b] - attn_tensor[b].min()) / (attn_tensor[b].max() - attn_tensor[b].min() + 1e-6)

            plt.imshow(attn_tensor[b], cmap='viridis', vmin=0.0, vmax=1.0, interpolation='nearest')
            ax.axis('off')  # remove axis for clean image
            plt.tight_layout(pad=0)
            plt.title(f"Attention Map (sample {b})")
            plt.colorbar()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            images.append(img)
            plt.close()

        return images