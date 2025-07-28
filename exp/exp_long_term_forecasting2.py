import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer,LlamaForCausalLM
from utils.metrics import metric

warnings.filterwarnings('ignore')


def norm(input_emb):
    input_emb = input_emb - input_emb.mean(1, keepdim=True).detach()
    input_emb = input_emb / torch.sqrt(
        torch.var(input_emb, dim=1, keepdim=True, unbiased=False) + 1e-5
    )
    return input_emb


class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class CoAttention(nn.Module):
    def __init__(self, dim_kv=512, dim_q=12, seq_len=24):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.linear = nn.Linear(seq_len, dim_q)

    def forward(self, q, k, v, mask=None):
        b, q_len, dim_q = q.shape
        _, kv_len, dim_kv = k.shape

        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)

        qk = torch.bmm(q_proj, k_proj.transpose(1, 2))  # (b, 24, 5)
        a_q = F.softmax(qk, dim=1)
        a_k = F.softmax(qk, dim=2)

        c_q = torch.bmm(a_q, k_proj)  # (b, 24, 12)
        c_k = torch.bmm(
            a_k.transpose(1, 2),
            torch.cat((q, c_q), 2)
        )  # (b, 5, 24)
        attn = self.linear(c_k.transpose(1, 2))  # (b, 24, 12)
        return attn


class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, kv_dim, n_heads, output_dim, pred_len):
        super().__init__()
        assert query_dim % n_heads == 0
        assert kv_dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim_q = query_dim // n_heads

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)

    def forward(self, queries, keys, values, mask=None):
        B, L_q, _ = queries.shape
        B, L_kv, _ = keys.shape

        Q = self.q_proj(queries)
        K = self.k_proj(keys)
        V = self.v_proj(values)

        Q = Q.view(B, L_q, self.n_heads, self.head_dim_q).transpose(1, 2)
        K = K.view(B, L_kv, self.n_heads, self.head_dim_q).transpose(1, 2)
        V = V.view(B, L_kv, self.n_heads, self.head_dim_q).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim_q ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L_q, -1)

        return queries + attn_output


class CALayer(nn.Module):
    def __init__(
        self,
        embedding_dim=13,
        embedding_seq=26,
        d_model=512,
        seq_len=24,
        text_embedding_dim=12,
        pred_len=6
    ):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.pred_len = pred_len

        self.pre_emb = nn.Sequential(
            nn.Conv1d(embedding_seq, seq_len, 1, 1),
            nn.Linear(embedding_dim, d_model)
        )
        self.coattn = CoAttention(512, self.text_embedding_dim, seq_len)
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
        self.ca_fusion = nn.Sequential(nn.Conv1d(24, 1, 1, 1), nn.LeakyReLU())
        self.linear = nn.Linear(self.text_embedding_dim * 2, self.pred_len)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(1)

    def forward(self, prompt_emb, preds_prompt_emb, encoder_emb):
        prompt_emb = F.normalize(prompt_emb, p=2, dim=2)
        preds_prompt_emb = F.normalize(preds_prompt_emb, p=2, dim=2)
        encoder_emb = F.normalize(encoder_emb, p=2, dim=1)

        encoder_emb = self.pre_emb(encoder_emb)

        coattn_out = self.coattn(prompt_emb, encoder_emb, encoder_emb)
        ca1 = self.ca1(preds_prompt_emb, coattn_out, coattn_out)
        ca2 = self.ca2(coattn_out, preds_prompt_emb, preds_prompt_emb)

        fus = self.ca_fusion(torch.cat((ca1, ca2), dim=-1))
        out = self.linear(self.dropout(fus)).transpose(1, 2)
        out = self.norm(out)
        return out


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        args.task_name = 'long_term_forecast'
        super().__init__(args)

        configs = args
        self.text_path = configs.text_path
        self.prompt_weight = configs.prompt_weight
        self.prior_weight = configs.prior_weight
        self.nce_weight = configs.nce_weight
        self.nce_tau = getattr(args, 'nce_tau', 0.07)

        self.attribute = "final_sum"
        self.type_tag = configs.type_tag
        self.text_len = configs.text_len
        self.d_llm = configs.llm_dim
        self.pred_len = configs.pred_len
        self.text_embedding_dim = configs.text_emb
        self.pool_type = configs.pool_type
        self.use_fullmodel = configs.use_fullmodel
        self.hug_token = configs.huggingface_token

        mlp_sizes = [self.d_llm, int(self.d_llm / 8), self.text_embedding_dim]
        self.Doc2Vec = False
        if mlp_sizes is not None:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_sizes[0], mlp_sizes[1]),
                nn.ReLU(),
                nn.Linear(mlp_sizes[1], mlp_sizes[2]),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
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

        # instantiate CA layer
        #self.ca_layer = CALayer(
        #    embedding_dim=self.args.d_model,
        #    embedding_seq=self.args.enc_in,
        #    d_model=self.args.d_model,
        #    seq_len=self.args.seq_len,
        #    text_embedding_dim=self.text_embedding_dim,
        #    pred_len=self.pred_len
        #).to(self.device)

        
        if configs.llm_model == 'GPT2':
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
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.llm_model=self.llm_model.to(self.device)
        # weights for prior fusion (if used)
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

        self.mlp = self.mlp.to(self.device)
        self.learning_rate2 = 1e-2
        self.learning_rate3 = 1e-4

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
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag, self.llm_model, self.tokenizer)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_optimizer_mlp(self):
        lr = getattr(self.args, 'learning_rate2', self.learning_rate2)
        return optim.Adam(self.mlp.parameters(), lr=lr)

    def _select_optimizer_ca(self):
        return optim.Adam(self.ca_layer.parameters(), lr=self.learning_rate3)

    def _select_criterion(self):
        return nn.MSELoss()

    def run_one_batch(
        self,
        batch,
        data_provider,
        text_embedding=None,
        preds_text_embedding=None,
        prior_y=None,
        training=False,
        test=False,
        return_align_feats=False
    ):
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

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

        batch_x = batch_x.detach()
        dec_inp = dec_inp.detach()

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        outputs = outputs[:, :, 0].unsqueeze(-1)  # (B, T, 1)

        # CA fusion
        if self.prompt_weight > 0:
            m = self.model.module if hasattr(self.model, 'module') else self.model
            encoder_emb = m.get_encoder_embedding()
            fus = self.ca_layer(prompt_emb, preds_prompt_emb, encoder_emb)
            outputs = outputs + fus
            
        m = self.model.module if hasattr(self.model, 'module') else self.model
        raw_enc = m.get_encoder_embedding()  # (B, enc_in, d_model)
        
        # prior fusion
        if self.prior_weight > 0:
            outputs = (1 - self.prior_weight) * outputs + self.prior_weight * prior_y

        true = batch_y[:, -self.args.pred_len:, f_dim:]

        # InfoNCE features
        z_ts = z_txt = None
        if return_align_feats:
            z_txt = prompt_emb.mean(dim=1)  # (B, d)
            #enc_proj = self.ca_layer.coattn.k_proj(encoder_emb)  # (B, L_ts, d)
            proc_enc = self.ca_layer.pre_emb(raw_enc)          # (B, seq_len, d_model)
            enc_proj = self.ca_layer.coattn.k_proj(proc_enc)   # (B, seq_len, text_dim)
            z_ts = enc_proj.mean(dim=1)  # (B, d)

        if test:
            if data_provider.scale and self.args.inverse:
                outputs = data_provider.inverse_transform(outputs.squeeze(0)).reshape(outputs.shape)
                true = data_provider.inverse_transform(true.squeeze(0)).reshape(true.shape)

        if return_align_feats:
            return outputs, true, z_ts, z_txt
        return outputs, true

    def vali(self, vali_data, vali_loader, criterion, all_metric=False):
        total_loss = []
        total_mae = total_mse = total_rmse = total_mape = total_mspe = None
        if all_metric:
            total_mae, total_mse, total_rmse, total_mape, total_mspe = [], [], [], [], []

        self.model.eval()
        self.mlp.eval()
        self.ca_layer.eval()
        with torch.no_grad():
            for batch in vali_loader:
                pred, true = self.run_one_batch(batch, vali_data)
                loss = criterion(pred, true)
                total_loss.append(loss.item())
                if all_metric:
                    mae, mse, rmse, mape, mspe = metric(
                        np.array(pred.cpu()), np.array(true.cpu())
                    )
                    total_mae.append(mae)
                    total_mse.append(mse)
                    total_rmse.append(rmse)
                    total_mape.append(mape)
                    total_mspe.append(mspe)

        self.model.train()
        self.mlp.train()
        self.ca_layer.train()

        avg_loss = np.average(total_loss)
        if all_metric:
            return (
                avg_loss,
                np.average(total_mae),
                np.average(total_mse),
                np.average(total_rmse),
                np.average(total_mape),
                np.average(total_mspe),
            )
        return avg_loss

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_model = EarlyStopping(patience=self.args.patience, verbose=True)
        early_mlp = EarlyStopping(patience=self.args.patience, verbose=True)
        early_ca = EarlyStopping(patience=self.args.patience, verbose=True)

        optim_model = self._select_optimizer()
        optim_mlp = self._select_optimizer_mlp()
        optim_ca = self._select_optimizer_ca()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            self.model.train()
            self.mlp.train()
            self.ca_layer.train()

            train_losses = []
            for i, batch in enumerate(train_loader):
                optim_model.zero_grad()
                optim_mlp.zero_grad()
                optim_ca.zero_grad()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        pred, true, z_ts, z_txt = self.run_one_batch(
                            batch, train_data, training=True, return_align_feats=True
                        )
                        loss_main = criterion(pred, true)
                        loss = loss_main
                        if self.prompt_weight > 0:
                            loss_nce = self._info_nce_loss(z_ts, z_txt)
                            loss = loss + self.nce_weight * loss_nce
                        else:
                            loss_nce = torch.tensor(0.0, device=self.device)
                    scaler.scale(loss).backward()
                    scaler.step(optim_model)
                    scaler.step(optim_mlp)
                    scaler.step(optim_ca)
                    scaler.update()
                else:
                    pred, true, z_ts, z_txt = self.run_one_batch(
                        batch, train_data, training=True, return_align_feats=True
                    )
                    loss_main = criterion(pred, true)
                    loss = loss_main
                    if self.prompt_weight > 0:
                        loss_nce = self._info_nce_loss(z_ts, z_txt)
                        loss = loss + self.nce_weight * loss_nce
                    else:
                        loss_nce = torch.tensor(0.0, device=self.device)

                    loss.backward()
                    optim_model.step()
                    optim_mlp.step()
                    optim_ca.step()

                train_losses.append(loss_main.item())
                if (i + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1} Iter {i+1}/{len(train_loader)} "
                        f"Loss_main={loss_main.item():.7f} "
                        f"Loss_nce={loss_nce.item():.7f}"
                    )

            train_loss = np.mean(train_losses)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae, *_ = self.vali(test_data, test_loader, criterion, all_metric=True)

            print(
                f"Epoch {epoch+1} | Train: {train_loss:.7f} "
                f"Val: {vali_loss:.7f} Test MSE: {test_loss:.7f} MAE: {test_mae:.7f}"
            )

            early_model(vali_loss, self.model, os.path.join(path, "checkpoint_model.pth"))
            early_mlp(vali_loss, self.mlp, os.path.join(path, "checkpoint_mlp.pth"))
            early_ca(vali_loss, self.ca_layer, os.path.join(path, "checkpoint_ca_layer.pth"))
            if early_model.early_stop or early_mlp.early_stop or early_ca.early_stop:
                print("Early stopping")
                break

        # load best
        self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint_model.pth")))
        self.mlp.load_state_dict(torch.load(os.path.join(path, "checkpoint_mlp.pth")))
        self.ca_layer.load_state_dict(torch.load(os.path.join(path, "checkpoint_ca_layer.pth")))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints', setting, 'checkpoint_model.pth')))
            self.mlp.load_state_dict(torch.load(os.path.join('./checkpoints', setting, 'checkpoint_mlp.pth')))
            self.ca_layer.load_state_dict(torch.load(os.path.join('./checkpoints', setting, 'checkpoint_ca_layer.pth')))

        self.model.eval()
        self.mlp.eval()
        self.ca_layer.eval()

        preds, trues = [], []
        folder = os.path.join('./test_results', setting)
        os.makedirs(folder, exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_text = test_data.get_text(batch[-1])
                batch_text_flattened = batch_text.reshape(-1).tolist()
                batch_preds_text_flattened = test_data.get_preds_text(batch[-1]).reshape(-1).tolist()

                tokenized = self.tokenizer(
                    batch_text_flattened,
                    return_tensors="pt", padding=True, truncation=True, max_length=256
                )
                input_ids = tokenized['input_ids'].view(self.args.batch_size, self.args.seq_len, -1).to(self.device)
                attn_mask = tokenized['attention_mask'].view(self.args.batch_size, self.args.seq_len, -1).to(self.device)
                prompt_embeddings = self.llm_model.get_input_embeddings()(input_ids)

                # 1) prompt 토크나이징 → embeddings, mask 생성
                tokenized = self.tokenizer(
                    batch_text_flattened,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                input_ids = tokenized['input_ids'].view(self.args.batch_size, self.args.seq_len, -1).to(self.device)
                mask = tokenized['attention_mask'].view(self.args.batch_size, self.args.seq_len, -1).to(self.device)
                prompt_embeddings = self.llm_model.get_input_embeddings()(input_ids)

                # 2) preds 토크나이징 → embeddings, mask 생성
                tokenized_p = self.tokenizer(
                    batch_preds_text_flattened,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                input_ids_p = tokenized_p['input_ids'].view(self.args.batch_size, self.args.seq_len, -1).to(self.device)
                mask_p     = tokenized_p['attention_mask'].view(self.args.batch_size, self.args.seq_len, -1).to(self.device)
                preds_prompt_embeddings = self.llm_model.get_input_embeddings()(input_ids_p)

                # 3) full model 여부에 따라 LLM 적용
                if self.use_fullmodel:
                    prompt_emb       = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
                    preds_prompt_emb = self.llm_model(inputs_embeds=preds_prompt_embeddings).last_hidden_state
                else:
                    prompt_emb       = prompt_embeddings
                    preds_prompt_emb = preds_prompt_embeddings

                # 4) 각 스트림별로 mask 확장
                prompt_mask = mask.unsqueeze(-1).expand_as(prompt_emb)
                preds_mask  = mask_p.unsqueeze(-1).expand_as(preds_prompt_emb)

                # 5) avg pooling 예시
                if self.pool_type == "avg":
                    # prompt pooling
                    m1 = prompt_emb * prompt_mask
                    cnt1 = prompt_mask.sum(dim=2, keepdim=True).clamp(min=1)
                    prompt_emb = (m1.sum(dim=2) / cnt1.squeeze(2))

                    # preds pooling
                    m2 = preds_prompt_emb * preds_mask
                    cnt2 = preds_mask.sum(dim=2, keepdim=True).clamp(min=1)
                    preds_prompt_emb = (m2.sum(dim=2) / cnt2.squeeze(2))

                prior_y = torch.from_numpy(test_data.get_prior_y(batch[-1])).float().to(self.device)
                # 1) GPU→CPU→NumPy 변환
                pred_tensor, true_tensor = self.run_one_batch(
                    batch, test_data,
                    text_embedding=prompt_emb,
                    preds_text_embedding=preds_prompt_emb,
                    prior_y=prior_y,
                    test=True
                )
                pred_np = pred_tensor.detach().cpu().numpy()
                true_np = true_tensor.detach().cpu().numpy()

                # 2) NumPy 배열을 리스트에 저장
                preds.append(pred_np)
                trues.append(true_np)

                # 3) 시각화에도 NumPy 배열 사용
                if i % 20 == 0:
                    inp_np = batch[0].cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        inp_np = test_data.inverse_transform(inp_np.squeeze(0)).reshape(inp_np.shape)
                    gt = np.concatenate((inp_np[0, :, -1], true_np[0, :, -1]), axis=0)
                    pd = np.concatenate((inp_np[0, :, -1], pred_np[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder, f'{i}.pdf'))


        preds = np.vstack(preds)
        trues = np.vstack(trues)

        folder = os.path.join('./results', setting)
        os.makedirs(folder, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}')

        with open(self.args.save_name, 'a') as f:
            f.write(
                f"{self.args.model}\t{self.args.data_path[:-4]}\t"
                f"{self.args.llm_model}\t{self.args.seq_len}\t"
                f"{self.args.label_len}\t{self.args.pred_len}\t"
                f"{self.args.text_emb}\t{self.args.prior_weight}\t"
                f"{self.args.prompt_weight}\t"
                f"{mse}\t{mae}\t{rmse}\t{mape}\t{mspe}\n"
            )

        np.save(os.path.join(folder, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder, 'pred.npy'), preds)
        np.save(os.path.join(folder, 'true.npy'), trues)

        return mse
