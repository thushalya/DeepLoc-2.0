from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attr_prior import *
from src.constants import *
from src.utils import *

class AttentionHead(nn.Module):
      def __init__(self, hidden_dim, n_heads):
          super(AttentionHead, self).__init__()
          self.n_heads = n_heads
          self.hidden_dim = hidden_dim
          self.preattn_ln = nn.LayerNorm(hidden_dim//n_heads)
          self.Q = nn.Linear(hidden_dim//n_heads, n_heads, bias=False)
          torch.nn.init.normal_(self.Q.weight, mean=0.0, std=1/(hidden_dim//n_heads))

      def forward(self, x, np_mask, lengths):
          # input (batch, seq_len, embed)
          n_heads = self.n_heads
          hidden_dim = self.hidden_dim
          x = x.view(x.size(0), x.size(1), n_heads, hidden_dim//n_heads)
          x = self.preattn_ln(x)
          mul = (x * \
                self.Q.weight.view(1, 1, n_heads, hidden_dim//n_heads)).sum(-1) \
                #* np.sqrt(5)
                #/ np.sqrt(hidden_dim//n_heads)
          mul_score_list = []
          for i in range(mul.size(0)):
              # (1, L) -> (1, 1, L) -> (1, L) -> (1, L, 1)
              mul_score_list.append(F.pad(smooth_tensor_1d(mul[i, :lengths[i], 0].unsqueeze(0), 2).unsqueeze(0),(0, mul.size(1)-lengths[i]),"constant").squeeze(0))
          
          mul = torch.cat(mul_score_list, dim=0).unsqueeze(-1)
          mul = mul.masked_fill(~np_mask.unsqueeze(-1), float("-inf"))
          
          attns = F.softmax(mul, dim=1) # (b, l, nh)
          x = (x * attns.unsqueeze(-1)).sum(1)
          x = x.view(x.size(0), -1)
          return x, attns.squeeze(2)

class ProtT5Frozen(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.initial_ln = nn.LayerNorm(1024)
        self.lin = nn.Linear(1024, 256)
        self.attn_head = AttentionHead(256, 1)
        self.clf_head = nn.Linear(256, 11)
        self.kld = nn.KLDivLoss(reduction="batchmean")
        self.lr = 1e-3

    def forward(self, embedding, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        #print(x_pred, x_attns)
        return x_pred, x_attns

    def predict(self, embedding, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        #print(x_pred, x_attns)
        return x_pred, x_pool, x_attns
    
    def attn_reg_loss(self, y_true, y_attn, y_tags, lengths, n):
        loss = 0
        count = 0
        reg_loss = 0
        for i in range(y_true.size(0)):
            reg_loss += fourier_att_prior_loss_dct(
                  F.pad(y_attn[i, :lengths[i]].unsqueeze(0).unsqueeze(0), (8,8),"replicate").squeeze(1),
                  lengths[i]//6,
                  0.2, 3)
        reg_loss = reg_loss / y_true.size(0)
        kld_loss = 0
        kld_count = 0
        for i in range(y_true.size(0)):
            if y_tags[i].sum() > 0:         
                for j in range(9):
                    if (j+1) in y_tags[i]:
                        pos_tar = (y_tags[i]==(j+1)).float()
                        kld_count += 1
                        kld_loss += pos_weights_annot[j] * self.kld(
                            torch.log(y_attn[i, :+lengths[i]].unsqueeze(0)), 
                            pos_tar[:lengths[i]].unsqueeze(0) / pos_tar[:lengths[i]].sum().unsqueeze(0))
        return reg_loss, kld_loss / torch.tensor(kld_count + 1e-5), kld_count

    def training_step(self, batch, batch_idx):
        #self.unfreeze()
        x, l, n, y, y_tags, _ = batch
        y_pred, y_attns =  self.forward(x, l, n)
        reg_loss, seq_loss, seq_count = self.attn_reg_loss(y, y_attns, y_tags, l, n)
        bce_loss = focal_loss(y_pred, y)
        loss = bce_loss + SUP_LOSS_MULT * seq_loss + REG_LOSS_MULT * reg_loss
        self.log('train_loss_batch', loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        #self.unfreeze()
        x, l, n, y, y_tags, _ = batch
        y_pred, y_attns =  self.forward(x, l, n)
        reg_loss, seq_loss, seq_count = self.attn_reg_loss(y, y_attns, y_tags, l, n)
        bce_loss = focal_loss(y_pred, y)
        loss = bce_loss + SUP_LOSS_MULT * seq_loss + REG_LOSS_MULT * reg_loss
        self.log('val_loss_batch', loss, on_epoch=True)
        self.log('bce_loss', bce_loss, on_epoch=True)
        return {'loss': loss, 
                'seq_loss': seq_loss,
                'reg_loss': reg_loss,
                'bce_loss': bce_loss,
                'seq_count': seq_count}
    
    

class ESM1bFrozen(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.initial_ln = nn.LayerNorm(1280)
        self.lin = nn.Linear(1280, 256)
        self.attn_head = AttentionHead(256, 1)
        self.clf_head = nn.Linear(256, 11)
        self.kld = nn.KLDivLoss(reduction="batchmean")
        self.lr = 1e-3

    def forward(self, embedding, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        #print(x_pred, x_attns)
        return x_pred, x_attns

    def predict(self, embedding, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        #print(x_pred, x_attns)
        return x_pred, x_pool, x_attns
    
    def attn_reg_loss(self, y_true, y_attn, y_tags, lengths, n):
        loss = 0
        count = 0
        reg_loss = 0
        for i in range(y_true.size(0)):
            reg_loss += fourier_att_prior_loss_dct(
                  F.pad(y_attn[i, :lengths[i]].unsqueeze(0).unsqueeze(0), (8,8),"replicate").squeeze(1),
                  lengths[i]//6,
                  0.2, 3)
        reg_loss = reg_loss / y_true.size(0)
        kld_loss = 0
        kld_count = 0
        for i in range(y_true.size(0)):
            if y_tags[i].sum() > 0:         
                for j in range(9):
                    if (j+1) in y_tags[i]:
                        pos_tar = (y_tags[i]==(j+1)).float()
                        kld_count += 1
                        kld_loss += pos_weights_annot[j] * self.kld(
                            torch.log(y_attn[i, :+lengths[i]].unsqueeze(0)), 
                            pos_tar[:lengths[i]].unsqueeze(0) / pos_tar[:lengths[i]].sum().unsqueeze(0))
        return reg_loss, kld_loss / torch.tensor(kld_count + 1e-5), kld_count

    def training_step(self, batch, batch_idx):
        #self.unfreeze()
        x, l, n, y, y_tags, _ = batch
        y_pred, y_attns =  self.forward(x, l, n)
        reg_loss, seq_loss, seq_count = self.attn_reg_loss(y, y_attns, y_tags, l, n)
        bce_loss = focal_loss(y_pred, y)
        loss = bce_loss + SUP_LOSS_MULT * seq_loss + REG_LOSS_MULT * reg_loss
        self.log('train_loss_batch', loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        #self.unfreeze()
        x, l, n, y, y_tags, _ = batch
        y_pred, y_attns =  self.forward(x, l, n)
        reg_loss, seq_loss, seq_count = self.attn_reg_loss(y, y_attns, y_tags, l, n)
        bce_loss = focal_loss(y_pred, y)
        loss = bce_loss + SUP_LOSS_MULT * seq_loss + REG_LOSS_MULT * reg_loss
        self.log('val_loss_batch', loss, on_epoch=True)
        self.log('bce_loss', bce_loss, on_epoch=True)
        return {'loss': loss, 
                'seq_loss': seq_loss,
                'reg_loss': reg_loss,
                'bce_loss': bce_loss,
                'seq_count': seq_count}



pos_weights_annot = torch.tensor([0.23, 0.92, 0.98, 2.63, 5.64, 1.60, 2.37, 1.87, 2.03])
class SignalTypeMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(267, 32)
        self.ln2 = nn.Linear(32, 9)
        self.lr = 1e-3

    def forward(self, x):
        x = nn.Tanh()(self.ln1(x))
        x = self.ln2(x)
        return x

