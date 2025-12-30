import torch
import torch.nn as nn
import torch.nn.functional
from layer.transformer import Encoder, EncoderLayer, FullAttention, iTransformer_Embedder, AttentionLayer
from layer.wtconv2d import WTConv2d


class FrequencyFusion(nn.Module):
    def __init__(self, C, out_feature, window, n_fft=16, hop_length=None, win_length=None):
        super().__init__()
        self.C = C
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else max(1, n_fft // 4)
        self.win_length = win_length if win_length is not None else n_fft
        self.out_feature = out_feature
        self.window = window
        self.freq_conv = nn.Conv1d(C * (n_fft // 2 + 1), out_feature, kernel_size=3)

    def forward(self, x):
        B, L, C = x.shape
        x_cpu = x.cpu()
        window_tensor = torch.hann_window(self.win_length)
        mags_all = []

        for b in range(B):
            mags_per_channel = []
            for c in range(C):
                x_bc = x_cpu[b, :, c]
                Xc = torch.stft(
                    x_bc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=window_tensor,
                    return_complex=True
                )
                mag = torch.abs(Xc)
                mags_per_channel.append(mag)
            mags_b = torch.stack(mags_per_channel, dim=0)
            mags_all.append(mags_b)

        mags = torch.stack(mags_all, dim=0)
        B, C, F, T_spec = mags.shape
        mags_reshaped = mags.view(B, C * F, T_spec)
        mags_reshaped = mags_reshaped.to(x.device)
        x_fre = self.freq_conv(mags_reshaped)
        x_fre = torch.nn.functional.adaptive_avg_pool1d(x_fre, self.window)
        x_fre = x_fre.permute(0, 2, 1).contiguous()
        return x_fre


class CausalResize(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.padding_size = padding_size

    def forward(self, x):
        return x[..., : - self.padding_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, n_features, n_filters, filter_size, dilation='dilation factor', dropout_rate='dropout_rate'):
        super().__init__()
        self.padding_size = filter_size - 1
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_filters, kernel_size=filter_size,
                              stride=1, padding=self.padding_size, dilation=dilation)
        self.resize = CausalResize(padding_size=self.padding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv_ = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=filter_size,
                               stride=1, padding=self.padding_size, dilation=dilation)
        self.resize_ = CausalResize(padding_size=self.padding_size)
        self.relu_ = nn.ReLU()
        self.dropout_ = nn.Dropout(p=dropout_rate)
        self.net = nn.Sequential(self.conv, self.resize, self.relu, self.dropout)
        self.conv_residual = nn.Conv1d(in_channels=n_features, out_channels=n_filters, kernel_size=1) \
            if n_features != n_filters else None
        self.relu__ = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x_ = self.net(x)
        residual = x if self.conv_residual is None else self.conv_residual(x)
        return self.relu__(x_ + residual).permute(0, 2, 1)


class fretime_iTransformer(nn.Module):
    def __init__(self, configs):
        super(fretime_iTransformer, self).__init__()

        self.seq_len = configs.hist_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_WTConv = configs.UseWTConv
        self.use_TCN = configs.UseTCN
        self.use_Fre = configs.UseFre
        self.enc_embedding = iTransformer_Embedder(self.seq_len, configs.d_model, configs.dropout)
        self.conv = WTConv2d(configs.hist_len, configs.hist_len,
                             kernel_size=configs.kernel_size,
                             stride=1, wt_levels=configs.wt_levels)
        self.tcn1 = nn.Sequential(TCNBlock(configs.enc_in, configs.channel, 3),
                                  TCNBlock(configs.channel, configs.enc_in, 3))
        self.tcn2 = nn.Sequential(TCNBlock(configs.enc_in, configs.channel, 5),
                                  TCNBlock(configs.channel, configs.enc_in, 5))
        self.tcn3 = nn.Sequential(TCNBlock(configs.enc_in, configs.channel, 7),
                                  TCNBlock(configs.channel, configs.enc_in, 7))
        self.fre = FrequencyFusion(configs.enc_in, configs.enc_in, configs.hist_len)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projection = nn.Linear(configs.d_model, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape
        if self.use_WTConv:
            x_enc = self.conv(x_enc.unsqueeze(-1)).squeeze(-1)
            x_in = x_enc
        if self.use_TCN:
            x1 = self.tcn1(x_enc)
            x2 = self.tcn2(x_enc)
            x3 = self.tcn3(x_enc)
            x_in = x1 + x2 + x3
        if self.use_Fre:
            x_fre = self.fre(x_enc)
            x_in = torch.cat([x_in, x_fre], dim=-1)
        enc_out = self.enc_embedding(x_in, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]
