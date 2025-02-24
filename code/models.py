import torch
import torch.nn as nn


class Model_mlp_mse(nn.Module):
    # NN with three relu hidden layers
    # quantile outputs are independent of eachother
    def __init__(self, n_input, n_hidden, n_output):
        super(Model_mlp_mse, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output, bias=True),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def loss_on_batch(self, x_batch, y_batch):
        # add this here so can sync w diffusion model
        y_pred_batch = self(x_batch)
        loss = self.loss_fn(y_pred_batch, y_batch)
        return loss


class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__()
        # just a fully connected NN with sin activations
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class FCBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        # one layer of non-linearities (just a useful building block to use below)
        self.model = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.BatchNorm1d(num_features=out_feats),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, trans_emb_dim, transformer_dim, nheads):
        super(TransformerEncoderBlock, self).__init__()
        # mainly going off of https://jalammar.github.io/illustrated-transformer/

        self.trans_emb_dim = trans_emb_dim
        self.transformer_dim = transformer_dim
        self.nheads = nheads

        self.input_to_qkv1 = nn.Linear(self.trans_emb_dim, self.transformer_dim * 3)
        self.multihead_attn1 = nn.MultiheadAttention(self.transformer_dim, num_heads=self.nheads)
        self.attn1_to_fcn = nn.Linear(self.transformer_dim, self.trans_emb_dim)
        self.attn1_fcn = nn.Sequential(
            nn.Linear(self.trans_emb_dim, self.trans_emb_dim * 4),
            nn.GELU(),
            nn.Linear(self.trans_emb_dim * 4, self.trans_emb_dim),
        )
        self.norm1a = nn.BatchNorm1d(self.trans_emb_dim)
        self.norm1b = nn.BatchNorm1d(self.trans_emb_dim)

    def split_qkv(self, qkv):
        assert qkv.shape[-1] == self.transformer_dim * 3
        q = qkv[:, :, :self.transformer_dim]
        k = qkv[:, :, self.transformer_dim: 2 * self.transformer_dim]
        v = qkv[:, :, 2 * self.transformer_dim:]
        return (q, k, v)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        qkvs1 = self.input_to_qkv1(inputs)
        # shape out = [3, batchsize, transformer_dim*3]

        qs1, ks1, vs1 = self.split_qkv(qkvs1)
        # shape out = [3, batchsize, transformer_dim]

        attn1_a = self.multihead_attn1(qs1, ks1, vs1, need_weights=False)
        attn1_a = attn1_a[0]
        # shape out = [3, batchsize, transformer_dim = trans_emb_dim x nheads]

        attn1_b = self.attn1_to_fcn(attn1_a)
        attn1_b = attn1_b / 1.414 + inputs / 1.414  # add residual
        # shape out = [3, batchsize, trans_emb_dim]

        # normalise
        attn1_b = self.norm1a(attn1_b.transpose(0, 2).transpose(0, 1))
        attn1_b = attn1_b.transpose(0, 1).transpose(0, 2)
        # batchnorm likes shape = [batchsize, trans_emb_dim, 3]
        # so have to shape like this, then return

        # fully connected layer
        attn1_c = self.attn1_fcn(attn1_b) / 1.414 + attn1_b / 1.414
        # shape out = [3, batchsize, trans_emb_dim]

        # normalise
        # attn1_c = self.norm1b(attn1_c)
        attn1_c = self.norm1b(attn1_c.transpose(0, 2).transpose(0, 1))
        attn1_c = attn1_c.transpose(0, 1).transpose(0, 2)
        return attn1_c


class Model_mlp_diff_embed(nn.Module):
    # this model embeds x, y, t, before input into a fc NN (w residuals)
    def __init__(
        self,
        x_dim,
        n_hidden,
        y_dim,
        embed_dim,
        output_dim=None,
        net_type="fc",
    ):
        super(Model_mlp_diff_embed, self).__init__()
        self.embed_dim = embed_dim  # input embedding dimension
        self.n_hidden = n_hidden
        self.net_type = net_type
        self.x_dim = x_dim
        self.y_dim = y_dim
        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models

        # embedding NNs
        self.x_embed_nn = nn.Sequential(
            nn.Linear(x_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )  # no prev hist
        self.y_embed_nn = nn.Sequential(
            nn.Linear(y_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.t_embed_nn = TimeSiren(1, self.embed_dim)

        # fc nn layers
        if self.net_type == "fc":
            self.fc1 = nn.Sequential(FCBlock(self.embed_dim * 3, n_hidden))  # no prev hist
            self.fc2 = nn.Sequential(FCBlock(n_hidden + y_dim + 1, n_hidden))  # will concat y and t at each layer
            self.fc3 = nn.Sequential(FCBlock(n_hidden + y_dim + 1, n_hidden))
            self.fc4 = nn.Sequential(nn.Linear(n_hidden + y_dim + 1, self.output_dim))

        # transformer layers
        elif self.net_type == "transformer":
            self.nheads = 16  # 16
            self.trans_emb_dim = 64
            self.transformer_dim = self.trans_emb_dim * self.nheads  # embedding dim for each of q,k and v (though only k and v have to be same I think)

            self.t_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            self.y_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            self.x_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)

            self.pos_embed = TimeSiren(1, self.trans_emb_dim)

            self.transformer_block1 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block2 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block3 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block4 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)

            self.final = nn.Linear(self.trans_emb_dim * 3, self.output_dim)
        else:
            raise NotImplementedError

    def forward(self, y, x, t):
        # embed y, x, t
        x_e = self.x_embed_nn(x)  # no prev hist
        y_e = self.y_embed_nn(y)
        t_e = self.t_embed_nn(t)

        x_e = -x_e

        # pass through fc nn
        if self.net_type == "fc":
            net_output = self.forward_fcnn(x_e, y_e, t_e, x, y, t)

        # or pass through transformer encoder
        elif self.net_type == "transformer":
            net_output = self.forward_transformer(x_e, y_e, t_e, x, y, t)

        return net_output

    def forward_fcnn(self, x_e, y_e, t_e, x, y, t):
        net_input = torch.cat((x_e, y_e, t_e), 1)
        nn1 = self.fc1(net_input)
        nn2 = self.fc2(torch.cat((nn1 / 1.414, y, t), 1)) + nn1 / 1.414  # residual and concat inputs again
        nn3 = self.fc3(torch.cat((nn2 / 1.414, y, t), 1)) + nn2 / 1.414
        net_output = self.fc4(torch.cat((nn3, y, t), 1))
        return net_output

    def forward_transformer(self, x_e, y_e, t_e, x, y, t):
        # roughly following this: https://jalammar.github.io/illustrated-transformer/

        t_input = self.t_to_input(t_e)
        y_input = self.y_to_input(y_e)
        x_input = self.x_to_input(x_e)
        # shape out = [batchsize, trans_emb_dim]

        # add 'positional' encoding
        # note, here position refers to order tokens are fed into transformer
        t_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 1.0)
        y_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 2.0)
        x_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 3.0)

        inputs1 = torch.cat((t_input[None, :, :], y_input[None, :, :], x_input[None, :, :]), 0)
        # shape out = [3, batchsize, trans_emb_dim]

        block1 = self.transformer_block1(inputs1)
        block2 = self.transformer_block2(block1)
        block3 = self.transformer_block3(block2)
        block4 = self.transformer_block4(block3)

        # flatten and add final linear layer
        # transformer_out = block2
        transformer_out = block4
        transformer_out = transformer_out.transpose(0, 1)  # roll batch to first dim
        # shape out = [batchsize, 3, trans_emb_dim]

        flat = torch.flatten(transformer_out, start_dim=1, end_dim=2)
        # shape out = [batchsize, 3 x trans_emb_dim]

        out = self.final(flat)
        # shape out = [batchsize, n_dim]
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    beta_t = (beta2 - beta1) * torch.arange(-1, T, dtype=torch.float32) / (T - 1) + beta1
    beta_t[0] = beta1  # modifying this so that beta_t[1] = beta1, and beta_t[n_T]=beta2, while beta[0] is never used
    # this is as described in Denoising Diffusion Probabilistic Models paper, section 4
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class Model_Cond_Diffusion(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, x_dim, y_dim):
        super().__init__()

        # define alpha_t, beta_t, etc
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.nn_model = nn_model
        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()
        self.x_dim = x_dim
        self.y_dim = y_dim

    def loss_on_batch(self, x_batch, y_batch):
        # choose random times
        _ts = torch.randint(1, self.n_T + 1, (y_batch.shape[0], 1)).to(self.device)

        # randomly sample some noise, noise ~ N(0, 1)
        noise = torch.randn_like(y_batch).to(self.device)

        # add noise to clean target actions
        y_t = self.sqrtab[_ts] * y_batch + self.sqrtmab[_ts] * noise

        # use nn model to predict noise
        noise_pred_batch = self.nn_model(y_t, x_batch, _ts / self.n_T)

        # return mse between predicted and true noise
        return self.loss_mse(noise, noise_pred_batch)


    def sample_extra(self, x_batch, extra_steps=4, return_y_trace=False):

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        # for i_dummy in range(self.n_T, 0, -1):
        for i_dummy in range(self.n_T, -extra_steps, -1):
            i = max(i_dummy, 1)
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is)

            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.same_channels:
            # if num channels has not changed, resuidual is original x
            x = x + x2
        else:
            # if num channels has changed, residual is first layer with same num channels
            x = x1 + x2
        return x / 1.414


class Model_cnn_mlp(nn.Module):
    """CNN followed by either transformer or MLP."""

    def __init__(self, x_shape, n_hidden, y_dim, embed_dim, net_type, output_dim=None):
        super().__init__()

        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.n_feat = 64
        self.net_type = net_type

        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models

        # set up CNN for image
        self.conv_down1 = nn.Sequential(
            ResidualConvBlock(self.x_shape[-1], self.n_feat),
            nn.MaxPool2d(2),
        )
        self.conv_down3 = nn.Sequential(
            ResidualConvBlock(self.n_feat, self.n_feat * 2),
            nn.MaxPool2d(2),
        )
        self.imageembed = nn.Sequential(nn.AvgPool2d(8))

        cnn_out_dim = self.n_feat * 2  # how many features after flattening -- WARNING, will have to adjust this for diff size input resolution
        # it is the flattened size after CNN layers, and average pooling

        # then once have flattened vector out of CNN, just feed into previous Model_mlp_diff_embed
        self.nn_downstream = Model_mlp_diff_embed(
            cnn_out_dim,
            self.n_hidden,
            self.y_dim,
            self.embed_dim,
            self.output_dim,
            net_type=self.net_type,
        )

    def forward(self, y, x, t):
        # torch expects batch_size, channels, height, width
        # but we feed in batch_size, height, width, channels
        x_embed = self.embed_context(x)
        return self.nn_downstream(y, x_embed, t)

    def embed_context(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.conv_down1(x)
        x3 = self.conv_down3(x1)  # [batch_size, 128, 35, 18]
        # c3 is [batch size, 128, 4, 4]
        x_embed = self.imageembed(x3)
        # c_embed is [batch size, 128, 1, 1]
        x_embed = x_embed.view(x.shape[0], -1)
        # c_embed is now [batch size, 128]
        return x_embed
