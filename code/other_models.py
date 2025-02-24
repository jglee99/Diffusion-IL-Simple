import torch
import torch.nn as nn
import numpy as np


# below stuff for outter MSE class
class Model_Cond_MSE(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim):
        super(Model_Cond_MSE, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.loss_mse = nn.MSELoss()

    def loss_on_batch(self, x_batch, y_batch):
        # to keep mse as close as poss to the diffusion pipeline
        # we manufacture timesteps and context masks and noisy y
        # as we did for diffusion models
        # but these are just sent into the archicture as zeros for mse model
        # this means we can use _exactly_ the same architecture as for the diff model
        # although, I don't know if there's any point as we'll have to develop a new model later
        # for discretised model
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        y_t = y_batch * 0.0
        y_pred = self.nn_model(y_t, x_batch, _ts, context_mask)
        return self.loss_mse(y_batch, y_pred)

    def sample(self, x_batch):
        n_sample = x_batch.shape[0]
        _ts = torch.zeros((n_sample, 1)).to(self.device)
        y_shape = (n_sample, self.y_dim)
        y_i = torch.zeros(y_shape).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        return self.nn_model(y_i, x_batch, _ts, context_mask)


def matrix_diag(diagonal):
    # batched diag operation
    # taken from here
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


class Model_Cond_MeanVariance(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim):
        super(Model_Cond_MeanVariance, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim

    def loss_on_batch(self, x_batch, y_batch):
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        y_t = y_batch * 0.0
        y_pred = self.nn_model(y_t, x_batch, _ts, context_mask)

        y_pred_mean = y_pred[:, :self.y_dim]
        y_pred_var = torch.log(1 + torch.exp(y_pred[:, self.y_dim:]))  # softplus ensure var>0

        covariance_matrix = matrix_diag(y_pred_var)
        # covariance_matrix is shape: batch_size, y_dim, y_dim, but off diagonal entries are zero
        dist = torch.distributions.multivariate_normal.MultivariateNormal(y_pred_mean, covariance_matrix)
        return -torch.mean(dist.log_prob(y_batch))  # return average negative log likelihood

    def sample(self, x_batch):
        n_sample = x_batch.shape[0]
        _ts = torch.zeros((n_sample, 1)).to(self.device)
        y_shape = (n_sample, self.y_dim)
        y_i = torch.zeros(y_shape).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        y_pred = self.nn_model(y_i, x_batch, _ts, context_mask)
        y_pred_mean = y_pred[:, :self.y_dim]
        y_pred_var = torch.log(1 + torch.exp(y_pred[:, self.y_dim:]))  # softplus ensure var>0

        covariance_matrix = matrix_diag(y_pred_var)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(y_pred_mean, covariance_matrix)
        return dist.sample()


class Model_Cond_Discrete(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim, n_bins):
        super(Model_Cond_Discrete, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.loss_crossent = nn.CrossEntropyLoss()
        self.n_bins = n_bins  # this is number of bins to discretise each action dimension into

    def loss_on_batch(self, x_batch, y_batch):
        # y_batch comes in continuous
        # we work out the bin it should be in and compute independent cross ent losses per action dim
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        y_t = y_batch * 0.0
        y_pred = self.nn_model(y_t, x_batch, _ts, context_mask)
        # y_pred is shape: batch_size, y_dim x n_bins
        loss = 0.0
        y_batch = torch.clip(y_batch, min=-0.99, max=0.99)
        for i in range(self.y_dim):
            idx_start = i * self.n_bins
            idx_end = (i + 1) * self.n_bins
            y_pred_dim = y_pred[:, idx_start:idx_end]
            y_true_dim_continuous = y_batch[:, i]
            # now find which bin y_true_dim_continuous is in
            # 1) convert from [-1,1] to [0, n_bins]
            y_true_dim_continuous += 1
            y_true_dim_continuous = y_true_dim_continuous / 2 * self.n_bins
            # 2) round _down_ to nearest integer
            y_true_dim_label = torch.floor(y_true_dim_continuous).long()

            # note that torch's crossent expects logits
            loss += self.loss_crossent(y_pred_dim, y_true_dim_label)
        return loss

    def sample(self, x_batch, sample_type="probabilistic"):
        # sample_type can be 'argmax' or 'probabilistic'
        # argmax selects most probable class,
        # probabilistic samples via softmax probs
        n_sample = x_batch.shape[0]
        _ts = torch.zeros((n_sample, 1)).to(self.device)
        y_shape = (n_sample, self.y_dim)
        y_i = torch.zeros(y_shape).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        y_output = torch.zeros((x_batch.shape[0], self.y_dim))  # set this up
        y_pred = self.nn_model(y_i, x_batch, _ts, context_mask)  # these are logits
        for i in range(self.y_dim):
            idx_start = i * self.n_bins
            idx_end = (i + 1) * self.n_bins
            y_pred_dim = y_pred[:, idx_start:idx_end]

            # 1) get class
            if sample_type == "argmax":
                class_idx = torch.argmax(y_pred_dim, dim=-1)
            elif sample_type == "probabilistic":
                # pass through softmax and sample
                y_pred_dim_probs = nn.functional.softmax(y_pred_dim, dim=-1)
                class_idx = torch.squeeze(torch.multinomial(y_pred_dim_probs, num_samples=1))
            # 2) do reverse of scaling, so now [0, n_bins] -> [-1, 1]
            y_output[:, i] = (((class_idx + 0.5) / self.n_bins) * 2) - 1

        return y_output


class Model_Cond_Kmeans(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim, kmeans_model):
        super(Model_Cond_Kmeans, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.loss_crossent = nn.CrossEntropyLoss()
        self.kmeans_model = kmeans_model

    def loss_on_batch(self, x_batch, y_batch):
        # y_batch comes in continuous
        # we work out the kmeans bin it should be in and compute cross ent losses
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        y_t = y_batch * 0.0
        y_pred = self.nn_model(y_t, x_batch, _ts, context_mask)
        # y_pred is shape: batch_size, n_clusters

        # figure out which kmeans bin it should be in
        y_true_label = torch.Tensor(self.kmeans_model.predict(y_batch.cpu())).to(self.device).long()
        return self.loss_crossent(y_pred, y_true_label)

    def sample(self, x_batch, sample_type="probabilistic"):
        # sample_type can be 'argmax' or 'probabilistic'
        # argmax selects most probable class,
        # probabilistic samples via softmax probs
        n_sample = x_batch.shape[0]
        _ts = torch.zeros((n_sample, 1)).to(self.device)
        y_shape = (n_sample, self.y_dim)
        y_i = torch.zeros(y_shape).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        y_pred = self.nn_model(y_i, x_batch, _ts, context_mask)  # these are logits
        # 1) get class
        if sample_type == "argmax":
            class_idx = torch.argmax(y_pred, dim=-1)
        elif sample_type == "probabilistic":
            # pass through softmax and sample
            y_pred_probs = nn.functional.softmax(y_pred, dim=-1)
            class_idx = torch.squeeze(torch.multinomial(y_pred_probs, num_samples=1))

        # 2) convert from class to kmeans centroid
        y_output = torch.index_select(
            torch.tensor(self.kmeans_model.cluster_centers_).to(self.device),
            dim=0,
            index=class_idx,
        )

        return y_output


class Model_Cond_BeT(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim, kmeans_model):
        super(Model_Cond_BeT, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.loss_crossent = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.kmeans_model = kmeans_model
        self.n_k = self.kmeans_model.n_clusters

    def loss_on_batch(self, x_batch, y_batch):
        # y_batch comes in continuous
        # we work out the kmeans bin it should be in and compute cross ent losses
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        y_t = y_batch * 0.0
        y_pred = self.nn_model(y_t, x_batch, _ts, context_mask)
        # y_pred is shape: batch_size,
        y_pred_label = y_pred[:, :self.n_k]

        # figure out which kmeans bin it should be in
        y_true_label = torch.Tensor(self.kmeans_model.predict(y_batch.cpu())).to(self.device).long()

        # now add in the residual
        # y_batch is shape [n_batch, y_dim]
        # y_pred is shape [n_batch, n_k + (y_dim*n_k)]
        # first find chunk of y_pred corresponding to y_true_label
        y_pred_residual_all = y_pred[:, self.n_k:].view(y_pred.shape[0], self.n_k, self.y_dim)  # batch_size, n_k, y_dim
        y_pred_residual = y_pred_residual_all.index_select(1, y_true_label)  # [batch_size,batch_size,y_dim]
        y_pred_residual = torch.diagonal(y_pred_residual, offset=0, dim1=0, dim2=1).T  # I think this is right, may need to check again

        # compute true residual
        K_centers = torch.Tensor(self.kmeans_model.cluster_centers_).to(self.device)  # n_k, y_dim
        y_true_label_center = K_centers.index_select(0, y_true_label)
        y_true_residual = y_batch - y_true_label_center
        return self.loss_crossent(y_pred_label, y_true_label) + 100 * self.loss_mse(y_true_residual, y_pred_residual)

    def sample(self, x_batch, sample_type="probabilistic"):
        # sample_type can be 'argmax' or 'probabilistic'
        # argmax selects most probable class,
        # probabilistic samples via softmax probs
        # "we first sample an action center according to the predicted bin centerprobabilities on thetthindex.
        # Once we have chosen an action centerAt,j, we add the correspondingresidual action〈ˆa(j)t〉to it
        # to recover a predicted continuous actionˆat=At,j+〈ˆa(j)t〉"
        n_sample = x_batch.shape[0]
        _ts = torch.zeros((n_sample, 1)).to(self.device)
        y_shape = (n_sample, self.y_dim)
        y_i = torch.zeros(y_shape).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        y_pred = self.nn_model(y_i, x_batch, _ts, context_mask)
        y_pred_label = y_pred[:, :self.n_k]  # these are logits
        # 1) get class
        if sample_type == "argmax":
            class_idx = torch.argmax(y_pred_label, dim=-1)
        elif sample_type == "probabilistic":
            # pass through softmax and sample
            y_pred_probs = nn.functional.softmax(y_pred_label, dim=-1)
            class_idx = torch.squeeze(torch.multinomial(y_pred_probs, num_samples=1))

        # 2) convert from class to kmeans centroid
        K_centers = torch.Tensor(self.kmeans_model.cluster_centers_).to(self.device)  # n_k, y_dim
        y_pred_label_center = K_centers.index_select(dim=0, index=class_idx)

        # 3) add on residual
        y_pred_residual_all = y_pred[:, self.n_k:].view(y_pred.shape[0], self.n_k, self.y_dim)  # batch_size, n_k, y_dim
        y_pred_residual = y_pred_residual_all.index_select(1, class_idx)  # [batch_size,batch_size,y_dim]
        y_pred_residual = torch.diagonal(y_pred_residual, offset=0, dim1=0, dim2=1).T  # I think this is right, may need to check again

        # 4) add center bin and residual
        y_output = y_pred_label_center + y_pred_residual
        # y_output = y_pred_residual
        return y_output


class Model_Cond_EBM(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim, n_counter_egs=256, ymin=-1, ymax=1, n_samples=4096, n_iters=3, stddev=0.33, K=0.5, sample_mode="derivative_free"):
        super(Model_Cond_EBM, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_counter_egs = n_counter_egs  # how many counter examples to draw
        self.ymin = ymin  # upper and lower bounds of y
        self.ymax = ymax

        # params used for sampling
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.stddev = stddev
        self.K = K
        self.sample_mode = sample_mode

        # langevin sampling
        self.l_noise_scale = 0.5
        self.l_coeff_start = 0.5
        self.l_coeff_end = 0.005
        self.l_delta_clip = 0.5  # not used currently
        self.l_M = 1
        self.n_mcmc_iters = 40
        self.l_n_sampleloops = 2  # 1 or 2 (only inference time)
        self.l_overwrite = False

    def loss_on_batch(self, x_batch, y_batch, extract_embedding=True):
        # following appendix B, Algorithm 1 of Implicit Behavioral Cloning
        batch_size = x_batch.shape[0]
        loss = 0

        # first need to make batchsize much larger
        y_batch = y_batch.repeat(self.n_counter_egs + 1, 1)

        if extract_embedding:
            x_embed = self.nn_model.embed_context(x_batch)
            x_embed = x_embed.repeat(self.n_counter_egs + 1, 1)
        else:
            if len(x_batch.shape) == 2:
                x_batch = x_batch.repeat(self.n_counter_egs + 1, 1)
            else:
                x_batch = x_batch.repeat(self.n_counter_egs + 1, 1, 1, 1)

        # unused inputs
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(y_batch.shape[0]).to(self.device)

        if self.sample_mode == "langevin":
            l_noise_scale = self.l_noise_scale
            l_coeff_start = self.l_coeff_start
            n_mcmc_iters = self.n_mcmc_iters

            y_samples = y_batch[self.n_counter_egs:, :]

            # random init
            y_samples = torch.rand(y_samples.size()).to(self.device) * (self.ymax - self.ymin) + self.ymin

            # run mcmc chain
            for i in range(n_mcmc_iters):

                l_coeff = self.l_coeff_end + l_coeff_start * (1 - (i / n_mcmc_iters)) ** 2
                y_samples.requires_grad = True

                if extract_embedding:
                    y_pred = self.nn_model(
                        y_samples, x_batch[self.n_counter_egs:], _ts[self.n_counter_egs:], context_mask[self.n_counter_egs:], x_embed[self.n_counter_egs:]
                    )  # forward pass
                else:
                    y_pred = self.nn_model(y_samples, x_batch[self.n_counter_egs:], _ts[self.n_counter_egs:], context_mask[self.n_counter_egs:])  # forward pass

                y_pred_grad = torch.autograd.grad(y_pred, y_samples, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]  # compute gradients
                delta_action = 0.5 * y_pred_grad + torch.randn(size=y_samples.size()).to(self.device) * l_noise_scale
                y_samples = y_samples - l_coeff * (delta_action)
                y_samples = torch.clip(y_samples, min=self.ymin, max=self.ymax)
                y_samples = y_samples.detach()

            # these form negative samples
            y_batch[self.n_counter_egs:, :] = y_samples

            # B.3.1 gradient penalty
            loss_grad = torch.maximum(torch.zeros_like(y_pred_grad[:, 0]).to(self.device), (torch.linalg.norm(y_pred_grad, dim=1, ord=np.inf) - self.l_M)) ** 2
            loss += torch.mean(loss_grad)
        else:
            # draw counter-examples from U(y_min, y_max)
            y_batch[self.n_counter_egs:, :] = torch.rand(y_batch[self.n_counter_egs:, :].size()) * (self.ymax - self.ymin) + self.ymin
            # (note we now use the y input from the diffusion model again)

        # forward pass
        if extract_embedding:
            y_pred = self.nn_model(y_batch, x_batch, _ts, context_mask, x_embed)
        else:
            y_pred = self.nn_model(y_batch, x_batch, _ts, context_mask)  # (n_batch x (n_counter_egs+1), 1)

        # y_pred comes out in a vector of size (n_batch x n_counter_egs, 1), we need to reshape this
        y_pred_reshape = y_pred.view(self.n_counter_egs + 1, batch_size).T

        if self.l_overwrite:
            # could overwrite half to be drawn uniformly
            y_pred_reshape[:, 1 + int(self.n_counter_egs / 2)] = (
                torch.rand(y_pred_reshape[:, 1 + int(self.n_counter_egs / 2)].size()) * (self.ymax - self.ymin) + self.ymin
            )

        loss_NCE = -(-y_pred_reshape[:, 0] - torch.logsumexp(-y_pred_reshape, dim=1))
        loss += torch.mean(loss_NCE)

        return loss

    def sample(self, x_batch, extract_embedding=True):
        batch_size = x_batch.shape[0]
        # note that n_samples is only used derivative_free, it refers to the population pool
        # NOT the number of samples to return
        n_samples = self.n_samples
        n_iters = self.n_iters
        stddev = self.stddev
        K = self.K

        if self.sample_mode == "derivative_free":
            # this follows algorithm 1 in appendix B1 of Implicit BC
            y_samples = torch.rand((n_samples * batch_size, self.y_dim)).to(self.device) * (self.ymax - self.ymin) + self.ymin
            if extract_embedding:
                x_embed = self.nn_model.embed_context(x_batch)
                x_embed = x_embed.repeat(n_samples, 1)
                x_size_0 = x_embed.shape[0]
            else:
                if len(x_batch.shape) == 2:
                    x_batch = x_batch.repeat(n_samples, 1)
                else:
                    x_batch = x_batch.repeat(n_samples, 1, 1, 1)
                x_size_0 = x_batch.shape[0]
            for i in range(n_iters):
                # compute energies
                _ts = torch.zeros((x_size_0, 1)).to(self.device)
                context_mask = torch.zeros(x_size_0).to(self.device)
                if extract_embedding:
                    y_pred = self.nn_model(y_samples, x_batch, _ts, context_mask, x_embed)
                else:
                    y_pred = self.nn_model(y_samples, x_batch, _ts, context_mask)
                y_pred_reshape = y_pred.view(n_samples, batch_size).T

                # softmax
                y_probs_reshape = torch.nn.functional.softmax(-y_pred_reshape, dim=1)

                y_samples_reshape = torch.permute(y_samples.view(n_samples, batch_size, self.y_dim), (1, 0, 2))
                if i < n_iters - 1:  # don't want to do this for last iteration
                    # loop over individual samples here
                    for j in range(batch_size):
                        idx_sample = torch.multinomial(y_probs_reshape[j, :], n_samples, replacement=True)
                        y_samples[j * n_samples:(j + 1) * n_samples] = y_samples_reshape[j, idx_sample]
                    y_samples = y_samples + torch.randn(y_samples.size()).to(self.device) * stddev
                    y_samples = torch.clip(y_samples, min=self.ymin, max=self.ymax)
                    stddev = stddev * K  # shrink sampling scale

            y_idx = torch.argmin(y_pred_reshape, dim=-1)  # same as doing argmax over probs
            y_output = torch.diagonal(torch.index_select(y_samples_reshape, dim=1, index=y_idx), dim1=0, dim2=1).T
        elif self.sample_mode == "langevin":
            l_noise_scale = self.l_noise_scale
            l_coeff_start = self.l_coeff_start
            l_coeff_end = self.l_coeff_end
            n_mcmc_iters = self.n_mcmc_iters

            y_samples = torch.rand((batch_size, self.y_dim)).to(self.device) * (self.ymax - self.ymin) + self.ymin

            _ts = torch.zeros((x_batch.shape[0], 1)).to(self.device)
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

            if extract_embedding:
                x_embed = self.nn_model.embed_context(x_batch)

            for j in range(self.l_n_sampleloops):
                # run mcmc chain
                for i in range(n_mcmc_iters):
                    if j == 0:
                        l_coeff = self.l_coeff_end + l_coeff_start * (1 - (i / n_mcmc_iters)) ** 2
                    else:
                        l_coeff = l_coeff_end
                    y_samples.requires_grad = True

                    if extract_embedding:
                        y_pred = self.nn_model(y_samples, x_batch, _ts, context_mask, x_embed)
                    else:
                        y_pred = self.nn_model(y_samples, x_batch, _ts, context_mask)  # forward pass
                    y_pred_grad = torch.autograd.grad(y_pred, y_samples, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]  # compute gradients
                    delta_action = 0.5 * y_pred_grad + torch.randn(size=y_samples.size()).to(self.device) * l_noise_scale
                    y_samples = y_samples - l_coeff * (delta_action)
                    y_samples = torch.clip(y_samples, min=self.ymin, max=self.ymax)
                    y_samples = y_samples.detach()
                y_output = y_samples

        return y_output
