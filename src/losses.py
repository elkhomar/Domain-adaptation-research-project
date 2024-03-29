import torch
import torch.functional as F
import torch.nn as nn
import ot
import skada

class RBF(nn.Module):

    def __init__(self, n_kernels=1, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth = bandwidth
        self.n_kernels = n_kernels

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (
                self.get_bandwidth(L2_distances)
                * self.bandwidth_multipliers.to(X.device)
            )[:, None, None]
        ).sum(dim=0)


class RQ(nn.Module):
    def __init__(self, n_kernels=6, mul_factor=2.0, alpha=1.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth = bandwidth
        self.n_kernels = n_kernels
        self.alpha = alpha

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        bandwidth = self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(
            X.device
        )
        K = (
            1 + L2_distances[None, ...] / (2 * self.alpha * bandwidth[:, None, None])
        ) ** (-self.alpha)
        return K.sum(dim=0)


class MMDLossBandwidth(nn.Module):

    def __init__(self, kernel=RQ()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y, **kwargs):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class WassersteinLoss(nn.Module):
    def __init__(self, reg=0, unbiased=False):
        super(WassersteinLoss, self).__init__()
        self.reg = reg
        self.unbiased = unbiased

    def forward(self, source, target, **kwargs):
        d = source.shape[1]
        if self.unbiased:
            s1 = source[: len(source) // 2]
            s2 = source[len(source) // 2 :]
            t1 = target[: len(target) // 2]
            t2 = target[len(target) // 2 :]

            #
            ls1s2 = (ot.solve_sample(s1, s2, reg=self.reg).value) / d
            lt1t2 = (ot.solve_sample(t1, t2, reg=self.reg).value) / d

            ls1t1 = (ot.solve_sample(s1, t1, reg=self.reg).value) / d
            # ls1t2 = (ot.solve_sample(s1, t2, reg=self.reg).value) / d
            # ls2t1 = (ot.solve_sample(s2, t1, reg=self.reg).value) / d
            # ls2t2 = (ot.solve_sample(s2, t2, reg=self.reg).value) / d
            loss = ls1t1 - 0.5 * (ls1s2 + lt1t2)
        else:
            loss = (ot.solve_sample(source, target, reg=self.reg).value) / d
        return loss


class SlicedWassersteinLoss(nn.Module):
    def __init__(self, unbiased=False, n_proj=1000):
        super(SlicedWassersteinLoss, self).__init__()
        self.unbiased = unbiased
        self.n_proj = n_proj
        self.seed = torch.initial_seed()

    def forward(self, source, target, **kwargs):
        torch.use_deterministic_algorithms(True, warn_only=True)
        if self.unbiased:
            s1 = source[: len(source) // 2]
            s2 = source[len(source) // 2 :]
            t1 = target[: len(target) // 2]
            t2 = target[len(target) // 2 :]

            #
            ls1s2 = ot.sliced_wasserstein_distance(s1, s2, n_projections=self.n_proj)
            lt1t2 = ot.sliced_wasserstein_distance(t1, t2, n_projections=self.n_proj)

            ls1t1 = ot.sliced_wasserstein_distance(s1, t1, n_projections=self.n_proj)
            # ls1t2 = ot.sliced_wasserstein_distance(s1, t2, n_projections=self.n_proj)
            # ls2t1 = ot.sliced_wasserstein_distance(s2, t1, n_projections=self.n_proj)
            # ls2t2 = ot.sliced_wasserstein_distance(s2, t2, n_projections=self.n_proj)
            loss = ls1t1 - 0.5 * (ls1s2 + lt1t2)
        else:
            loss = ot.sliced_wasserstein_distance(
                source, target, n_projections=self.n_proj
            )
        return loss


class CoralLoss(nn.Module):
    def __init__(self):
        super(CoralLoss, self).__init__()

    def forward(self, source, target, **kwargs):
        d = source.size(1)  # dim vector

        source_c = self.compute_covariance(source)
        target_c = self.compute_covariance(target)

        loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

        loss = loss / (4 * d * d)
        return loss

    def compute_covariance(self, input_data):
        """
        Compute Covariance matrix of the input data
        """
        n = input_data.size(0)  # batch_size

        id_row = torch.ones(n).resize(1, n).to(device=input_data.device)
        sum_column = torch.mm(id_row, input_data)
        mean_column = torch.div(sum_column, n)
        term_mul_2 = torch.mm(mean_column.t(), mean_column)
        d_t_d = torch.mm(input_data.t(), input_data)
        c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

        return c


class DeepJDOT_Loss(nn.Module):
    """
    reg_d : float, default=1
    Distance term regularization parameter.
    reg_cl : float, default=1
    Class distance term regularization parameter.

    """

    def __init__(self, reg_d=1, reg_cl=1):
        super(DeepJDOT_Loss, self).__init__()
        self.reg_d = reg_d
        self.reg_cl = reg_cl

    def forward(self, source, target, **kwargs):
        """
        We pass the labels through the kwargs argument so the other losses don't have to explicitly use y and y_target
        """
        y_source = kwargs["y_source"] if kwargs["y_source"] else torch.zeros(0)
        y_target = kwargs["preds_target"] if kwargs["preds_target"] else torch.zeros(0)
        return self.deepjdot_loss(source, target, y_source, y_target)

    def deepjdot_loss(
        self,
        embedd,
        embedd_target,
        logits_source,
        logits_target,
        sample_weights=None,
        target_sample_weights=None,
        criterion=None,
    ):
        """Compute the OT loss for DeepJDOT method [1]_.

        Parameters
        ----------
        embedd : tensor
            embeddings of the source data used to perform the distance matrix.
        embedd_target : tensor
            embeddings of the target data used to perform the distance matrix.
        y : tensor
            labels of the source data used to perform the distance matrix.
        y_target : tensor
            labels of the target data used to perform the distance matrix.
        sample_weights : tensor
            Weights of the source samples.
            If None, create uniform weights.
        target_sample_weights : tensor
            Weights of the source samples.
            If None, create uniform weights.
        criterion : torch criterion (class)
            The criterion (loss) used to compute the
            DeepJDOT loss. If None, use the CrossEntropyLoss.

        Returns
        -------
        loss : ndarray
            The loss of the method.

        References
        ----------
        .. [1]  Bharath Bhushan Damodaran, Benjamin Kellenberger,
                Remi Flamary, Devis Tuia, and Nicolas Courty.
                DeepJDOT: Deep Joint Distribution Optimal Transport
                for Unsupervised Domain Adaptation. In ECCV 2018
                15th European Conference on Computer Vision,
                September 2018. Springer.
        """

        y = logits_source
        y_target = logits_target

        dist = torch.cdist(embedd, embedd_target, p=2) ** 2
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # Simple MSE loss (for symmetry, 0/1 loss also works)
        dist_y = torch.cdist(y, y_target, p=2) ** 2

        # y = torch.argmax(y, dim=1)
        # y_target_matrix = y_target.repeat(len(y_target), 1, 1).permute(1, 2, 0)
        # loss_target_2 = criterion(y_target_matrix, y.repeat(len(y), 1)).T

        M = self.reg_d * dist + self.reg_cl * dist_y

        # Compute the loss
        if sample_weights is None:
            sample_weights = torch.full(
                (len(embedd),), 1.0 / len(embedd), device=embedd.device
            )
        if target_sample_weights is None:
            target_sample_weights = torch.full(
                (len(embedd_target),),
                1.0 / len(embedd_target),
                device=embedd_target.device,
            )
        loss = ot.emd2(sample_weights, target_sample_weights, M)

        return loss


class UnbiasedDeepJDOT_Loss(nn.Module):
    """
    reg_d : float, default=1
    Distance term regularization parameter.
    reg_cl : float, default=1
    Class distance term regularization parameter.

    """

    def __init__(self, reg_d=1, reg_cl=1):
        super(UnbiasedDeepJDOT_Loss, self).__init__()
        self.reg_d = reg_d
        self.reg_cl = reg_cl

    def forward(self, source, target, **kwargs):
        """
        We pass the labels through the kwargs argument so the other losses don't have to explicitly use y and y_target
        """
        y_source = kwargs["y_source"] if kwargs["y_source"] else torch.zeros(0)
        y_target = kwargs["preds_target"] if kwargs["preds_target"] else torch.zeros(0)
        return self.unbiased_deepjdot_loss(source, target, y_source, y_target)

    def unbiased_deepjdot_loss(
        self,
        embedd,
        embedd_target,
        logits_source,
        logits_target,
        sample_weights=None,
        target_sample_weights=None,
        criterion=None,
    ):
        """Compute the OT loss for DeepJDOT method [1]_.

        Parameters
        ----------
        embedd : tensor
            embeddings of the source data used to perform the distance matrix.
        embedd_target : tensor
            embeddings of the target data used to perform the distance matrix.
        y : tensor
            labels of the source data used to perform the distance matrix.
        y_target : tensor
            labels of the target data used to perform the distance matrix.
        sample_weights : tensor
            Weights of the source samples.
            If None, create uniform weights.
        target_sample_weights : tensor
            Weights of the source samples.
            If None, create uniform weights.
        criterion : t²orch criterion (class)
            The criterion (loss) used to compute the
            DeepJDOT loss. If None, use the CrossEntropyLoss.

        Returns
        -------
        loss : ndarray
            The loss of the method.

        References
        ----------
        .. [1]  Bharath Bhushan Damodaran, Benjamin Kellenberger,
                Remi Flamary, Devis Tuia, and Nicolas Courty.
                DeepJDOT: Deep Joint Distribution Optimal Transport
                for Unsupervised Domain Adaptation. In ECCV 2018
                15th European Conference on Computer Vision,
                September 2018. Springer.
        """

        y_s1 = logits_source[: len(logits_source) // 2]
        y_s2 = logits_source[len(logits_source) // 2 :]
        y_t1 = logits_target[: len(logits_source) // 2]
        y_t2 = logits_target[len(logits_source) // 2 :]

        x_s1 = embedd[: len(embedd) // 2]
        x_s2 = embedd[len(embedd) // 2 :]
        x_t1 = embedd_target[: len(embedd) // 2]
        x_t2 = embedd_target[len(embedd) // 2 :]

        dist_xs1t1 = torch.cdist(x_s1, x_t1, p=2) ** 2
        dist_xs1s2 = torch.cdist(x_s1, x_s2, p=2) ** 2
        dist_xt1t2 = torch.cdist(x_t1, x_t2, p=2) ** 2

        # Simple MSE loss (for symmetry, 0/1 loss also works)
        dist_ys1t1 = torch.cdist(y_s1, y_t1, p=2) ** 2
        dist_ys1s2 = torch.cdist(y_s1, y_s2, p=2) ** 2
        dist_yt1t2 = torch.cdist(y_t1, y_t2, p=2) ** 2

        # y = torch.argmax(y, dim=1)
        # y_target_matrix = y_target.repeat(len(y_target), 1, 1).permute(1, 2, 0)
        # loss_target_2 = criterion(y_target_matrix, y.repeat(len(y), 1)).T

        M_s1t1 = self.reg_d * dist_xs1t1 + self.reg_cl * dist_ys1t1
        M_s1s2 = self.reg_d * dist_xs1s2 + self.reg_cl * dist_ys1s2
        M_t1t2 = self.reg_d * dist_xt1t2 + self.reg_cl * dist_yt1t2

        # Compute the loss
        if sample_weights is None:
            sample_weights = torch.full(
                (len(embedd) // 2,), 1.0 / (len(embedd) // 2), device=embedd.device
            )
        if target_sample_weights is None:
            target_sample_weights = torch.full(
                (len(embedd_target) // 2,),
                1.0 / (len(embedd_target) // 2),
                device=embedd_target.device,
            )
        loss_s1t1 = ot.emd2(sample_weights, target_sample_weights, M_s1t1)
        loss_s1s2 = ot.emd2(sample_weights, target_sample_weights, M_s1s2)
        loss_t1t2 = ot.emd2(sample_weights, target_sample_weights, M_t1t2)

        loss = loss_s1t1 - 0.5 * (loss_s1s2 + loss_t1t2)
        return loss


if __name__ == "__main__":
    import inspect
    import sys

    instances = {}
    not_losses = ["RBF", "RQ", "DeepJDOT_Loss", "UnbiasedDeepJDOT_Loss", "CoralLoss"]
    # Iterate through all attributes in the current module
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        # Check if the attribute is a class and defined in the current module
        if inspect.isclass(obj) and obj.__module__ == __name__ and not (name in not_losses) :
            instances[name] = obj()
    n = 30
    instances["Sinkhorn"] = WassersteinLoss(reg=0.1*10)
    instances["unbiased_Wasserstein"] = WassersteinLoss(unbiased=True)

    import matplotlib.pyplot as plt

    # n_samples_sweep = {}
    # # checking the scaling against
    # fig, ax = plt.subplots()
    # for name, instance in instances.items():
    #     n_samples_sweep[name] = [instance(2*torch.ones(i, 10), torch.zeros(i, 10)) for i in range(1, n)]
    #     ax.plot(n_samples_sweep[name], label=name)
    # ax.legend()
    # plt.show()

    # n_dim_sweep = {}
    # # checking the scaling against
    # fig, ax = plt.subplots()
    # for name, instance in instances.items():
    #     n_dim_sweep[name] = [instance(2*torch.ones(10, n), torch.zeros(10, n)) for i in range(1, n)]
    #     ax.plot(n_dim_sweep[name], label=name)
    # ax.legend() 
    # plt.show()

    # Plot the evolution of the loss with translation/ scaling on gaussian samples
    fig, ax = plt.subplots()
    translation_scaling_sweep = {}
    from torch.distributions.multivariate_normal import MultivariateNormal
    batch_size = 64
    m = MultivariateNormal(torch.zeros(2), torch.eye(2))
    x = m.rsample((batch_size*100,))
    y = x.clone()
    y_target = (y * 10) + torch.ones_like(y)* 10
    # Initialize an array to store interpolated points
    interpolated_points = []
    for t in torch.linspace(0, 1, steps=n):
        interpolated_point = y + (y_target - y) * t
        interpolated_points.append(interpolated_point)

    def get_minibatch_estimates(x, y, distance, batch_size):
        distance_batch = [distance(x[(i-1)*(batch_size):i*batch_size], y[(i-1)*(batch_size):i*batch_size]) for i in range(1, x.shape[0]//batch_size)]
        return sum(distance_batch) / len(distance_batch)

    for name, instance in instances.items():
        # instantiate multivariate gaussian
        # translation_scaling_sweep[name] = [instance(x, interpolated_points[i]) for i in range(1, n)]
        translation_scaling_sweep[name] = [get_minibatch_estimates(x, interpolated_points[i], instance, batch_size) for i in range(0, n)]
        ax.plot(translation_scaling_sweep[name], label=name)

    ax.legend()
    plt.show()
a = 0