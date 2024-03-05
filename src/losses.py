import torch
import torch.functional as F
import torch.nn as nn
import ot


class RBF(nn.Module):

    def __init__(self, n_kernels=6, mul_factor=2.0, bandwidth=None):
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


class MMDLossBandwith(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y, **kwargs):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class MMDLoss(nn.Module):
    def __init__(self, sigma=32):
        super(MMDLoss, self).__init__()
        self.sigma = sigma
        self.counter = 0

    def calculate_mean_distance(self, mnist_dataloader, usps_dataloader):
        total_distance = 0
        count = 0

        # Assurez-vous que les DataLoader retournent des lots de la même taille
        # ou gérez les cas où les derniers lots pourraient être de tailles différentes.
        for mnist_images, usps_images in zip(mnist_dataloader, usps_dataloader):
            # Aplatir les images
            mnist_flat = mnist_images.view(mnist_images.shape[0], -1)
            usps_flat = usps_images.view(usps_images.shape[0], -1)

            # Calculer les distances euclidiennes
            distances = torch.cdist(mnist_flat, usps_flat, p=2)

            # Ajouter la distance moyenne de ce lot
            total_distance += distances.mean()
            count += 1
            # Calculer la distance moyenne globale
            mean_distance = total_distance / count
            return mean_distance.item()

    def gram_RBF(self, x, y, sigma):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]
        x = x.unsqueeze(1)  # Shape: [x_size, 1, dim]
        y = y.unsqueeze(0)  # Shape: [1, y_size, dim]
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        squared_diff = (tiled_x - tiled_y) ** 2
        squared_dist = torch.sum(squared_diff, -1)  # Sum over the feature dimension
        return torch.exp(-squared_dist / sigma)

    def forward(self, input, target, sigma=32, **kwargs):
        if input.grad_fn is not None:
            self.sigma = self.calculate_mean_distance(input, target)
        XX = self.gram_RBF(input, input, sigma)
        YY = self.gram_RBF(target, target, sigma)
        XY = self.gram_RBF(input, target, sigma)
        loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
        if input.grad_fn is not None:
            self.counter += 1
        return loss


class WassersteinLoss(nn.Module):
    def __init__(self, reg=1, unbiased=False):
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
        y_source = kwargs["y_source"]
        y_target = kwargs["preds_target"]
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
        y_source = kwargs["y_source"]
        y_target = kwargs["preds_target"]
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
                (len(embedd) // 2,), 1.0 / len(embedd) // 2, device=embedd.device
            )
        if target_sample_weights is None:
            target_sample_weights = torch.full(
                (len(embedd_target) // 2,),
                1.0 / len(embedd_target) // 2,
                device=embedd_target.device,
            )
        loss_s1t1 = ot.emd2(sample_weights, target_sample_weights, M_s1t1)
        loss_s1s2 = ot.emd2(sample_weights, target_sample_weights, M_s1s2)
        loss_t1t2 = ot.emd2(sample_weights, target_sample_weights, M_t1t2)

        loss = loss_s1t1 - 0.5 * (loss_s1s2 + loss_t1t2)
        return loss


if __name__ == "__main__":
    X = torch.randn(64, 9000).to("cuda")
    print(X)
    Y = torch.randn(64, 9000).to("cuda")
    loss = MMDLossBandwith()
    val = loss(X, Y)
