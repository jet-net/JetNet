from typing import Union, Tuple

import torch
from torch import nn, Tensor
import numpy as np


class EMDLoss(nn.Module):
    """
    Calculates the energy mover's distance between two batches of jets differentiably as a convex optimization problem
    either through the linear programming library ``cvxpy`` or by converting it to a quadratic programming problem and
    using the ``qpth`` library. ``cvxpy`` is marginally more accurate but ``qpth`` is significantly faster so defaults
    to ``qpth``.

    **JetNet must be installed with the extra option** ``pip install jetnet[emdloss]`` **to use this.**

    *Note: PyTorch <= 1.9 has a bug which will cause this to fail for >= 32 particles. This PR should fix this from 1.10 onwards*
    https://github.com/pytorch/pytorch/pull/61815.

    Args:
        method (str): 'cvxpy' or 'qpth'. Defaults to 'qpth'.
        num_particles (int): number of particles per jet - onlyneeds to be specified if method is 'cvxpy'.
        qpth_form (str): 'L2' or 'QP'. Defaults to 'L2'.
        qpth_l2_strength (float): regularization parameter for 'L2' qp form. Defaults to 0.0001.
        device (str): 'cpu' or 'cuda'. Defaults to 'cpu'.

    """

    def __init__(
        self,
        method: str = "qpth",
        num_particles: int = None,
        qpth_form: str = "L2",
        qpth_l2_strength: float = 0.0001,
        device: str = "cpu",
    ):
        super(EMDLoss, self).__init__()

        if method == "qpth":
            try:
                global qpth
                qpth = __import__("qpth", globals(), locals())
            except:
                print(
                    "QPTH needs to be installed separately to use this method - try pip install jetnet[emdloss]"
                )
                raise
        else:
            try:
                global cp, cvxpylayers
                cp = __import__("cvxpy", globals(), locals())
                cvxpylayers = __import__("cvxpylayers", globals(), locals())
            except:
                print(
                    "cvxpy needs to be installed separately to use this method - try pip install jetnet[emdloss]"
                )
                raise

        assert method == "qpth" or method == "cvxpy", "invalid method type"
        assert method != "cvxpy" or (
            num_particles is not None and num_particles > 0
        ), "num_particles must be specified to use 'cvxpy' method"
        assert qpth_form == "L2" or qpth_form == "QP", "invalid qpth form"
        assert device == "cpu" or device == "cuda", "invalid device type"

        self.num_particles = num_particles
        self.method = method
        if method == "qpth":
            self.form = qpth_form
            self.l2_strength = qpth_l2_strength
        self.device = device

        if method == "cvxpy":
            x = cp.Variable(num_particles * num_particles)  # flows
            c = cp.Parameter(num_particles * num_particles)  # costs
            w = cp.Parameter(num_particles + num_particles)  # weights
            Emin = cp.Parameter(1)  # min energy out of the two jets

            g1 = np.zeros((num_particles, num_particles * num_particles))
            for i in range(num_particles):
                g1[i, i * num_particles : (i + 1) * num_particles] = 1
            g2 = np.concatenate([np.eye(num_particles) for i in range(num_particles)], axis=1)
            g = np.concatenate((g1, g2), axis=0)

            constraints = [x >= 0, g @ x <= w, cp.sum(x) == Emin]
            objective = cp.Minimize(c.T @ x)
            problem = cp.Problem(objective, constraints)

            self.cvxpylayer = cvxpylayers.torch.CvxpyLayer(
                problem, parameters=[c, w, Emin], variables=[x]
            ).to(device)

    def _emd_inference_qpth(
        self, distance_matrix: Tensor, weight1: Tensor, weight2: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Using the QP solver QPTH to get EMDs (LP problem), adapted from https://github.com/icoz69/DeepEMD/blob/master/Models/models/emd_utils.py
        One can transform the LP problem to QP, or omit the QP term by multiplying it with a small value, i.e. l2_strngth.

        Args:
            distance_matrix (Tensor): nbatch * element_number * element_number.
            weight1 (Tensor): nbatch  * weight_number.
            weight2 (Tensor): nbatch  * weight_number.

        Returns:
            emd distance: nbatch*1
            flow : nbatch * weight_number *weight_number
        """
        nbatch = distance_matrix.shape[0]
        nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]
        nelement_weight1 = weight1.shape[1]
        nelement_weight2 = weight2.shape[1]

        # reshape dist matrix too (nbatch, 1, n1 * n2)
        Q_1 = distance_matrix.view(-1, 1, nelement_distmatrix).double()

        if (
            self.form == "QP"
        ):  # converting to QP - after testing L2 reg performs marginally better than QP
            # version: QTQ
            Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double() + 1e-4 * torch.eye(
                nelement_distmatrix
            ).double().unsqueeze(0).repeat(
                nbatch, 1, 1
            )  # 0.00001 *
            p = torch.zeros(nbatch, nelement_distmatrix).double().to(self.device)
        elif self.form == "L2":  # regularizing a trivial Q term with l2_strength
            # version: regularizer
            Q = (
                (self.l2_strength * torch.eye(nelement_distmatrix).double())
                .unsqueeze(0)
                .repeat(nbatch, 1, 1)
                .to(self.device)
            )
            p = distance_matrix.view(nbatch, nelement_distmatrix).double()
        else:
            raise ValueError("Unkown form")

        # h = [0 ... 0 w1 w2]
        h_1 = torch.zeros(nbatch, nelement_distmatrix).double().to(self.device)
        h_2 = torch.cat([weight1, weight2], 1).double()
        h = torch.cat((h_1, h_2), 1)

        G_1 = (
            -torch.eye(nelement_distmatrix)
            .double()
            .unsqueeze(0)
            .repeat(nbatch, 1, 1)
            .to(self.device)
        )
        G_2 = (
            torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix])
            .double()
            .to(self.device)
        )
        # sum_j(xij) = si
        for i in range(nelement_weight1):
            G_2[:, i, nelement_weight2 * i : nelement_weight2 * (i + 1)] = 1
        # sum_i(xij) = dj
        for j in range(nelement_weight2):
            G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1

        # xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
        G = torch.cat((G_1, G_2), 1)
        A = torch.ones(nbatch, 1, nelement_distmatrix).double().to(self.device)
        b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).double()
        flow = qpth.qp.QPFunction(verbose=-1)(Q, p, G, h, A, b)

        energy_diff = torch.abs(torch.sum(weight1, dim=1) - torch.sum(weight2, dim=1))

        emd_score = torch.sum((Q_1).squeeze() * flow, 1)
        emd_score += energy_diff

        return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)

    def forward(
        self, jets1: Tensor, jets2: Tensor, return_flows: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Calculate EMD between ``jets1`` and ``jets2``.

        Args:
            jets1 (Tensor): tensor of shape ``[num_jets, num_particles, num_features]``, with features in order ``[eta, phi, pt]``.
            jets2 (Tensor): tensor of same format as ``jets1``.
            return_flows (bool): return energy flows between particles in each jet. Defaults to False.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]:
            - **Tensor**: EMD scores tensor of shape [num_jets].
            - **Tensor** *Optional*, if ``return_flows`` is True: tensor of flows between particles of shape
              ``[num_jets, num_particles, num_particles]``.

        """
        assert (len(jets1.shape) == 3) and (len(jets2.shape) == 3), "Jets shape incorrect"
        assert jets1.shape[0] == jets2.shape[0], "jets1 and jets2 have different numbers of jets"
        assert (jets1.shape[1] == self.num_particles) and (
            jets2.shape[1] == self.num_particles
        ), "jets don't have num_particles particles"

        if self.method == "cvxpy":
            diffs = -(jets1[:, :, :2].unsqueeze(2) - jets2[:, :, :2].unsqueeze(1)) + 1e-12
            dists = torch.norm(diffs, dim=3).view(-1, self.num_particles * self.num_particles)

            weights = torch.cat((jets1[:, :, 2], jets2[:, :, 2]), dim=1)

            E1 = torch.sum(jets1[:, :, 2], dim=1)
            E2 = torch.sum(jets2[:, :, 2], dim=1)

            Emin = torch.minimum(E1, E2).unsqueeze(1)
            EabsDiff = torch.abs(E2 - E1).unsqueeze(1)

            (flows,) = self.cvxpylayer(dists, weights, Emin)

            emds = torch.sum(dists * flows, dim=1) + EabsDiff
        elif self.method == "qpth":
            diffs = -(jets1[:, :, :2].unsqueeze(2) - jets2[:, :, :2].unsqueeze(1)) + 1e-12
            dists = torch.norm(diffs, dim=3)

            emds, flows = self._emd_inference_qpth(dists, jets1[:, :, 2], jets2[:, :, 2])

        return (emds, flows) if return_flows else emds
