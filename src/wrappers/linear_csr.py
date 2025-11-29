# import torch

# class LinearCSRForward(torch.nn.Module):
#     """Minimal CSR forward wrapper: y = x @ W^T + b, with W stored as CSR."""
#     def __init__(self, W_dense: torch.Tensor, bias: torch.Tensor | None = None):
#         super().__init__()
#         self.register_buffer("W_csr", W_dense.to_sparse_csr())
#         self.register_buffer("bias", bias if bias is not None else None)

#     # W_csr: [out_features, in_features] (CSR sparse)
#     # x:     [batch, in_features]       (dense)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         W = self.W_csr  # rester sparse
#         try:
#             # S @ D -> D  (exploite MKL/cuSPARSE selon device/build)
#             out = torch.matmul(W, x.T).T               # [out,b] -> [b,out]
#         except RuntimeError:
#             # Fallback si l’opération sparse n’est pas dispo
#             out = x @ W.to_dense().T
#         if self.bias is not None:
#             out = out + self.bias
#         return out

#####################################################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class LinearCSRForward(nn.Module):
#     """
#     Drop-in replacement pour nn.Linear, mais qui conserve un poids en CSR
#     (pour les métriques) et utilise pour l'instant un chemin dense pour le forward.

#     - `weight_in` peut être dense ou CSR.
#     - On stocke:
#         * weight_csr : pour compter nnz, vérifier la sparsité, etc.
#         * weight_dense : utilisée pour le forward avec F.linear (stable et rapide).
#     """

#     def __init__(self, weight_in: torch.Tensor, bias: torch.Tensor | None = None):
#         super().__init__()

#         # 1) Stockage CSR pour les métriques
#         if weight_in.layout == torch.sparse_csr:
#             weight_csr = weight_in.detach().clone()
#         else:
#             weight_csr = weight_in.detach().to_sparse_csr()

#         self.register_buffer("weight_csr", weight_csr)

#         out_features, in_features = weight_csr.shape
#         self.in_features = in_features
#         self.out_features = out_features

#         # 2) Copie dense pour le forward (F.linear)
#         #    -> même comportement qu'un nn.Linear classique
#         weight_dense = weight_csr.to_dense()
#         # buffer = pas de gradient, on est en "inference only"
#         self.register_buffer("weight_dense", weight_dense)

#         # 3) Biais optionnel
#         if bias is not None:
#             # on garde un paramètre figé (requires_grad=False par défaut en inference)
#             self.register_parameter(
#                 "bias",
#                 nn.Parameter(bias.detach().clone(), requires_grad=False),
#             )
#         else:
#             self.bias = None

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [batch_size, in_features]
#         Retourne: [batch_size, out_features]
#         """
#         return F.linear(x, self.weight_dense, self.bias)

####################################################################################### 
# import torch
# import torch.nn as nn


# class LinearCSRForward(nn.Module):
#     """
#     Couche linéaire qui fait F * x avec un poids en CSR,
#     mais qui garde aussi des méta-données pour l'analyse
#     (nombre de paramètres denses, nnz, sparsité, etc.).
#     """

#     def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
#         """
#         `weight` est un tenseur DENSE déjà pruné, de shape [out_features, in_features].
#         On le convertit en CSR pour le forward, mais on garde les stats denses.
#         """
#         super().__init__()

#         # --- 1) Stats "denses" AVANT compression -------------------------
#         assert weight.dim() == 2, "LinearCSRForward attend un poids [out, in]"

#         out_features, in_features = weight.shape
#         dense_total = out_features * in_features
#         dense_nnz = int((weight != 0).sum().item())

#         self.meta_out_features = out_features
#         self.meta_in_features = in_features
#         self.meta_total_params = int(dense_total)      # nb de poids dans la matrice dense
#         self.meta_nnz = int(dense_nnz)                 # nb de poids non nuls
#         self.meta_sparsity = 1.0 - self.meta_nnz / self.meta_total_params

#         # On garde la forme dense sous forme de buffer (pour l'analyse)
#         self.register_buffer(
#             "meta_dense_shape",
#             torch.tensor([out_features, in_features], dtype=torch.long),
#             persistent=False,
#         )

#         # --- 2) Stockage CSR pour le forward -----------------------------
#         # to_sparse_csr garde la même shape (out,in) mais ne stocke que les nnz
#         weight_csr = weight.to_sparse_csr()
#         # On le stocke comme buffer (pas comme Parameter, on n'entraîne pas ça)
#         self.register_buffer("weight_csr", weight_csr, persistent=False)

#         # Le biais reste un Parameter (ou rien)
#         if bias is not None:
#             self.bias = nn.Parameter(bias.detach())
#         else:
#             self.bias = None

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [batch, in_features]
#         output: [batch, out_features]

#         On calcule (x @ W^T) + b, avec W en CSR.
#         """
#         # sparse_csr (out,in)  x^T (in,b) -> (out,b) puis transpose -> (b,out)
#         W = self.weight_csr
#         out = torch.matmul(W, x.T).T

#         if self.bias is not None:
#             out = out + self.bias

#         return out

#########################################################################################################
import torch
import torch.nn as nn


class LinearCSRForward(nn.Module):
    """
    Couche linéaire avec poids stocké en CSR.

    - Forward : (x @ W^T) + b, comme nn.Linear
    - Stats : garde nnz, total, sparsité, shape d'origine
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()

        assert weight.dim() == 2, "LinearCSRForward attend un poids [out_features, in_features]"
        out_features, in_features = weight.shape

        # --- Stats denses AVANT compression ---
        dense_total = out_features * in_features
        dense_nnz = int((weight != 0).sum().item())

        self.meta_out_features = out_features
        self.meta_in_features = in_features
        self.meta_total_params = int(dense_total)
        self.meta_nnz = int(dense_nnz)
        self.meta_sparsity = 1.0 - self.meta_nnz / self.meta_total_params

        self.register_buffer(
            "meta_dense_shape",
            torch.tensor([out_features, in_features], dtype=torch.long),
            persistent=False,
        )

        # --- Poids en CSR pour le forward ---
        weight_csr = weight.to_sparse_csr()
        self.register_buffer("weight_csr", weight_csr, persistent=False)

        # Biais (on peut garder Parameter, même si on n'entraîne pas)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        out: (..., out_features)
        """
        W = self.weight_csr

        # S'assurer que tout est sur le même device
        if W.device != x.device:
            W = W.to(x.device)

        # Aplatir sur la dernière dimension comme nn.Linear
        orig_shape = x.shape                       # (..., in_features)
        in_features = self.meta_in_features
        out_features = self.meta_out_features

        x_flat = x.view(-1, in_features)           # (N, in_features)

        # W: (out,in), x_flat.T: (in,N) -> (out,N) -> (N,out)
        out_flat = torch.matmul(W, x_flat.T).T     # (N, out_features)

        if self.bias is not None:
            out_flat = out_flat + self.bias

        out = out_flat.view(*orig_shape[:-1], out_features)
        return out

    # Petites helpers pour l'analyse
    @property
    def dense_total_params(self):
        return self.meta_total_params

    @property
    def dense_nnz(self):
        return self.meta_nnz

    @property
    def dense_sparsity(self):
        return self.meta_sparsity
