"""
model.py
--------
NSCLC-FusionNet: an uncertainty-aware heterogeneous graph network with
conditional diffusion imputation for multimodal lung cancer
characterization.

Architecture (one graph per patient):
    - modality encoders  (Phase 1, frozen in Phase 2)
    - conditional diffusion imputer  (Phase 2)
    - heterogeneous graph with gated attention  (Phase 2)
    - seven evidential / Cox / focal heads  (Phase 2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 3
K_REGIONS = 4
G_PROGRAMS = 6
N_MODALITIES = 5          # wsi, ct, pet, rna, clin
DIFFUSION_T = 1000
DIFFUSION_HIDDEN = 256
N_DIFFUSION_LAYERS = 4

# ----------------------------------------------------------------------
# 1. Modality encoders (Phase 1)
# ----------------------------------------------------------------------
class WSIEncoder(nn.Module):
    """Mean-pooled UNI patch features + soft assignment to K region prototypes."""
    def __init__(self, in_dim=1024, d=D_MODEL, k=K_REGIONS):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.prototypes = nn.Parameter(torch.randn(k, d))
        self.attn = nn.Linear(d, 1)

    def forward(self, patches):  # (N_patches, 1024)
        if patches is None:
            return None
        h = self.proj(patches)                    # (N, d)
        # soft assignment
        sim = F.normalize(h, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        assign = F.softmax(sim * 10.0, dim=0)     # (N, K)
        b = F.softmax(self.attn(h).squeeze(-1), dim=0)  # (N,)
        # region nodes: sum_p assign[p,k] * b[p] * h[p]
        w = (assign * b.unsqueeze(-1)).unsqueeze(-1)    # (N, K, 1)
        regions = (w * h.unsqueeze(1)).sum(0) / (w.sum(0) + 1e-8)  # (K, d)
        return regions                               # (K, d)


class VolumeEncoder(nn.Module):
    """Small 3D CNN for CT/PET; replace with 3D ResNet-50 in production."""
    def __init__(self, in_ch=1, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),    nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(32, d),
        )

    def forward(self, vol):  # (1, 1, 128, 128, 128)
        if vol is None:
            return None
        return self.net(vol)   # (1, d)


class RNAEncoder(nn.Module):
    """50 Hallmark scores -> 6 program nodes."""
    def __init__(self, in_dim=50, d=D_MODEL, g=G_PROGRAMS):
        super().__init__()
        self.progs = nn.ModuleList([nn.Linear(in_dim, d) for _ in range(g)])

    def forward(self, rna):  # (50,)
        if rna is None:
            return None
        return torch.stack([p(rna) for p in self.progs], 0)  # (G, d)


class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim=5, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, clin):  # (5,)
        if clin is None:
            return None
        return self.net(clin).unsqueeze(0)  # (1, d)


# ----------------------------------------------------------------------
# 2. Conditional diffusion imputer (Phase 2)
# ----------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionTransformer(nn.Module):
    """
    A small transformer denoiser. Input tokens = noisy target nodes
    (augmented with modality-type and timestep embeddings). Cross-attends
    to context tokens (observed modalities).
    """
    def __init__(self, d=D_MODEL, n_heads=N_HEADS, n_layers=N_DIFFUSION_LAYERS):
        super().__init__()
        self.t_emb = SinusoidalPosEmb(d)
        self.mod_emb = nn.Embedding(N_MODALITIES, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,
                                           dim_feedforward=2 * d, dropout=0.1,
                                           batch_first=True)
        self.blocks = nn.ModuleList([layer for _ in range(n_layers)])
        self.out = nn.Linear(d, d)

    def forward(self, x_t, t, mod_id, context):
        # x_t: (B, N_target, d);  context: (B, N_ctx, d)
        tok = x_t + self.t_emb(t).unsqueeze(1) + self.mod_emb(mod_id).unsqueeze(1)
        for blk in self.blocks:
            tok = blk(tok, src=context) if context is not None else blk(tok)
        return self.out(tok)


class ConditionalDiffusionImputer(nn.Module):
    """
    Wraps the diffusion transformer with a forward noising process and a
    sampling routine. In practice, use a proper diffusion library
    (e.g. diffusers) for training; this class exposes the interface.
    """
    def __init__(self, d=D_MODEL, n_steps=DIFFUSION_T):
        super().__init__()
        self.d = d
        self.n_steps = n_steps
        self.denoiser = DiffusionTransformer(d=d)
        betas = torch.cos(torch.linspace(0, math.pi / 2, n_steps + 1)) ** 2
        self.register_buffer("alphas_cumprod",
                             torch.cumprod(1 - betas[:-1], dim=0))

    def q_sample(self, x0, t, noise):
        a = self.alphas_cumprod[t].view(-1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, context, mod_id, n_nodes, n_samples=8, n_steps=25):
        """
        Returns:
            mean: (1, n_nodes, d)   imputed node embedding
            var : (1, n_nodes, d)   per-dimension variance across samples
        """
        B = context.shape[0]
        outs = []
        for _ in range(n_samples):
            x = torch.randn(B, n_nodes, self.d, device=context.device)
            for step in reversed(range(n_steps)):
                t = torch.full((B,), step, dtype=torch.long, device=context.device)
                pred = self.denoiser(x, t, mod_id, context)
                x = pred  # DDIM step (simplified)
            outs.append(x)
        outs = torch.stack(outs, 0)          # (S, B, N, d)
        return outs.mean(0), outs.var(0)


# ----------------------------------------------------------------------
# 3. Heterogeneous graph layer with gated attention (Phase 2)
# ----------------------------------------------------------------------
class GatedHeteroLayer(nn.Module):
    def __init__(self, d=D_MODEL, n_heads=N_HEADS, n_types=4):
        super().__init__()
        self.d = d
        self.h = n_heads
        self.q = nn.Linear(d, d * n_types)
        self.k = nn.Linear(d, d * n_types)
        self.v = nn.Linear(d, d * n_types)
        self.o = nn.Linear(d * n_types, d)
        self.n_types = n_types

    def forward(self, x, adj, node_type, gate):
        # x: (N, d);  adj: (N, N);  gate: (N,) reliability in (0,1]
        N, d = x.shape
        hd = d // self.h
        Q = self.q(x).view(N, self.n_types, self.h, hd)
        K = self.k(x).view(N, self.n_types, self.h, hd)
        V = self.v(x).view(N, self.n_types, self.h, hd)

        # type-specific attention
        att = torch.einsum("itnh,jtnh->ijtnh", Q, K) / math.sqrt(hd)
        # gated logit: multiply by gate of the *target* node
        att = att + torch.log(gate.clamp(min=1e-3))[None, :, None, None]
        att = att.masked_fill(adj[..., None, None, None] == 0, float("-inf"))
        att = torch.softmax(att, dim=1)                        # (N, N, T, H, hd)
        out = torch.einsum("ijtnh,jtnh->itnh", att, V)         # (N, T, H, hd)
        out = out.reshape(N, -1)
        return self.o(out)


# ----------------------------------------------------------------------
# 4. Evidential heads (Phase 2)
# ----------------------------------------------------------------------
class EvidentialHead(nn.Module):
    def __init__(self, d=D_MODEL, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(d, n_classes)

    def forward(self, x, reliability):
        # x: (d,)  reliability: scalar in (0,1]
        evidence = F.softplus(self.fc(x)) * reliability + 1.0
        alpha = evidence + 1.0
        S = alpha.sum(-1, keepdim=True)
        prob = alpha / S
        u = alpha.shape[-1] / S
        return prob.squeeze(0), u.squeeze(0), alpha.squeeze(0)


class CoxHead(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


# ----------------------------------------------------------------------
# 5. The full model
# ----------------------------------------------------------------------
class NSCLCFusionNet(nn.Module):
    def __init__(self,
                 d=D_MODEL,
                 k=K_REGIONS,
                 g=G_PROGRAMS,
                 freeze_encoders=True):
        super().__init__()
        self.d = d
        self.k = k
        self.g = g

        # --- Phase 1 encoders ---------------------------------------
        self.wsi_enc  = WSIEncoder(d=d, k=k)
        self.ct_enc   = VolumeEncoder(d=d)
        self.pet_enc  = VolumeEncoder(d=d)
        self.rna_enc  = RNAEncoder(d=d, g=g)
        self.clin_enc = ClinicalEncoder(d=d)

        if freeze_encoders:
            for p in (self.wsi_enc, self.ct_enc, self.pet_enc,
                      self.rna_enc, self.clin_enc):
                for param in p.parameters():
                    param.requires_grad = False

        # --- Phase 2 imputer ----------------------------------------
        self.imputer = ConditionalDiffusionImputer(d=d)

        # --- Phase 2 graph ------------------------------------------
        self.graph_layers = nn.ModuleList([
            GatedHeteroLayer(d=d) for _ in range(N_LAYERS)
        ])

        # patient readout node
        self.patient_token = nn.Parameter(torch.randn(1, 1, d))

        # --- Phase 2 heads ------------------------------------------
        self.head_subtype = EvidentialHead(d, 2)
        self.head_T       = EvidentialHead(d, 4)
        self.head_N       = EvidentialHead(d, 2)
        self.head_driver  = EvidentialHead(d, 3)
        self.head_stage   = EvidentialHead(d, 2)
        self.head_cox     = CoxHead(d)
        self.head_rec     = nn.Linear(d, 1)

        # learned reliability coefficient
        self.kappa = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------
    def _encode(self, batch):
        """Return a dict of modality node tensors (None if missing)."""
        nodes = {}
        wsi_list = []
        for p in batch["wsi"]:
            wsi_list.append(self.wsi_enc(p) if p is not None else None)
        nodes["wsi"] = wsi_list

        ct = batch["ct"][0] if isinstance(batch["ct"], list) else batch["ct"]
        pet = batch["pet"][0] if isinstance(batch["pet"], list) else batch["pet"]
        nodes["ct"]  = self.ct_enc(ct)
        nodes["pet"] = self.pet_enc(pet)

        rna = batch["rna"][0] if isinstance(batch["rna"], list) else batch["rna"]
        nodes["rna"] = self.rna_enc(rna)

        nodes["clin"] = self.clin_enc(batch["clin"][0])
        return nodes

    # ------------------------------------------------------------------
    def _impute_if_missing(self, nodes, mask):
        """
        For each missing modality, run the diffusion imputer conditioned on
        the observed modalities of the same patient.
        """
        observed = [m for m, ok in zip(nodes, mask) if ok == 1.0]
        if len(observed) == 0:
            return nodes, {m: torch.ones(1) for m in nodes}

        # build context = concat of observed nodes
        ctx_list = []
        for m in observed:
            v = nodes[m]
            if v is None:
                continue
            if v.dim() == 2:  # (N, d)
                ctx_list.append(v)
            else:
                ctx_list.append(v.unsqueeze(0))
        context = torch.cat(ctx_list, 0).unsqueeze(0)  # (1, N_ctx, d)

        gates = {}
        for i, m in enumerate(nodes):
            if mask[i] == 1.0:
                gates[m] = torch.ones(1)
                continue
            # number of target nodes for this modality
            if m == "wsi":
                n_nodes = self.k
            elif m == "rna":
                n_nodes = self.g
            else:
                n_nodes = 1
            mod_id = torch.tensor([i], dtype=torch.long)
            mean, var = self.imputer.sample(context, mod_id, n_nodes,
                                            n_samples=8, n_steps=25)
            nodes[m] = mean.squeeze(0)                 # (n_nodes, d)
            u = var.mean().item()
            gates[m] = torch.tensor([math.exp(-self.kappa.item() * u)])
        return nodes, gates

    # ------------------------------------------------------------------
    def _build_graph(self, nodes, gates, mask):
        """
        Assemble a single graph with all modality nodes + patient token.
        Returns:
            x:       (N_total, d)
            adj:     (N_total, N_total)
            ntype:   (N_total,)
            gate:    (N_total,)
            idx_map: dict modality -> (start, end)
        """
        all_nodes, all_gates, all_types = [], [], []
        idx_map = {}
        cursor = 0
        # modality index in the canonical order
        order = ["wsi", "ct", "pet", "rna", "clin"]
        for t, m in enumerate(order):
            v = nodes[m]
            if v is None:
                idx_map[m] = (cursor, cursor)
                continue
            if v.dim() == 1:
                v = v.unsqueeze(0)
            n = v.shape[0]
            all_nodes.append(v)
            g = gates.get(m, torch.ones(1))
            all_gates.append(g.expand(n))
            all_types.append(torch.full((n,), t, dtype=torch.long))
            idx_map[m] = (cursor, cursor + n)
            cursor += n

        x = torch.cat(all_nodes, 0)                       # (N, d)
        gate = torch.cat(all_gates, 0).to(x.dtype)        # (N,)
        ntype = torch.cat(all_types, 0)

        N = x.shape[0]
        adj = torch.eye(N, dtype=x.dtype)
        # structural edges: all nodes connect to every other node
        adj = torch.ones(N, N, dtype=x.dtype)
        # prior edges (co-occurrence in development cohort):
        #   wsi  <-> rna,  clin
        #   ct   <-> pet, rna, clin
        #   rna  <-> clin
        prior_pairs = [
            ("wsi", "rna"), ("wsi", "clin"),
            ("ct", "pet"), ("ct", "rna"), ("ct", "clin"),
            ("rna", "clin"), ("pet", "rna"),
        ]
        for a, b in prior_pairs:
            (s1, e1), (s2, e2) = idx_map[a], idx_map[b]
            adj[s1:e1, s2:e2] = 1.0
            adj[s2:e2, s1:e1] = 1.0
        return x, adj, ntype, gate, idx_map

    # ------------------------------------------------------------------
    def forward(self, batch):
        mask = batch["mask"][0]                # (5,)
        nodes = {k: v[0] if isinstance(v, list) else v
                 for k, v in self._encode(batch).items()}
        nodes, gates = self._impute_if_missing(nodes, mask)

        x, adj, ntype, gate, idx_map = self._build_graph(nodes, gates, mask)

        # patient token
        pt = self.patient_token[0]
        x = torch.cat([pt, x], 0)
        N = x.shape[0]
        adj = torch.zeros(N, N)
        adj[0, 1:] = 1.0
        adj[1:, 0] = 1.0
        # prior edges between modality nodes
        for a, b in [("wsi", "rna"), ("wsi", "clin"), ("ct", "pet"),
                     ("ct", "rna"), ("ct", "clin"), ("rna", "clin"),
                     ("pet", "rna")]:
            (s1, e1), (s2, e2) = idx_map[a], idx_map[b]
            adj[1 + s1:1 + e1, 1 + s2:1 + e2] = 1.0
            adj[1 + s2:1 + e2, 1 + s1:1 + e1] = 1.0
        gate = torch.cat([torch.ones(1), gate], 0)
        ntype = torch.cat([torch.full((1,), 4, dtype=torch.long), ntype], 0)

        for layer in self.graph_layers:
            x = layer(x, adj, ntype, gate) + x     # residual

        patient_emb = x[0]                          # (d,)
        reliability = gate.mean()

        out = {}
        for name, head in [
            ("subtype", self.head_subtype),
            ("T",       self.head_T),
            ("N",       self.head_N),
            ("driver",  self.head_driver),
            ("stage",   self.head_stage),
        ]:
            prob, unc, alpha = head(patient_emb, reliability)
            out[name] = {"prob": prob, "uncertainty": unc, "alpha": alpha}
        out["survival"] = self.head_cox(patient_emb)
        out["recurrence"] = torch.sigmoid(self.head_rec(patient_emb)).squeeze(-1)
        return out


# ----------------------------------------------------------------------
# 6. Multi-task loss (uncertainty weighting)
# ----------------------------------------------------------------------
class MultiTaskLoss(nn.Module):
    def __init__(self, n_tasks=8):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0.0
        for i, L in enumerate(losses):
            total = total + L / (2 * self.log_sigma[i].exp() ** 2) + self.log_sigma[i]
        return total


# ----------------------------------------------------------------------
# 7. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 1
    batch = {
        "wsi":   [torch.randn(64, 1024)],
        "ct":    [torch.randn(1, 128, 128, 128)],
        "pet":   [None],
        "rna":   [torch.randn(50)],
        "clin":  torch.randn(B, 5),
        "mask":  torch.tensor([[1., 1., 0., 1., 1.]]),
        "labels": [{"subtype": 0, "T": 1, "N": 0, "driver": 0, "stage": 0}],
        "survival": torch.tensor([[24.0, 1.0]]),
        "recurrence": [(None, None)],
    }
    model = NSCLCFusionNet(freeze_encoders=False)
    out = model(batch)
    print("subtype prob  :", out["subtype"]["prob"])
    print("survival risk :", out["survival"])
    print("recurrence    :", out["recurrence"])
