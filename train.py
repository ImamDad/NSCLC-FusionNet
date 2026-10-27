"""
train.py
--------
Minimal training loop demonstrating how dataloader.py and model.py fit
together. Swap in your own CV loop and hyperparameter search.
"""

import torch
from torch.utils.data import DataLoader
from dataloader import NSCLCFusionDataset, collate_fn, PATHS
from model import NSCLCFusionNet, MultiTaskLoss


def make_loader(cohort, batch_size=1, shuffle=True):
    ds = NSCLCFusionDataset(cohort)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NSCLCFusionNet(freeze_encoders=False).to(device)
    loss_fn = MultiTaskLoss().to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()),
                            lr=1e-4, weight_decay=1e-5)

    loader = make_loader("tcga_luad")
    for epoch in range(1):
        for batch in loader:
            for k in ("ct", "pet", "rna"):
                if isinstance(batch[k], list):
                    batch[k] = [x.to(device) if torch.is_tensor(x) else x
                                for x in batch[k]]
            batch["wsi"] = [x.to(device) if torch.is_tensor(x) else x
                            for x in batch["wsi"]]
            batch["clin"] = batch["clin"].to(device)
            batch["mask"] = batch["mask"].to(device)

            out = model(batch)
            # dummy losses — replace with proper evidential / Cox / focal losses
            L_sub  = -torch.log(out["subtype"]["prob"][0] + 1e-8)
            L_cox  = out["survival"].mean()
            L_rec  = out["recurrence"].mean()
            losses = [L_sub, L_cox, L_rec] + [torch.tensor(0.0, device=device)] * 5
            total  = loss_fn(losses)
            opt.zero_grad(); total.backward(); opt.step()
            print("loss:", float(total))
            break


if __name__ == "__main__":
    main()
