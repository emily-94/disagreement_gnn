# train_cv.py (UPDATED)
import argparse
import os
import json

from datetime import datetime


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from utils import (
    seed_everything,
    get_device,
    EarlyStopping,
    train_one_epoch,
    evaluate,
    collect_predictions_single,
    collect_edge_attention,
    collect_node_attention,
)
from metrics import regression_metrics_df
from dataset_emotions import EmotionGNNDataset
from models import RegressionModel



# args
def parse_args():
    p = argparse.ArgumentParser(description="Train a GNN model with cross-validation")

    # data paths and settings
    p.add_argument("--meta_csv", type=str, default="data/file_data.csv")
    p.add_argument("--text_root", type=str, default="data/TextEmotionFeatures/New/Pooled")
    p.add_argument("--audio_root", type=str, default="data/AudioEmotionFeatures/Audeering/Unmerged")
    p.add_argument("--transcripts_root", type=str, default="data/Transcripts")
    p.add_argument("--window_size", type=int, default=5)

    # modality and ablations
    p.add_argument("--modality", type=str, default = "text", choices=["text", "audio"])
    p.add_argument("--use_lstm", type=int, default=1)
    p.add_argument("--use_edge_attn", type=int, default=1)
    p.add_argument("--pooling", type=str, default="global_attn", choices=["mean", "global_attn"])
    p.add_argument("--gnn_type", type=str, default="rgcn", choices=["none", "rgcn"])
  
    

    p.add_argument("--use_speaker_embedding", type=int, default=0,
               help="If 1, append a learned speaker embedding before the LSTM.")
    p.add_argument("--speaker_emb_dim", type=int, default=8,
               help="Dimension of the learned speaker embedding.")
    p.add_argument("--num_speakers", type=int, default=2,
               help="Number of speaker IDs available to the embedding layer.")



    # model hyperparameters
    p.add_argument("--model_dim", type=int, default=256)
    p.add_argument("--rgcn_hidden", type=int, default=256)
    p.add_argument("--rgcn_layers", type=int, default=1)
    p.add_argument("--rgcn_dropout", type=float, default=0.2)
    p.add_argument("--lstm_hidden", type=int, default=128)

   


    # training hyperparameters
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    
    p.add_argument("--aux_weight", type=float, default=0.0,
               help="Weight for auxiliary loss (Option C). 0 disables it.")

    #training info  
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n_runs", type=int, default=1)

    return p.parse_args()






#crete experiment directories
def make_exp_dir(args):
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = args.run_tag.strip() if args.run_tag else stamp
    exp_name = f"exp_disagree_{args.gnn_type}_{args.modality}_{tag}"
    exp_dir = os.path.join(args.out_dir, exp_name)

    subdirs = {
        "root": exp_dir,
        "checkpoints": os.path.join(exp_dir, "checkpoints"),
        "predictions": os.path.join(exp_dir, "predictions"),
        "metrics": os.path.join(exp_dir, "metrics"),
        "debug": os.path.join(exp_dir, "debug"),
    }
    for pth in subdirs.values():
        os.makedirs(pth, exist_ok=True)
    return subdirs





#load metadata for the dataset inclyding ids, labels and fold assignments. 
def load_meta(meta_csv):
    meta = pd.read_csv(meta_csv)
    allowed_ids = meta["File_name"].astype(str).tolist()
    labels_by_id = dict(zip(meta["File_name"].astype(str), meta["Rating"].astype(float)))
    folds_by_id = dict(zip(meta["File_name"].astype(str), meta["Fold"].astype(int)))
    return allowed_ids, labels_by_id, folds_by_id


#build the dataset using EmotionGNNDataset
def build_dataset(args, allowed_ids, labels_by_id, folds_by_id):
    return EmotionGNNDataset(
        text_root=args.text_root,
        audio_root=args.audio_root,
        transcripts_root=args.transcripts_root,
        allowed_ids=allowed_ids,
        modality=args.modality,


        #set window size for the dataset
        window_size=args.window_size,

        labels_by_id=labels_by_id,
        folds_by_id=folds_by_id,
  
    )


#build the model using RegressionModel and specified hyperparameters
def build_model(args, text_dim, audio_dim,device="cpu"):


    model = RegressionModel(
        text_dim=text_dim,
        audio_dim=audio_dim,
        model_dim=args.model_dim,
        rgcn_hidden=args.rgcn_hidden,
        rgcn_layers=args.rgcn_layers,
        rgcn_dropout=args.rgcn_dropout,
        fusion_type=args.modality,
        lstm_hidden=args.lstm_hidden,
        use_lstm=bool(args.use_lstm),
        use_edge_attn=bool(args.use_edge_attn),
        gnn_type=args.gnn_type,
        pooling=args.pooling,
        use_speaker_embedding=bool(args.use_speaker_embedding),
        speaker_emb_dim=args.speaker_emb_dim,
        num_speakers=args.num_speakers,
    )

    return model.to(device)

#create PyG dataloader from dataset and list of indices

def make_loader(ds, idx, args, shuffle=False):
    return DataLoader(
        Subset(ds, idx),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )

#split train indice into train/val indices
def split_train_val(train_idx, val_frac, seed):
    idx = np.array(train_idx, dtype=int)
    np.random.default_rng(seed).shuffle(idx)

    n_val = max(1, int(len(idx) * val_frac))
    return idx[n_val:].tolist(), idx[:n_val].tolist()

#store the experiment settings
def metadata(args, run, seed, fold):
    return {
        "run": run,
        "seed": seed,
        "cv_test_fold": fold,
        "modality": args.modality,
        "gnn_type": args.gnn_type,
        "pooling": args.pooling,
        "use_lstm": args.use_lstm,
        "use_edge_attn": args.use_edge_attn,
        "hp_lr": args.lr,
        "hp_weight_decay": args.weight_decay,
        "hp_model_dim": args.model_dim,
        "hp_rgcn_hidden": args.rgcn_hidden,
        "hp_rgcn_dropout": args.rgcn_dropout,
        "hp_window_size": args.window_size,
        "hp_rgcn_layers": args.rgcn_layers,
        "hp_lstm_hidden": args.lstm_hidden,
    }


def save_outputs(args, paths, model, df_test, test_loader, run, seed, fold, device):
    pred_dir = os.path.join(paths["predictions"], f"fold_{fold:02d}")
    ckpt_dir = os.path.join(paths["checkpoints"], f"fold_{fold:02d}")
    attn_dir = os.path.join(pred_dir, "attention")

    for d in [pred_dir, ckpt_dir, attn_dir]:
        os.makedirs(d, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"run_{run:02d}_best.pt"))

    df_test.to_csv(
        os.path.join(pred_dir, f"run_{run:02d}_test_predictions.csv"),
        index=False,
    )

    meta = metadata(args, run, seed, fold)

    if args.gnn_type == "rgcn":
        edge_df = collect_edge_attention(model, test_loader, device=device, topk=None)
        edge_df.assign(**meta).to_csv(
            os.path.join(attn_dir, f"run_{run:02d}_test_edge_attention_all.csv"),
            index=False,
        )

    if args.pooling == "global_attn":
        node_df = collect_node_attention(model, test_loader, device=device, topk=None)
        node_df.assign(**meta).to_csv(
            os.path.join(attn_dir, f"run_{run:02d}_test_node_attention_all.csv"),
            index=False,
        )

def main():
    args = parse_args()
    paths = make_exp_dir(args)

    print("Experiment folder:", os.path.abspath(paths["root"]))

    with open(os.path.join(paths["debug"], "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = get_device()
    print(f"Device: {device}")

    allowed_ids, labels_by_id, folds_by_id = load_meta(args.meta_csv)

    #dataset
    ds = build_dataset(args, allowed_ids, labels_by_id, folds_by_id)

    sample = ds[0]
    #get textt and audio dim
    text_dim = sample.text.size(-1)
    audio_dim = sample.audio.size(-1)
    print(f"Text dim={text_dim}, Audio dim={audio_dim}, Total clips={len(ds)}")
    
    #check that all conv_ids have fold assignments
    missing = [cid for cid in ds.conv_ids if cid not in folds_by_id]
    assert not missing, f"Missing fold assignments for {len(missing)} conv_ids: {missing[:5]}"

    all_folds = sorted({folds_by_id[cid] for cid in ds.conv_ids})
    idx_by_fold = {
        fold: [i for i, cid in enumerate(ds.conv_ids) if folds_by_id[cid] == fold]
        for fold in all_folds
    }

    loss_fn = nn.MSELoss()
    all_cv_rows = []

    for run in range(args.n_runs):
        seed = args.seed + run
        seed_everything(seed)

        print(f"\n========== RUN {run} seed={seed} ==========")

        run_rows = []

        for test_fold in all_folds:
            test_idx = idx_by_fold[test_fold]
            train_idx = [
                i
                for fold, idxs in idx_by_fold.items()
                if fold != test_fold
                for i in idxs
            ]

            tr_idx, val_idx = split_train_val(
                train_idx,
                args.val_frac,
                seed + test_fold,
            )

            print(
                f"\n[fold={test_fold}] "
                f"train={len(tr_idx)} val={len(val_idx)} test={len(test_idx)}"
            )

            train_loader = make_loader(ds, tr_idx, args, shuffle=True)
            val_loader = make_loader(ds, val_idx, args)
            test_loader = make_loader(ds, test_idx, args)

            model = build_model(args, text_dim, audio_dim, device)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            stopper = EarlyStopping(patience=args.patience, mode="min")

            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    loss_fn,
                    opt,
                    grad_clip=args.grad_clip,
                    device=device,
                )

                val_loss, val_mae = evaluate(
                    model,
                    val_loader,
                    loss_fn,
                    device=device,
                )

                print(
                    f"[run={run} fold={test_fold}] ep {epoch:02d} | "
                    f"train={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_MAE={val_mae:.4f}"
                )

                if stopper.step(val_loss, model):
                    print("Early stop triggered.")
                    model.load_state_dict(stopper.state)
                    break

            df_test = collect_predictions_single(
                model,
                test_loader,
                device=device,
            )

            df_test = df_test.assign(**metadata(args, run, seed, test_fold))
            run_rows.append(df_test)

            save_outputs(
                args,
                paths,
                model,
                df_test,
                test_loader,
                run,
                seed,
                test_fold,
                device,
            )

        all_cv_rows.append(pd.concat(run_rows, ignore_index=True))

    cv_df = pd.concat(all_cv_rows, ignore_index=True)

    out_csv = os.path.join(paths["predictions"], "all_runs_predictions.csv")
    cv_df.to_csv(out_csv, index=False)
    print(f"\nSaved predictions to {out_csv} ({len(cv_df)} rows)")

    metrics = regression_metrics_df(cv_df, "y_true_rating", "y_pred_rating")

    metrics_path = os.path.join(paths["metrics"], "overall_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("CV Overall Results:", metrics)
    print("METRICS_JSON=" + json.dumps(metrics))


if __name__ == "__main__":
    main()