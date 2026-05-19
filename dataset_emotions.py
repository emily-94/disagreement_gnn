# dataset_gnn.py
import os, numpy as np, torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import glob




def _to_speaker_ids(speakers):
    mapping = {"R": 0, "C": 1}
    return torch.tensor([mapping[str(s)] for s in speakers], dtype=torch.long)

def get_edge_index(
    speakers: torch.Tensor,
    past_window: int = 10,
    future_window: int = 10,
    temporal_types: bool = True,
    speaker_4types: bool = True,   # DialogueGCN uses speaker-pair types (RR/RC/CR/CC)
    add_self_edges: bool = True,
):
    """
    Build directed edges:
      For each target node i, add incoming edges j -> i for j in [i-w, ..., i-1] 
    Relation types depend on:
      - speaker pair (src speaker -> dst speaker)
      - temporal direction (src in past of dst vs src in future of dst)
    """
    N = int(speakers.numel())
    edge_map = {}

    def base_type(s_src: int, s_dst: int) -> int:
        if not speaker_4types:
            # 2 types: same vs cross
            return 0 if (s_src == s_dst) else 1
        # 4 types: RR, RC, CR, CC for dyads (0/1)
        return s_src * 2 + s_dst  # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3

    B = 4 if speaker_4types else 2  # number of speaker-pair base relations

    for i in range(N):
        j_min = max(0, i - past_window)
        j_max = min(N - 1, i + future_window)

        for j in range(j_min, j_max + 1):
            if (not add_self_edges) and (j == i):
                continue

            s_src = int(speakers[j])
            s_dst = int(speakers[i])
            b = base_type(s_src, s_dst)

            if temporal_types:
                # DialogueGCN direction is relative order between u_j and u_i.
                # "past" if j <= i, "future" if j > i
                direction = 0 if (j <= i) else 1
                r = b + direction * B      # total relations = 2B (e.g., 8 when B=4)
            else:
                r = b

            # IMPORTANT: store edge as (src=j, dst=i) to make it explicit that i aggregates from window
            edge_map[(j, i)] = r

    pairs = sorted(edge_map.keys())
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    edge_type  = torch.tensor([edge_map[p] for p in pairs], dtype=torch.long)
    return edge_index, edge_type

class EmotionGNNDataset(Dataset):
    """
    One item = one conversation graph 

    Input: 
      - Text per-utterance features in:
          text_root / {conv_id}_{segment}.npz
      - Audio per-utterance features in:
          audio_root / {conv_id}_{segment}.npz
      - Transcript CSVs in:
          transcripts_root / {conv_id}_words_utterances.csv
     

      - labels_by_id[conv_id]  -> disagreement rating
      - folds_by_id[conv_id]   -> fold index

    PyG Data object:
      - data.text       : [T, Dt]
      - data.audio      : [T, Da]
      - data.speakers   : [T] (0=R, 1=C)
      - data.edge_index : [2, E]
      - data.edge_type  : [E] (0 same, 1 cross)
      - data.time_idx   : [T] (0..T-1)
      - data.y          : scalar label 
      - data.fold       : int
      - data.conv_id    : string, labels conversation id
      - data.num_nodes  : T
    """

    def __init__(self,
             text_root,
             audio_root,
             transcripts_root,
             labels_by_id=None,
             folds_by_id=None,
             allowed_ids=None,
             modality="text",
             window_size=5):
        

        assert modality in ("text", "audio")
  

        self.text_root = text_root
        self.audio_root = audio_root
        self.transcripts_root = transcripts_root

        self.labels = labels_by_id or {}
        self.folds  = folds_by_id or {}

        self.modality = modality
    
      

        self.window_size = int(window_size)
     


        # conv_ids we will load (from meta)
        conv_ids = [str(c) for c in allowed_ids]
        # filter to those that actually have transcript + required features (by modality)
        kept_ids = []
        skipped_missing_transcript = []
        skipped_missing_feats = []

        for cid in conv_ids:
            trans_path = os.path.join(transcripts_root, f"{cid}_words_utterances.csv")
            if not os.path.exists(trans_path):
                skipped_missing_transcript.append(cid)
                continue

            pattern_t = os.path.join(text_root,  f"{cid}_*.npz")
            pattern_a = os.path.join(audio_root, f"{cid}_*.npz")
            has_text  = len(glob.glob(pattern_t)) > 0
            has_audio = len(glob.glob(pattern_a)) > 0

            # Require the modalities you're actually training with
            ok = has_text if self.modality == "text" else has_audio

            if not ok:
                skipped_missing_feats.append(cid)
                continue

            kept_ids.append(cid)

        if not kept_ids:
            raise RuntimeError(
                f"EmotionGNNDataset: no conversations found after filtering "
                f"(modality={self.modality})."
            )

        self.conv_ids = sorted(kept_ids)

        print(f"[DATA] modality={self.modality}")
        print(f"[DATA] kept {len(self.conv_ids)}/{len(conv_ids)} conversations")
        print(f"[DATA] skipped missing transcript: {len(skipped_missing_transcript)}")
        print(f"[DATA] skipped missing required feats: {len(skipped_missing_feats)}")


    def __len__(self):
        return len(self.conv_ids)

    def _load_segment_vec(self, root, conv_id, segment):


        conv_id = str(conv_id).strip()
        root = os.fspath(root)

        if isinstance(segment, np.generic):
            segment = segment.item()
        if isinstance(segment, float):
            if np.isnan(segment):
                raise ValueError(f"segment is NaN for conv_id={conv_id}")
            segment = int(segment)

        segment_str = str(int(segment))  # force clean int string
        path = os.path.join(root, f"{conv_id}_{segment_str}.npz")

        if not os.path.isfile(path):
            return None

        try:

            with open(path, "rb") as fh:
                with np.load(fh, allow_pickle=False) as f:
                    for key in ("vec", "audio_embeddings", "text_embeddings",
                                "hidden_states", "emb", "arr_0"):
                        if key in f:
                            arr = f[key]
                            break
                    else:
                        raise KeyError(f"No expected array key in {path}. Found keys: {list(f.keys())}")

            arr = np.asarray(arr, dtype=np.float32).squeeze()
            return arr

        except Exception as e:
            print("\n[NP.LOAD FAILED]")
            print("conv_id:", conv_id)
            print("segment:", segment, type(segment))
            print("path repr:", repr(path))
            print("path len:", len(path))
            print("exists:", os.path.exists(path))
            raise

    

    def __getitem__(self, idx):
        conv_id = self.conv_ids[idx]

        trans_path = os.path.join(
            self.transcripts_root,
            f"{conv_id}_words_utterances.csv"
        )
        trans_df = pd.read_csv(trans_path)

        feat_root = self.text_root if self.modality == "text" else self.audio_root

        vecs = []
        spk_list = []

        for row_idx, utt in trans_df.iterrows():
            segment = row_idx + 1
            vec = self._load_segment_vec(feat_root, conv_id, segment)

            if vec is None:
                continue

            vecs.append(np.asarray(vec, dtype=np.float32).reshape(-1))
            spk_list.append(str(utt.speaker))

        if not vecs:
            raise RuntimeError(f"{conv_id} has no usable {self.modality} features.")

        x = torch.tensor(np.stack(vecs), dtype=torch.float32)
        speakers = _to_speaker_ids(spk_list)

        T = x.size(0)

        if self.modality == "text":
            text = x
            audio = torch.zeros((T, 1), dtype=torch.float32)
        else:
            audio = x
            text = torch.zeros((T, 1), dtype=torch.float32)

        edge_index, edge_type = get_edge_index(
            speakers,
            past_window=self.window_size,
            future_window=self.window_size,
            temporal_types=False,
            speaker_4types=False,
            add_self_edges=True,
        )

        data = Data(
            text=text,
            audio=audio,
            speakers=speakers,
            edge_index=edge_index,
            edge_type=edge_type,
            time_idx=torch.arange(T, dtype=torch.long),
            y=torch.tensor([self.labels.get(conv_id, float("nan"))], dtype=torch.float32),
            fold=torch.tensor([self.folds.get(conv_id, -1)], dtype=torch.long),
        )

        data.num_nodes = T
        data.conv_id = conv_id



        return data