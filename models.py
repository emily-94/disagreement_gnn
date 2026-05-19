import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GlobalAttention
from torch_geometric.utils import softmax, degree, scatter



#if edge attention is not used, uniform weights over incoming edges per dst node are used instead.
def uniform_edge_attention(edge_index, device, num_nodes=None):
    """
    Uniform weights over ALL incoming edges per dst node:
      α_ij = 1 / |N_i|  for j in N_i

    """
    src, dst = edge_index
    if num_nodes is None:
        num_nodes = int(dst.max().item()) + 1 if dst.numel() else 0
    deg_dst = degree(dst, num_nodes=num_nodes, dtype=torch.float32).clamp(min=1.0)  # [N]
    alpha = (1.0 / deg_dst[dst]).to(device)                                         # [E]
    return alpha



class LSTMEncoder(nn.Module):
   
   def __init__(self, 
                in_dim, 
                hidden_dim, 
                out_dim, 
                bidirectional = True, 
                use_speaker_embedding = False,
                num_speakers = 2,
                speaker_emb_dim = 8,):
      super().__init__()
      
      self.use_speaker_embedding = use_speaker_embedding
      self.speaker_emb_dim = speaker_emb_dim if use_speaker_embedding else 0

      if self.use_speaker_embedding:
          self.speaker_emb = nn.Embedding(num_speakers, speaker_emb_dim)
      else:
          self.speaker_emb = None

      #create lstm layer
      self.lstm = nn.LSTM(
         input_size = in_dim + self.speaker_emb_dim, # concatenate speaker embedding if used
         hidden_size = hidden_dim,
         num_layers = 1,
         batch_first = True,
         bidirectional = bidirectional
      )
      
      #concatenation of forward and backward directions
      lstm_out = 2*hidden_dim if bidirectional else hidden_dim
      #learned combination of both
      self.proj = nn.Linear(lstm_out, out_dim)
      #normalize
      self.out_norm = nn.LayerNorm(out_dim) 

   def encode_conversation(self, X_g, order_idx, speaker_g = None):

      #sort the utterance indexes in the correct temporal order
      sort_idx = order_idx.argsort()
      #add batch dimension
      Xs = X_g[sort_idx]
      
      if self.use_speaker_embedding:
          if speaker_g is None:
              raise ValueError("speaker_g needed if use_speaker_embedding=True")
          spk = self.speaker_emb(speaker_g[sort_idx].long())
      
          Xs = torch.cat([Xs, spk], dim =-1)

      Xs = Xs.unsqueeze(0)

      #run lstm and output sequence of context-aware embeddings [1, T, hidden_dim*2]
      Ys, _ = self.lstm(Xs)
      #project to original dim
      Ys = self.proj(Ys)
      Ys = self.out_norm(Ys) 
      #reverse sorting - outputs returned in the same order as input utterances
      inv = sort_idx.argsort()
      #remove the batch dimension
      return Ys.squeeze(0)[inv]
   
   def forward(self, X, batch_vec, time_idx, speaker_ids = None):
      
      #number of utterances in batch
      N = X.size(0)
      #number of graphs
      B = int(batch_vec.max().item()) + 1 if N else 0

      #empty tensort for LSTM outputs for all utterances
      out = X.new_zeros(N, self.proj.out_features)
      
      #loop over all graphs
      for g in range(B):
         
         #get indices of utterances of current graph 
         idx = (batch_vec == g).nonzero(as_tuple=True)[0]
         if idx.numel() == 0: continue
         spk_g = speaker_ids[idx] if (self.use_speaker_embedding and speaker_ids is not None) else None
         out[idx] = self.encode_conversation(X[idx], time_idx[idx], spk_g)
         
      #context aware vectors for all utterances in batch
      return out


#rgcn layer 
class EdgeAttentionRGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations=2,
                 bias=True, aggr='add', activation=F.relu):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.activation = activation
        
        #weight matrix for each relation type and self=loop
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty((in_channels, out_channels)))
            for _ in range(num_relations)
        ])
        self.W_self = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        #initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W:
            nn.init.xavier_uniform_(w)
        nn.init.xavier_uniform_(self.W_self)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    


    def forward(self, x, edge_index, edge_type, alpha_all,
                return_info=False, save_messages=False):
       

        N = x.size(0) #get number of nodes
        out = x.new_zeros(N, self.out_channels) #empty tensor for node representations 
        

        info = {} if return_info else None #dct for attention info
        
        #loop over relation types (r = 0 for same speaker edges, r = 1 for different speaker edges)
        for r in range(self.num_relations):
            idx = (edge_type == r).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            
            #apply transformation for this relation type
            xW = x @ self.W[r]              # [N, Dout]
            #get edges of this relation type
            ei_r = edge_index[:, idx]       # [2, E_r]
            #split source and destination nodes
            src_r, dst_r = ei_r[0], ei_r[1]
            
            #get attention weights for this relation type
            alpha_r = alpha_all[idx].to(x.dtype)                 # [E_r]
            #compute messages and aggregate to dst nodes
            msg = alpha_r.view(-1, 1) * xW[src_r]                # [E_r, Dout]
            out.index_add_(0, dst_r, msg)

            #save attention info
            if return_info:
                info[f"r{r}_edge_index"] = ei_r.detach().cpu()
                info[f"r{r}_alpha"] = alpha_r.detach().cpu()
                if save_messages:
                    info[f"r{r}_msg_norm"] = msg.norm(dim=1).detach().cpu()
        
        #add self-loop contribution
        out = out + (x @ self.W_self)
        
        #add bias and apply activation
        if self.bias is not None:
            out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)

        return (out, info) if return_info else out



class RelationSimilarityEdgeAttention(nn.Module):
    """
    Similarity attention for edges:

        e_ij = g_i^T W_e g_j
        alpha_ij = softmax over incoming edges to dst node

    """
    def __init__(self, dim, tau=1.0, learnable_tau=False):
        super().__init__()
        
        #learnable transformation
        self.W_e = nn.Linear(dim, dim, bias=False)

        if learnable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))
        else:
            self.register_buffer("log_tau", torch.log(torch.tensor(float(tau))))

        self.reset_parameters()

    @property
    def tau(self):
        return torch.exp(self.log_tau)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_e.weight)

    def forward(self, g, edge_index, num_nodes=None):
        """
        g: [N, D]
        edge_index: [2, E]
        returns:
            alpha: [E], normalized over incoming edges per dst node
        """
        #split edges into source j and destination i 
        src, dst = edge_index
        
        #transform source node vectors
        g_j = self.W_e(g[src])              # [E, D]
        #get destination node vectors
        g_i = g[dst]                        # [E, D]
        
        #calculate similarity score for each edge
        e = (g_i * g_j).sum(dim=-1) / self.tau   # [E]
        #normalise scores over incoming edges per dst node to get attention weights
        alpha = softmax(e, dst)                  # [E]

        return alpha



#simple local GCN layer with no edge types and no attention
class LocalGCN(MessagePassing):

  def __init__(self, in_channels, out_channels, bias = True, activation=F.relu):

    super().__init__(aggr = 'add')
    self.W = nn.Parameter(torch.Tensor(in_channels, out_channels))
    self.W_self = nn.Parameter(torch.Tensor(in_channels, out_channels))
    self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
    self.activation = activation

    self.reset_parameters()


  def reset_parameters(self):

    nn.init.xavier_uniform_(self.W)
    nn.init.xavier_uniform_(self.W_self)
    if self.bias is not None:
      nn.init.zeros_(self.bias)


  def forward(self, h1, edge_index):

    #transform all nodes

    xW = h1 @ self.W    #[N, Dout]
    out = self.propagate(edge_index, x = xW, size = (h1.size(0), h1.size(0)))

    #add the self term
    out = out + (h1 @ self.W_self)

    if self.bias is not None:
      out = out + self.bias

    if self.activation is not None:
      out = self.activation(out)

    return out



  def message(self, x_j):

    return x_j
  


class DialogueGCNEncoder(nn.Module):
    """
    Stack of attention-weighted RGCN layers,
    followed by a local GCN layer
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim=None,
        num_relations=2,
        num_layers=2,
        dropout=0.2,
        activation=F.relu,
 
    ):
        super().__init__()

        

        out_dim = out_dim or hidden_dim

        dims = [in_dim] + [hidden_dim] * num_layers

        self.rgcn_layers = nn.ModuleList([
            EdgeAttentionRGCN(
                dims[i],
                dims[i + 1],
                num_relations=num_relations,
                bias=True,
                activation=activation,
            )
            for i in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)


        self.local = LocalGCN(hidden_dim, out_dim, bias=True, activation=activation)
        self.local_norm = nn.LayerNorm(out_dim)
    

    def forward(self, x, edge_index, edge_type, alpha_all, return_info=False, save_messages=False):
        info_all = [] if return_info else None

        h = x

        for layer, norm in zip(self.rgcn_layers, self.norms):
            if return_info:
                h, info = layer(
                    h,
                    edge_index,
                    edge_type,
                    alpha_all,
                    return_info=True,
                    save_messages=save_messages,
                )
                info_all.append(info)
            else:
                h = layer(h, edge_index, edge_type, alpha_all)

            h = self.dropout(norm(h))

        
            h = self.local(h, edge_index)
       

        h = self.dropout(self.local_norm(h))

        return (h, {"layers": info_all}) if return_info else h


class RegressionModel(nn.Module):
    def __init__(
        self,
        text_dim,
        audio_dim,
        model_dim=256,
        rgcn_hidden=128,
        rgcn_layers=2,
        rgcn_dropout=0.2,
        fusion_type="text",
        use_lstm=True,
        lstm_hidden=128,
        use_edge_attn=True,
        gnn_type="rgcn",
        pooling="global_attn",
        use_speaker_embedding=False,
        speaker_emb_dim=8,
        num_speakers=2,
    ):
         super().__init__()

         #flags
         self.fusion_type = fusion_type.lower()
         self.use_lstm = use_lstm
         self.use_edge_attn = use_edge_attn
         self.gnn_type = gnn_type.lower()
         self.pooling = pooling.lower()
         self.use_speaker_embedding = use_speaker_embedding
       
         print(f"Building RegressionModel: model_dim={model_dim}, rgcn_hidden={rgcn_hidden}, "
               f"rgcn_dropout={rgcn_dropout}, fusion_type={fusion_type}, gnn_type={gnn_type}")


         # --LSTM encoders

         if self.fusion_type == "text":
             
             self.encoder = LSTMEncoder(
                 in_dim = text_dim,
                 hidden_dim = lstm_hidden,
                 out_dim = text_dim, 
                 bidirectional = True,
                 use_speaker_embedding = use_speaker_embedding,
                 num_speakers = num_speakers,
                 speaker_emb_dim = speaker_emb_dim,
             )
             self.proj = nn.Linear(text_dim, model_dim)

         elif self.fusion_type == "audio":

             self.encoder = LSTMEncoder(
                 in_dim = audio_dim,
                 hidden_dim = lstm_hidden,
                 out_dim=audio_dim,
                 bidirectional=True,
                 use_speaker_embedding=use_speaker_embedding,
                 num_speakers=num_speakers,
                 speaker_emb_dim=speaker_emb_dim
             )
             self.proj = nn.Linear(audio_dim, model_dim)

         else:
             raise ValueError(f"Unsupported fusion_type: {fusion_type}")
         
         self.fused_norm = nn.LayerNorm(model_dim)


  
        #RGCN
         if self.gnn_type == 'rgcn':
            if self.use_edge_attn:
                self.edge_attn = RelationSimilarityEdgeAttention(
                    model_dim,
                    tau=1.0,
                    learnable_tau=False,
                )
            else:
                self.edge_attn = None

            self.rgcn = DialogueGCNEncoder(
               in_dim=model_dim,
               hidden_dim = rgcn_hidden,
               out_dim = model_dim,
               num_relations = 2,
               num_layers = rgcn_layers,
               dropout = rgcn_dropout,
            )

            
         else:

            self.edge_attn = None
            self.rgcn = None
        

        

         #use attention pooling for graph embedding 
         if self.pooling == "global_attn":
            pool_in_dim = model_dim
            gate_nn = nn.Sequential(
                nn.Linear(pool_in_dim, model_dim),
                nn.Tanh(),
                nn.Linear(model_dim, 1),
            )
            theta_nn = None
            self.pool = GlobalAttention(gate_nn=gate_nn, nn=theta_nn)

         else:
            self.pool = None

         
         head_in_dim = model_dim

         #regression head  
         self.head = nn.Sequential(
            nn.Linear(head_in_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(model_dim, 1),
            )


   
    def _fuse_nodes(self, batch):
        speaker_ids = batch.speakers if hasattr(batch, "speakers") else None

        if self.fusion_type == "text":
            x = batch.text
        elif self.fusion_type == "audio":
            x = batch.audio
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        if self.use_lstm:
            x = self.encoder(
                x,
                batch.batch,
                batch.time_idx,
                speaker_ids=speaker_ids,
            )

        return self.fused_norm(self.proj(x))
     



    def forward(self, batch, return_attn=False, save_messages=False):
        #lstm encoding and intialisaion of node representations
        g_nodes = self._fuse_nodes(batch)          # [N, d]
        B = batch.num_graphs if hasattr(batch, "num_graphs") else 1

        alpha_all = None
        gnn_info = None

        #if no GNN, go straight to  pooling
        if self.gnn_type == "none":
            h2_nodes = g_nodes


        # RGCN layers with edge attention 
        elif self.gnn_type == "rgcn":
            if self.use_edge_attn:
                alpha_all = self.edge_attn(g_nodes, batch.edge_index, batch.edge_type, num_nodes=g_nodes.size(0))
            
            #use uniform attention if edge attention is not used
            else:
                alpha_all = uniform_edge_attention(
                    batch.edge_index, g_nodes.device, num_nodes=g_nodes.size(0)
                )


            if return_attn:
                h2_nodes, gnn_info = self.rgcn(
                    g_nodes, batch.edge_index, batch.edge_type, alpha_all,
                    return_info=True, save_messages=save_messages
                )
            else:
                h2_nodes = self.rgcn(
                    g_nodes, batch.edge_index, batch.edge_type, alpha_all
                )

        else:
                    raise ValueError(f"Unknown gnn_type: {self.gnn_type}")
        

        H_all = h2_nodes
        batch_vec = batch.batch
        speaker_vec = batch.speakers if hasattr(batch, "speakers") else None
        

        #save node attention weights
        alpha_nodes = None

       
        if self.pooling == "global_attn":
            graph_emb = self.pool(H_all, batch_vec)      # [B, d]

            gate_logits = self.pool.gate_nn(H_all).view(-1)      # [N]
            alpha_nodes = softmax(gate_logits, batch_vec)        # [N]

        else:
            graph_emb = scatter(H_all, batch_vec, dim=0, reduce="mean")  # [B, d]

        out = self.head(graph_emb).squeeze(-1)            # [B]



        #returns edge and node attention weights if return_attn is True 
        if return_attn:
            extras = {
                "alpha_all": alpha_all.detach().cpu() if alpha_all is not None else None,
                "edge_index": batch.edge_index.detach().cpu(),
                "edge_type": batch.edge_type.detach().cpu() if hasattr(batch, "edge_type") else None,
                "batch": batch.batch.detach().cpu(),
                "gnn_info": gnn_info,
                "alpha_nodes": alpha_nodes.detach().cpu() if alpha_nodes is not None else None,
            }

            # detach any tensor in nested dict 
            for k, v in list(extras.items()):
                if isinstance(v, torch.Tensor):
                    extras[k] = v.detach().cpu()
                elif isinstance(v, dict):
                    for kk, vv in list(v.items()):
                        if isinstance(vv, torch.Tensor):
                            v[kk] = vv.detach().cpu()

            return out, extras

        return out

