import torch
from torch import nn
import torch.nn.functional as F
import random
from tqdm import tqdm
class ConvertableEmbedding(nn.Embedding):
  """This acts like a normal embedding but when it sees a UE loaded with its name it converts it to a normal embedding and deletes the UE weights.

  """
  def __init__(self, num_embeddings: int, embedding_dim: int, front_embed_mul: float):
    super().__init__(num_embeddings, embedding_dim)
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.front_embed_mul = front_embed_mul
    self._register_load_state_dict_pre_hook(self.check_initialize)

  def check_initialize(self, state_dict:dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if f'{prefix}integration1.weight' in state_dict:
      with torch.no_grad():
        me = UnifiedEmbedding(self.num_embeddings, self.embedding_dim, self.front_embed_mul)
        new_state_dict = {k[len(prefix):]:v for k, v in state_dict.items()}
        me.load_state_dict(new_state_dict)
        all_embedding_idxs = torch.arange(0, self.num_embeddings, dtype=torch.long)
        all_embeddings = me(all_embedding_idxs, allow_reorder=False)
        state_dict.clear()
        state_dict[f'{prefix}weight'] = all_embeddings


class UnifiedEmbedding(nn.Embedding):
  def __init__(self, vocab_size: int, embed_dim: int, front_embed_mul: float, emb_training_epochs=50):
    super().__init__(vocab_size, embed_dim)
    self.vocab_size = vocab_size
    self.embedding_size = embed_dim
    self.emb_training_epochs = emb_training_epochs
    # The larger the 'front_embed_mul' is the better this works. 16 makes a big diff but eats a lot of mem.
    # If you use really large multiples consider training with keep_embed_on_cpu. It is slower but you save all that GPU memory.
    #
    # I have several theories why this works and some evidence. First, using this for positional embeddings doesn't get you far.
    # Also, a front_embed_mul of 1 does help a little. (I think. Been a while since I tested it)
    # That makes me think this could be helping low run embeddings learn from high run ones via the weights.
    # The second idea is just the information content allows the generated embedding to stay exactly on the verge of utility for a lot of situations.
    # This means it is able to optimize against all data a lot better.
    # No idea how to test that idea. Probably would show up in the distribution of the generated weights.
    #
    #Finally, clearly for real prod weights you would just generate a tensor and save it in the out state dict with the 'weight' name so normal embedding code could use it.
    #hmmm... actually, the 'norm' on embeddings isn't trained on these.
    # Is that just for backprop? Doing more than a lookup is a waste so I imagine they are saved as normed so this may work.
    # gotta test.
    #
    #Anyway, Prod could lose the integration weights and big tok_embed.weight so the weight structure would go back to normal and have no prod costs.
    front_embed_dim = int(embed_dim * front_embed_mul)
    self.integration1 = nn.Linear(embed_dim, front_embed_dim)
    self.integration2 = nn.Linear(front_embed_dim, embed_dim)
    self.ln = nn.LayerNorm(embed_dim)

  def forward(self, idxs: torch.Tensor, allow_reorder=True) -> torch.Tensor:
    output_device = idxs.device
    if allow_reorder and idxs.size(1) > 1:
      #This greatly saves computation/CPU transfer costs but clearly adds a little complexity.
      #It doesn't appear to hurt training (it shouldn't but quick tests were done and showed similar training curves)
      #Basically, we find all the unique idxs used, look just those up then do the integration layers and re-scatter the results.
      #This way common tokens only get a single cost of transfer and calculation.
      batch, sequence = idxs.size()
      local_idxs = idxs.reshape(-1)
      local_idxs = local_idxs.tolist()
      ordered_idxs = list(set(local_idxs))
      idx_locations = {real: idx for idx, real in enumerate(ordered_idxs)}
      ordered_idxs = torch.tensor(ordered_idxs, dtype=torch.long, device=output_device)
      x = super().forward(ordered_idxs)
      x = self.integration1(x)
      x = F.gelu(x)
      x = self.integration2(x)
      # Now we can re-lookup the result
      reorg_idxs = x[torch.tensor(tuple(idx_locations[idx] for idx in local_idxs), dtype=torch.long, device=idxs.device)]
      x = reorg_idxs.view(batch, sequence, -1)
    else:
      x = super().forward(idxs)
      x = self.integration1(x)
      x = F.gelu(x)
      x = self.integration2(x)
    x = self.ln(x)
    return x
