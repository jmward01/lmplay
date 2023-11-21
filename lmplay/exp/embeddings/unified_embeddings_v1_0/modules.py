import torch
from torch import nn
import torch.nn.functional as F


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
        me = UnifiedEmbed(self.num_embeddings, self.embedding_dim, self.front_embed_mul)
        new_state_dict = {k[len(prefix):]:v for k, v in state_dict.items()}
        me.load_state_dict(new_state_dict)
        all_embedding_idxs = torch.arange(0, self.num_embeddings, dtype=torch.long)
        all_embeddings = me(all_embedding_idxs, allow_reorder=False)
        state_dict.clear()
        state_dict[f'{prefix}weight'] = all_embeddings


class UnifiedEmbed(nn.Module):
  def __init__(self, vocab_size: int, embed_dim: int, front_embed_mul: float, keep_embed_on_cpu=False):
    super().__init__()
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
    self.tok_embed = nn.Embedding(vocab_size, front_embed_dim)
    #This only works if the main model has overridden the 'to' call to check for it. See LMBase.
    if keep_embed_on_cpu:
      self.tok_embed.weight.force_device = "cpu"

    self.integration1 = nn.Linear(front_embed_dim, embed_dim)
    self.integration2 = nn.Linear(embed_dim, embed_dim)

  def forward(self, idxs: torch.Tensor, allow_reorder=True) -> torch.Tensor:
    tok_embed_device = self.tok_embed.weight.device
    output_device = idxs.device
    if allow_reorder and idxs.size(1) > 1:

      batch, sequence = idxs.size()
      local_idxs = idxs.reshape(-1)
      local_idxs = local_idxs.tolist()
      ordered_idxs = list(set(local_idxs))
      idx_locations = {real: idx for idx, real in enumerate(ordered_idxs)}
      if tok_embed_device != output_device:
        #This probably means the embeddings are on CPU to save memory.
        #Autocast doesn't like mixed devices.
        #What likes this even less is gradient scaling. I haven't figured out how to get it to ignore the CPU weights.
        #This means that amp stuff is really just here as a placeholder for the possible future where gradient scaling is figured out.
        with torch.amp.autocast(enabled=False, device_type=tok_embed_device.type):
          ordered_idxs = torch.tensor(ordered_idxs, dtype=torch.long, device=tok_embed_device)
          x = self.tok_embed(ordered_idxs)
          x = x.to(output_device)
      else:
        ordered_idxs = torch.tensor(ordered_idxs, dtype=torch.long, device=tok_embed_device)
        x = self.tok_embed(ordered_idxs)
      # Minimize the lookup/liner layer costs
      x = self.integration1(x)
      x = F.gelu(x)
      x = self.integration2(x)
      # Now we can re-lookup the result
      reorg_idxs = x[torch.tensor(tuple(idx_locations[idx] for idx in local_idxs), dtype=torch.long, device=idxs.device)]
      x = reorg_idxs.view(batch, sequence, -1)
    else:
      if tok_embed_device != output_device:
        #This probably means the embeddings are on CPU to save memory.
        #Autocast doesn't like mixed devices
        with torch.amp.autocast(enabled=False, device_type=tok_embed_device.type):
          x = self.tok_embed(idxs)
          x = x.to(output_device)
      else:
        x = self.tok_embed(idxs)
      x = self.integration1(x)
      x = F.gelu(x)
      x = self.integration2(x)

    return x
