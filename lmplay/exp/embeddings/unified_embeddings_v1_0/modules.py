import torch
from torch import nn
import torch.nn.functional as F
import random
from tqdm import tqdm
from lmplay.utils import create_linear

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


class UnifiedEmbedding(nn.Module):
  def __init__(self,
               vocab_size: int,
               embed_dim: int,
               front_embed_mul: float,
               keep_embed_on_cpu=False,
               emb_training_epochs=50,
               ln=False,
               gelu=False,
               integration1_5=False,
               integration2=True,
               linear=nn.Linear):
    """UEs are a better way to train embeddings. This is a drop in replacement for nn.Embedding

    :param vocab_size:
    :param embed_dim: This is the output size that your network will see.
    :param front_embed_mul: Bigger is better. This is the hidden embedding size used just for training.
    :param keep_embed_on_cpu: Big front embeddings take a lot of memory. Storing them on the CPU has a performance hit but as the batch size grows the hit is minimized. Basically, do things on the CPU to be able to train larger networks.
    :param emb_training_epochs: This is only used for converting a normal embedding into a UE. This allows you to 'bolt on' a UE to an existing well-trained network that doesn't have the sacrificial UE network. It works but there is still a bit of model shock when UEs first start training.
    :param ln: LN the value before returning. Just trying different approaches here.
    """
    super().__init__()
    self.vocab_size = vocab_size
    self.embedding_size = embed_dim
    self.emb_training_epochs = emb_training_epochs
    # The larger the 'front_embed_mul' is the better this works. 16 makes a big diff but eats a lot of mem.
    # If you use really large multiples consider training with keep_embed_on_cpu. It is slower but you save all that GPU memory.
    #
    # I have several theories why this works and some evidence. First, using this for positional embeddings doesn't get you far.
    # Also, a front_embed_mul of 1 does help a little. (I think. Been a while since I tested it)
    # Basically, I think this helps the main network not get disrupted by low-run tokens dragging it back to an eralier state.
    # It also allows the tokens to not shift due to a changing network so they can just keep storing more and more information without disruption.
    #
    #Finally, clearly for real prod weights you would just generate a tensor and save it in the out state dict with the 'weight' name so normal embedding code could use it.
    #Testing shows that the performance of frozen saved embeddings is exactly the same as using the UE training weight versions, without the overhead ofcourse.
    #
    #Anyway, Prod could lose the integration weights and big tok_embed.weight so the weight structure would go back to normal and have no prod costs.
    front_embed_dim = int(embed_dim * front_embed_mul)
    self.tok_embed = nn.Embedding(vocab_size, front_embed_dim)
    self.integration1 = create_linear(linear, "ue_integration1", front_embed_dim, embed_dim)
    if integration1_5:
      self.integration1_5 = create_linear(linear, "ue_integration15", embed_dim, embed_dim)
    else:
      self.register_parameter('integration1_5', None)
    #This only works if the main model has overridden the 'to' call to check for it. See LMBase.
    if keep_embed_on_cpu:
      #We are keeping both the embedding and the first integration on the CPU to save memory and reduce the memory transfer.
      #by doing the first integration we transfer much less information from the cpu to the gpu
      #The tradeoff generally becomes worth it when sequence*batch_size > 2k. At that point the transfer saving cover the linear layer costs.
      #Of course this is totally dependent on cpu->gpu bandwidth, how fast your CPU is, CPU memory bandwidth, etc etc so milage will vary.
      #Broken for a lot of the exps. Gotta do a new way for all this stuff.
      to_cpu = [self.tok_embed.weight]
      to_cpu.append(self.integration1.weight)
      to_cpu.append(self.integration1.bias)
      for p in to_cpu:
        p.force_device = "cpu"
        #the secondary_optimizer is needed to allow AMP to work on CUDA. Without it AMP gets confused with the mix of CPU and GPU parameters.
        p.secondary_optimizer = True


    if integration2:
      self.integration2 = create_linear(linear, "ue_integration2", embed_dim, embed_dim)
    else:
      self.register_parameter('integration2', None)
    if ln:
      self.ln = nn.LayerNorm(embed_dim)
    else:
      self.ln = lambda x:x
    if gelu == True:
      self.emb_activation = F.gelu
    else:
      self.emb_activation = lambda x:x
    self._register_load_state_dict_pre_hook(self.check_initialize)
    self.initialized_from_small_embed = False

  def initialize(self, embedding_w: torch.Tensor):
    print("Initializing UEs from an existing embedding layer.")
    # The embedding is our source of 'truth' we are trying to learn to predict.
    device = embedding_w.device
    with torch.no_grad():
      # Gives a huge hint to speed up learning
      self.tok_embed.weight[:, :embedding_w.size(1)] = embedding_w

    # These hyper parameters still need fine-tuning. It works reasonably well though.
    lr = 5e-3
    weight_decay = 0.00
    # more epochs seem to hurt which is odd. More testing needed here.
    epochs = self.emb_training_epochs
    batch_size = 1024 * 8
    final_batch_size = 1024 * 8
    optimizer = torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    training_set = list(range(self.vocab_size))
    embedding = nn.Embedding(self.vocab_size, self.embedding_size, _weight=embedding_w)
    with torch.enable_grad():
      with tqdm(total=self.vocab_size * epochs) as pbar:
        for epoch in range(epochs):
          random.shuffle(training_set)
          last = 0
          batch_size = min(batch_size + 128, final_batch_size)
          while last < len(training_set):
            optimizer.zero_grad()
            batch = training_set[last:last + batch_size]
            batch = torch.tensor(batch, dtype=torch.long, device=device)
            last = last + len(batch)
            truth = embedding(batch)
            prediction = self(batch, allow_reorder=False)
            loss = F.mse_loss(prediction, truth, reduction="mean")
            loss.backward()
            loss = float(loss)
            optimizer.step()
            pbar.set_description(f"loss: {loss:0.3f}, epoch:{epoch}, lr:{scheduler.get_last_lr()[0]}")
            pbar.update(len(batch))
          if epoch > 0 and epoch % 10 == 0:
            scheduler.step()
    self.initialized_from_small_embed = True

  def check_initialize(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if f'{prefix}weight' in state_dict:
      self.initialize(state_dict[f'{prefix}weight'])

  def forward(self, idxs: torch.Tensor, allow_reorder=True) -> torch.Tensor:
    tok_embed_device = self.tok_embed.weight.device
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
      if tok_embed_device != output_device:
        #This probably means the embeddings are on CPU to save memory.
        #Autocast doesn't like mixed devices.
        #What likes this even less is gradient scaling. I haven't figured out how to get it to ignore the CPU weights.
        #This means that amp stuff is really just here as a placeholder for the possible future where gradient scaling is figured out.
        with torch.amp.autocast(enabled=False, device_type=tok_embed_device.type):
          ordered_idxs = torch.tensor(ordered_idxs, dtype=torch.long, device=tok_embed_device)
          x = self.emb_activation(self.tok_embed(ordered_idxs))

          x = self.integration1(x)
          x = x.to(output_device)
      else:
        ordered_idxs = torch.tensor(ordered_idxs, dtype=torch.long, device=tok_embed_device)
        x = self.emb_activation(self.tok_embed(ordered_idxs))

        x = self.integration1(x)
      if not self.integration1_5 is None:
        x = self.integration1_5(F.gelu(x))

      #x = self.integration1(x)
      # Minimize the lookup/liner layer costs
      if not self.integration2 is None:
        x = self.integration2(F.gelu(x))
      # Now we can re-lookup the result
      reorg_idxs = x[torch.tensor(tuple(idx_locations[idx] for idx in local_idxs), dtype=torch.long, device=idxs.device)]
      x = reorg_idxs.view(batch, sequence, -1)
    else:
      if tok_embed_device != output_device:
        #This probably means the embeddings are on CPU to save memory.
        #Autocast doesn't like mixed devices
        with torch.amp.autocast(enabled=False, device_type=tok_embed_device.type):
          x = self.emb_activation(self.tok_embed(idxs))

          if not self.integration1 is None:
            x = self.integration1(x)
          x = x.to(output_device)
      else:
        x = self.emb_activation(self.tok_embed(idxs))

        if not self.integration1 is None:
          x = self.integration1(x)
      if not self.integration1_5 is None:
        x = self.integration1_5(F.gelu(x))
      #x = self.integration1(x)
      if not self.integration2 is None:
        x = self.integration2(F.gelu(x))

    return self.ln(x)
