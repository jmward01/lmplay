"""Embedding implementations including experimental Unified Embeddings.

This module provides embedding layers for transformer models with a focus on
the Unified Embedding (UE) architecture. UEs are designed to improve training
stability and performance by using a larger intermediate embedding space during
training that can be collapsed to standard embeddings for inference.

Key components:
- ConvertableEmbedding: Standard embedding that can load UE weights
- UnifiedEmbedding: Advanced embedding with sacrificial training parameters

The Unified Embedding approach addresses issues with rare token updates disrupting
the model by providing a more stable embedding representation through an
over-parameterized training architecture.
"""

import torch
from torch import nn
import torch.nn.functional as F
import random
from tqdm import tqdm
from lmplay.utils import create_linear

__all__ = ['ConvertableEmbedding', 'UnifiedEmbedding']

class ConvertableEmbedding(nn.Embedding):
  """Standard embedding layer that can automatically convert from Unified Embeddings.
  
  This class extends nn.Embedding to provide compatibility with Unified Embeddings.
  When loading a checkpoint that contains UE weights, it automatically converts
  them to standard embedding weights by computing all token embeddings through
  the UE network and storing them as a regular embedding table.
  
  This allows models to use UEs during training for better stability while
  deploying with standard embeddings for efficiency.
  
  Attributes:
      num_embeddings (int): Size of the vocabulary.
      embedding_dim (int): Dimension of the embedding vectors.
      front_embed_mul (float): Multiplier used by the UE during training.
          Stored for compatibility when loading UE checkpoints.
  """
  def __init__(self, num_embeddings: int, embedding_dim: int, front_embed_mul: float):
    super().__init__(num_embeddings, embedding_dim)
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.front_embed_mul = front_embed_mul
    self._register_load_state_dict_pre_hook(self.check_initialize)

  def check_initialize(self, state_dict:dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    """Pre-hook for loading state dict that converts UE weights to standard embeddings.
    
    This method is called before loading the state dict. If it detects Unified
    Embedding weights (by checking for 'integration1.weight'), it creates a
    temporary UE instance, loads the weights, computes all embeddings, and
    replaces the state dict with standard embedding weights.
    
    Args:
        state_dict (dict): The state dictionary being loaded.
        prefix (str): The prefix for this module in the state dict.
        local_metadata: Metadata for this module.
        strict (bool): Whether to strictly enforce matching keys.
        missing_keys (list): List to append missing keys to.
        unexpected_keys (list): List to append unexpected keys to.
        error_msgs (list): List to append error messages to.
    """
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
  """Advanced embedding layer with sacrificial training parameters.
  
  Unified Embeddings (UEs) address training instability caused by infrequent token
  updates by using a larger intermediate embedding space during training. The key
  insight is that rare tokens can disrupt training when their embeddings are
  suddenly updated after long periods of inactivity.
  
  Architecture:
  1. Large embedding table (vocab_size × embed_dim × front_embed_mul)
  2. Integration layers that project down to the target embedding dimension
  3. Optional activation functions and layer normalization
  4. Efficient reordering for common tokens to minimize computation
  
  During inference, the entire network can be collapsed to a standard embedding
  table by pre-computing all token embeddings, eliminating any overhead.
  
  Benefits:
  - More stable training, especially for rare tokens
  - Better gradient flow to embedding parameters
  - Can be converted to standard embeddings for deployment
  - Supports CPU offloading for memory-constrained training
  """
  
  def __init__(self,
               vocab_size: int,
               embed_dim: int,
               front_embed_mul: float,
               keep_embed_on_cpu=False,
               emb_training_epochs=50,
               ln=False,
               emb_activation=False,
               activation=F.gelu,
               integration1_5=False,
               integration2=True,
               linear=nn.Linear):
    """Initialize Unified Embedding layer.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Output embedding dimension that the model sees.
        front_embed_mul (float): Multiplier for the internal embedding dimension.
            Larger values (e.g., 16) work better but use more memory. The internal
            embedding dimension is embed_dim * front_embed_mul.
        keep_embed_on_cpu (bool): If True, keep embeddings and first integration
            layer on CPU to save GPU memory. This has a performance cost but enables
            training with larger front_embed_mul values. The tradeoff becomes
            worthwhile when sequence_length * batch_size > 2000. Defaults to False.
        emb_training_epochs (int): Number of epochs for training the UE when
            initializing from existing embeddings. Only used when converting a
            pre-trained model to use UEs. Defaults to 50.
        ln (bool): If True, apply layer normalization to output embeddings.
            Defaults to False.
        emb_activation (bool): If True, apply activation function to the initial
            embedding lookup. Defaults to False.
        activation (callable): Activation function to use in integration layers.
            Defaults to F.gelu.
        integration1_5 (bool): If True, add an additional integration layer
            between integration1 and integration2. Defaults to False.
        integration2 (bool): If True, use a second integration layer. If False,
            integration1 projects directly to embed_dim. Defaults to True.
        linear (type): Linear layer class to use for integration layers.
            Defaults to nn.Linear.
    
    Note:
        The large front embeddings help because:
        1. Rare tokens have more parameters to store information
        2. The integration layers can learn to ignore outdated information
        3. Common tokens benefit from the reordering optimization
        4. The network is less sensitive to sudden embedding updates
    """
    super().__init__()
    if isinstance(integration2, int):
      mid_size = integration2
    else:
      mid_size = embed_dim
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
    self.integration1 = create_linear(linear, "ue_integration1", front_embed_dim, mid_size)
    if integration1_5:
      self.integration1_5 = create_linear(linear, "ue_integration15", mid_size, mid_size)
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


    if integration2 != False:
      self.integration2 = create_linear(linear, "ue_integration2", mid_size, embed_dim)
    else:
      self.register_parameter('integration2', None)
    if ln:
      self.ln = nn.LayerNorm(embed_dim)
    else:
      self.ln = lambda x:x
    if emb_activation == True:
      self.emb_activation = activation
    else:
      self.emb_activation = lambda x:x
    self.ff_activation = activation
    self._register_load_state_dict_pre_hook(self.check_initialize)
    self.initialized_from_small_embed = False

  def initialize(self, embedding_w: torch.Tensor):
    """Initialize UE from existing embedding weights.
    
    This method trains the UE network to reproduce given embedding weights,
    allowing UEs to be added to pre-trained models. It uses a small neural
    network training loop to learn the integration weights.
    
    Args:
        embedding_w (torch.Tensor): Existing embedding weights of shape
            (vocab_size, embedding_dim) to reproduce.
    
    Note:
        The training uses AdaGrad optimizer with exponential learning rate
        decay. The initial embeddings are copied to the front part of the
        large embedding table to speed up convergence.
    """
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
    """Pre-hook to initialize from standard embedding weights if present.
    
    Args:
        state_dict (dict): State dictionary being loaded.
        prefix (str): Module prefix in state dict.
        local_metadata: Module metadata.
        strict (bool): Whether to strictly match keys.
        missing_keys (list): List of missing keys.
        unexpected_keys (list): List of unexpected keys.
        error_msgs (list): List of error messages.
    """
    if f'{prefix}weight' in state_dict:
      self.initialize(state_dict[f'{prefix}weight'])

  def forward(self, idxs: torch.Tensor = None, start_slice = None, end_slice = None, allow_reorder=True) -> torch.Tensor:
    """Compute embeddings for given token indices.
    
    This method supports several optimizations:
    1. Reordering tokens to compute common tokens only once
    2. CPU-GPU transfer minimization when embeddings are on CPU
    3. Batch computation of all embeddings when idxs is None
    
    Args:
        idxs (torch.Tensor, optional): Token indices to embed. Shape can be
            (batch_size, sequence_length) or (sequence_length,). If None,
            returns all embeddings from start_slice to end_slice.
        start_slice (int, optional): Start index when computing all embeddings.
            Only used when idxs is None. Defaults to 0.
        end_slice (int, optional): End index when computing all embeddings.
            Only used when idxs is None. Defaults to vocab_size.
        allow_reorder (bool): If True and sequence length > 1, reorder tokens
            to compute each unique token only once. This optimization is
            especially effective for common tokens. Defaults to True.
    
    Returns:
        torch.Tensor: Embedded representations. Shape is the same as input
            idxs but with an additional embedding dimension. If idxs was None,
            returns embeddings of shape (num_tokens, embed_dim).
    
    Note:
        The reordering optimization finds unique tokens, computes their
        embeddings once, then scatters results back to original positions.
        This significantly reduces computation for repeated tokens.
    """
    #If they send in 'none' then we will send back all embeddings.
    tok_embed_device = self.tok_embed.weight.device
    if not idxs is None:
      output_device = idxs.device
    else:
      output_device = tok_embed_device
    if len(idxs.shape) == 1:
      need_squeeze = True
      idxs = idxs.unsqueeze(0)
    else:
      need_squeeze = False
    if not idxs is None and allow_reorder and idxs.size(1) > 1:
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
        x = self.integration1_5(self.ff_activation(x))

      #x = self.integration1(x)
      # Minimize the lookup/liner layer costs
      if not self.integration2 is None:
        x = self.integration2(self.ff_activation(x))
      # Now we can re-lookup the result
      reorg_idxs = x[torch.tensor(tuple(idx_locations[idx] for idx in local_idxs), dtype=torch.long, device=idxs.device)]
      x = reorg_idxs.view(batch, sequence, -1)
    else:
      if idxs is None:
        if end_slice is None:
          end_slice = self.vocab_size
        if start_slice is None:
          start_slice = 0
      if tok_embed_device != output_device:
        #This probably means the embeddings are on CPU to save memory.
        #Autocast doesn't like mixed devices
        with torch.amp.autocast(enabled=False, device_type=tok_embed_device.type):
          if idxs is None:
            #They want them all!
            x = self.emb_activation(self.tok_embed.weight[start_slice:end_slice])
          else:
            x = self.emb_activation(self.tok_embed(idxs))

          if not self.integration1 is None:
            x = self.integration1(x)
          x = x.to(output_device)
      else:
        if idxs is None:
          #They want them all!
          x = self.emb_activation(self.tok_embed.weight[start_slice:end_slice])
        else:
          x = self.emb_activation(self.tok_embed(idxs))

        if not self.integration1 is None:
          x = self.integration1(x)
      if not self.integration1_5 is None:
        x = self.integration1_5(self.ff_activation(x))
      #x = self.integration1(x)
      if not self.integration2 is None:
        x = self.integration2(self.ff_activation(x))
    if need_squeeze:
      x = x.squeeze(0)
    return self.ln(x)
