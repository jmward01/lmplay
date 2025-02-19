from lmplay.base.base_model import BasicModelRunner
from .model import GPT2
from lmplay.base.runner_list import expose_runner


@expose_runner('nnm4_0', "Trying 1.0 (sort of) with the cheaper NNMem")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(),
                          **kwargs)

@expose_runner('nnm4_1', "nn_ln set to False to be closer to the old nnm1.0")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(nn_ln=False),
                          **kwargs)

@expose_runner('nnm4_2', "512 vs 256 cells")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(cells=512),
                          **kwargs)

@expose_runner('nnm4_3', "Do we actually need a softmax?")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(softmax=False),
                          **kwargs)

@expose_runner('nnm4_4', "Trying a block pattern of BN-BN-BNN-BNN-BNNN-BNNN")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(block_pattern="BN-BN-BNN-BNN-BNNN-BNNN"),
                          **kwargs)

@expose_runner('nnm4_5', "Trying a block pattern of BNN-BNN-BNN-BNN-BNN-B")
def rc(*args, **kwargs):
  return BasicModelRunner(GPT2,
                          *args,
                          overrides=dict(block_pattern="BNN-BNN-BNN-BNN-BNN-B"),
                          **kwargs)
