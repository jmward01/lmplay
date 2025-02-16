# LMPlay
tldr - Check out the [Unified Embeddings experiments](https://github.com/jmward01/lmplay/wiki/Unified-Embeddings), [Unified Weights experiments](https://github.com/jmward01/lmplay/wiki/Unified-Weights) and [Sacrificial Training](https://github.com/jmward01/lmplay/wiki/Sacrificial-Training) stuff on the wiki. Also look at [Other Explanations](https://github.com/jmward01/lmplay/wiki/Other-Explanations) for what I am testing to show results aren't a fluke. These techniques allow a 6 layer model to train faster, and deeper.

This repo contains two things:
- The 'LMPlay' training/plotting/dataloading/runner harness - This is designed to be relatively simple with the goal being to make it easy to try new experiments and compare the results against a baseline.  
- The experiments! These will change over time. There are some details below and more detials in the wiki. [UEs](https://github.com/jmward01/lmplay/wiki/Unified-Embeddings) appear to be an exceptionally successful experiment so give them a look. I have others like Unified Weights and value norm that also show solid gains. Still others are in a private repo and may show up here later.

The LMPlay framework is intended to be a playground to make it easy to try crazy things. It is rough, it is unfinished and, like the best things in life, it is messy.
I have tried to create a structure in this library that makes it easy to try things and compare results.
Everything here is targeted at quick testing and turnaround on commodity hardware. The two primary targets are mac M2 and a NVIDIA 3060 with 12GB ram.
The structure of the project is designed for copy/paste/change, not to provide a stable prod library. Most of the training code has been extracted but the model has been left wide open.
That being said, the training has been designed to be simple. No multi-gpu, multi-process, etc, so hopefully it is a very accessible project for other to try things in.

A note of caution here, because this is a playground the code here will likely change. That means that while you can import this in and use the classes, there is basically no guarantee that updates to this codebase won't badly break your code. Backward compatability is a very low priority compared to creating and comparing experiments. That being said, as results are posted I am trying to make `lmplay.modules` a one stop location to import useful, partially stable, classes. 

## The experiments
I have been playing with ideas 'on the side' for a while now and several show promise. The [wiki](https://github.com/jmward01/lmplay/wiki) is the best place to see the latest crazy ideas I am playing with but here are a few: 
### [Sacrificial Training](https://github.com/jmward01/lmplay/wiki/Sacrificial-Training)
These experiments fall into something I call 'Sacrificial Training'. These experiments add additional parameters during training time that can be removed for prod so the weights require no changes to code and have the same parameter counts as regularly trained weights. I am just starting to scratch the surface of these techniques but the results are very promising.
#### [Unified Embeddings](https://github.com/jmward01/lmplay/wiki/Unified-Embeddings) 
If you look at only one thing in this repo UEs should be it. They enable a 6 layer model to beat a 12 layer model by a wide margin. The method is a training only technique meaning it can be used on existing LLM architectures without requiring changes to inference code (just training code).
#### [Unified Weights](https://github.com/jmward01/lmplay/wiki/Unified-Weights)
This continues the idea of predicting weights to enable better training with some respectable performance improvements.
#### [NNMemory](https://github.com/jmward01/lmplay/wiki/NNMemory)
This tries to create a layer dedicated to knowledge storage.
#### [Attention Position](https://github.com/jmward01/lmplay/wiki/Attention-Position)
This is a twist on standard MHA that encodes reverse position information into the attn ranking. 

### Other experiments
####  [Value Norm](https://github.com/jmward01/lmplay/wiki/Value-Norm) 
This gives a respectible boost to the 'standard' transformer block by applying a simple layer norm to the value projection.
#### [Combined Sacrificial](https://github.com/jmward01/lmplay/wiki/Combined-Sacrificial) 
This series combines the sucessful sacraficial techniques into one model to see how well they stack.

## Experiment details
Unless otherwise stated experiments are run using the Adagrad optimizer with a batch size of 50, fixed lr of 0.0006 and no weight decay. Training is all done single GPU with --amp enabled. Mini-batch sizes are generally either 4, 10 or 25 depending on the GPU available and the test being run but quick tests show no significant deviation in results with different mini-batch sizes. --amp does have a minor, but noticable, impact but the performance gains make it a requirement and the impacts are much smaller than the gap between experiment results and baseline results.    
### Training data
The code currently points to the official wikipedia datasets on huggingface (wikimedia/wikipedia 20231101.XX). More may be added later (I want to add more robust datasets to make a more functional tiny model). Many of the graphs show results from models trained on other versions of the wiki corpus. I will likely try to re-train al the experiments on the official wiki datasets but that is weeks of GPU time on my tiny 3060. Any graph that shows results between different experiments had all models trained on the same data with the same seed.  

## GPT2ish (lmp_trainer --exp gpt2ish)
The baseline encoder I am using as the reference model is based loosely on GPT2. Any flaws in it are my own and you shouldn't take this as reference code for a GPT2 implementation.

## Baseline transformer code
The basic transformer encoder block is a very simple implementation that makes testing new ideas easy, but shouldn't be considered prod ready or appropriate code. It is however simple enough to make it easy to see what goes on inside of a basic transformer block. The actual code implementing it is a mixture of many dufferent sources around the internet along with a lot of my own code. I tried to 're-interpret/re-write' the code examples I saw elsewhere but there are only so many ways to write a simple transformer block. I have, unfortunately, forgotten the many references I used to write this baseline so if there is a line of code in there that looks like it came from some other repo please point it out to me and I will credit it if I can link it back to that repo. 

## Model weights / repeatability
I currently don't have plans to release trained model weights. The models here aren't well trained enough to really matter and the weights can all be re-created by running the experiments for a few days on a commodity GPU. If someone -really- wants the weights then please contact me and I will consider releasing the weights of a given experiment.


## Code/idea attribution
If you end up using any of the experiment ideas in this repo please attribute me and this repo. I would like to work on a more academic writeup of the ideas in here, but for now I am just getting things built and playing. Also, I would appreciate you contacting me and letting me know if my ideas made it into something. This isn't required, but it would be nice to see these ideas get implemented, especially in some of the big LLMs.   

## Project structure
### lmp_trainer
The training loop is implemented in `lmplay.train.__main__`. At the top are two imported `ModelRunner` classes. One is for the baseline transformer encoder model that is 'GPT2ish' in design. 
The other points to one of the experiments. Experiments are generaly cut/paste/modify from the base encoder model and this is by design. 
This style leads to a lot of copied code, but it also leads to readability, simple testing and avoids issues where code changes in an experiment impact other experiments. I based this repo off of an internal one I use and I have found that this structure has allowed for rapid testing of new ideas.

### lmp_generator
This is a very basic inference engine with the absolute minimum code needed to take a trained model ang generate results from it. Performance is a distant thought in this code.
The code is implemented in `lmplay.generate.__main__`

### lmp_plotstats
It is all about the graphs right? This will take outputs from different runs and put them all in the same plot files. Two basic stats are plotted, accuracy and loss. 
For each of these there are two different kinds of graphs, a regular log plot and a 'diff' plot where the longest run is used as the baseline for the other runs. This diff view makes it much easier to compare the performance of different models/runs.
The code is implemented in `lmplay.stats.__main__`. It is messy and does some bad math to make things look prettier so this is likely to change a bit in the future.

## Usage
### Run some models on the default training plan
The `default` training plan will download and prepare the official wikipedia en and es datasets to train against. These are large enough to show real difference between model changes but not so large that it takes a month to train on a reasonable GPU.
```
#All reported experiments had --amp turned on for speed. If you aren't on NVIDA leave it off. Results will be nearly the same.
lmp_trainer --help

#Show all the experiments that can be run
lmp-trainer --exp list

#Show a quick description of an expirement and then exit
lmp-trainer --exp XXX --describe

#To train the baseline 6L GPT2ish model
lmp_trainer --amp --device cuda

#To train one of the experiments
lmp_trainer --amp --device cuda --exp <exp name>

#To list the experiments
lmp_trainer --amp --device cuda --exp list

#To create plots (assuming all results are in the default gpt_out directory): 
lmp_plotstats
```
That's it! it will download datasets and start training. Check `--help` for other options like device, mini-batch-size, etc etc.
### Other training plans
The training plan stuff is designed to allow training steps so you can progressively train against larger or more targeted data in a series of well defined steps. See --help for the list of available training plans or look in the code. You can even design your own and use a json file to define everything. The `full_v1` and `pretrain` plans take a -lot- of drive space since they have almost 0.3T tokens in them. These will take a long time to train even on a good GPU so only use these if you are really interested. I am using these larger datasets to see if I can get the 6 layer model to saturation and to see how actual large training does on the experiments.



## Contribution
At the moment I am doing this mainly for myself but I am open to considering outside contributions, especially ones with GPU resources!
