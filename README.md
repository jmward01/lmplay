# lmplay
This is intended to be a playground to make it easy to try crazy things. It is rough, it is unfinished and, like the best things in life, it is messy.
I have tried to create a structure in this library that makes it easy to try things and compare results.
Everything here is targeted at quick testing and turnaround on commodity hardware. The two primary targets are mac M2 and a NVIDIA 3070 with 12GB ram.
The structure of the project is designed for copy/paste/change, not to provide a stable prod library. Most of the training code has been extracted but the model has been left wide open.
That being said, the training has been designed to be simple. No multi-gpu, multi-process, etc, so hopefully it is a very accessible project for other to try things in.

A note of caution here, because this is a playground it really isn't designed to be usable as a library. That means that while you can import this in and use the classes, there is basically no guarantee that updates to this codebase won't badly break your code. Backward compatability is a very low priority compared to creating and comparing experiments. 

## GPT2ish
The baseline encoder I am using as the reference model is based on GPT2. Any flaws in it are my own and you shouldn't take this as reference code for a GPT2 implementation.

## Baseline transformer code
The basic transformer encoder block is a very simple implementation that makes testing new ideas easy, but shouldn't be considered prod ready or appropriate code. It is however simple enough to make it easy to see what goes on inside of a basic transformer block. 

## Model weights
I currently don't have plans to release trained model weights. The models here aren't well trained enough to really matter and the weights can all be re-created by running the experiments for a few days on a commodity GPU. If someone -really- wants the weights then please contact me and I will consider releasing the weights of a given experiment.

## Code/idea attribution
If you end up using any of the experiment ideas in this repo please attribute me and this repo. I would like to work on a more academic writeup of the ideas in here, but for now I am just getting things built and playing.   

## Project structure
### lmp_trainer
The training loop is implemented in `lmplay.train.__main__`. At the top are two imoprted `ModelRunner` classes. One is for the baseline transformer encoder model that is 'GPT2ish' in design. 
The other points to one of the experiments. Experiments are generaly cut/paste/modify from the base encoder model. 
This does lead to a lot of copied code, but it also leads to readability and simple testing. I based this repo off of an internal one I use and I have found that this structure has allowed for rapid testing of new ideas.

### lmp_generator
This is a very basic inference engine with the absolute minimum code needed to take a trained model ang generate results from it. Performance is a distant thought in this code.
The code is implemented in `lmplay.generate.__main__`

### lmp_plotstats
It is all about the graphs right? This will take outputs from different runs and put them all in the same plot files. Two basic stats are plotted, accuracy and loss.
For each of these there are two different kinds of graphs, a regular log plot and a 'norm' plot where the longest run is used as the baseline for the other runs. This norm view makes it much easier to compare the performance of different models/runs.
The code is implemented in `lmplay.stats.__main__`

## The experiments
I have been playing with ideas 'on the side' for a while now and several show promise. The first one is something I call 'Unified Embeddings'.
### Unified Embeddings
Unified Embeddings provide a better way to train embeddings with no additional inference costs or model structure changes however there is a training memory penalty.
The basic idea is to take a very large embedding and run it through a ff layer to generate the output embedding during training. 
For production inference all vocab embeddings can be generated and stored and the embedding training weights can be tossed.
In the testing I have done so for it look quite promising:

![](results/ue_log_loss.png)
This graph shows a 6 layer UE model beating a 12 layer model, at least in initial training. The long term benefits are still unknown but these results are promising. The 6l UE model here has exactly the same prod inference costs/structure/weights/etc as the normal 6l model. Only during training does it have extra logic/parameters needed for training.
The norm plot shows it better:
![](results/ue_log_norm_loss.png)
Here you can clearly see how much better the UE model is over the 6 layer model. It is still gaining ground on it in fact. The 12 layer is also losing to the 6 layer UE model, but that gap is slowly shrinking. Clearly, longer runs are needed. I will update this plot as the runs continue.

## Future experiments
I am slowly 'cleaning up' many projects that I have been working on and intend to release them as I have longer training runs on them. I am currently limited to my one 3070 so even these limited runs take several days each. In fact, one epoch on the dataset in use will take roughly 10 days per model to complete. 

I picked UEs first since they are very easy to understand and implement. I will likely iterate on them a little more before releasing the next experiment. I hope to release a version of UEs that can be a drop-in replacement to an existing well trained model to allow 'bolting' on this enhanced training. I have done some internal experiments in this direction already but nothing ready to be released.

## Contribution
At the moment I am doing this mainly for myself but I am open to considering outside contributions.