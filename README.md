
## llama2.ml

Llama2 inference in one file of pure OCaml.

Based on [Andrej Karpathy](https://karpathy.ai/)'s [llama2.c](https://github.com/karpathy/llama2.c).


## Run

First, navigate to the folder when you keep your projects and clone this repository.


```bash
git clone https://github.com/jackpeck/llama2.ml.git
```

Then, open the repository folder:

```bash
cd llama2.ml
```

Now, let's run a baby Llama 2 model in OCaml. You need a model checkpoint. Download this 15M parameter model trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (~60MB download):

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Run the OCaml code:

```bash
ocaml llama2.ml stories15M.bin
```
> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.

You can also prompt the model with a prefix, and set the number of steps and temperature.

```bash
Usage: ocaml llama2.ml <checkpoint_file> [temperature] [steps] [prompt]
```

Eg to sample at temperature 0.8 for 256 steps and with a prompt:
```bash
ocaml llama2.ml stories15M.bin 0.8 256 "On the beach"
```


We can also try a bit bigger 42M parameter model, to generate more coherent and diverse stories:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
ocaml llama2.ml stories42M.bin
```
> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, yellow flower in the garden. It was a sunflower! Lily thought it was the most beautiful flower she had ever seen.
Lily's mom came outside and saw the sunflower too. "Wow, that's a big flower!" she said. "Let's pick it and put it in a vase." Lily was so excited to have the sunflower in her room.


```bash
ocaml llama2.ml stories42M.bin 0.8 256 "In the wild"
```

> In the wild jungle, there was a zebra. He loved to hide in the bushes. One day, he was running around looking for a good place to hide when he noticed a small hut. He ran over to it and peeked inside. Inside he could see a family of foxes playing around. The zebra stayed still and watched them.
The foxes noticed the zebra and ran away. The zebra was sad because he had no one to play with. He then noticed a tree full of leaves and ran over to it to hide behind its branches.