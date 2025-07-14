# Goal
This project aims to create a framework for distilling embedding spaces of foundation encoder models.

# Idea
The idea is the following. Given an encoder:
1) Define the teacher by using the encoder and adding a function that maps a finite set of vectors from the (big) vector space to a lower dimensional vector space such that it preserves their norms, and their pairwise euclidean distances and cosine similarities.
2) Define the student by using the encoder and adding a trainable projection layer that maps vectors from the (big) vector space to the lower dimensional space.
3) Train the student by defining a loss function that aims to, for every batch of vectors, bring the student vectors close to the teacher vectors (in the low dimensional space).

# Current process
Currently, the function used to distill the dimensionality of the vectors is an isometric function. This preserves the pairwise euclidean distances and cosine similarities in the lower space *exactly*. Please refer to this [Google Colab link](https://drive.google.com/file/d/16mqMwqXe_JDxvxHDw2qrfQY7JHsU7Rrx/view?usp=sharing) for a brief demo.

In the future we will investigate a random projection, under the Johnson Linderstrauss Lemma.

# Prerequisites
To set up the project, create a uv environment at the project root level using the following commands:
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Then, create a `.env` file in the project root directory according to the `.env.example` file.