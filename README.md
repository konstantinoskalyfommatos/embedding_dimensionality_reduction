# Goal
This project aims to create a framework for dimensionality reduction of foundation embedding models.

# Idea
**Given an embedding model that maps input text into vectors in $R^n$, learn a function that maps the output vectors in a lower dimensional space such that, given a batch of high-dimensional vectors, the low-dimensional vectors preserve the pairwise cosine similarities and euclidean distances. Then, evaluate if this distance-cosine similarity preservation translates into downstream tasks performance preservation.**

More formally, learn a function $$f : \mathbb{R}^n \to \mathbb{R}^k,$$ with $k << n$, such that given a set $X$ of points in $R^n$, $\forall x, y \in X$ $$\|x - y\| \approx \|f(x) - f(y)\|$$ and $$\cos{\theta_1} \approx \cos{\theta_2},$$ where $\theta_1, \theta_2$ are the angles between $x, y$ and $f(x), f(y)$ accordingly.

# Notes:
- If $f$ also preserves norms (i.e. $\|x\| = \|f(x)\| \forall x \in X$), then preserving the cosine of two vectors is equivalent to preserving their dot product.
- If $f(0) = 0 \in R^k$, then $f$ preserves the norms.

# Existing functions
- The Johnson-Linderastrauss lemma states that a random matrix has the above properties (see [Random projections and applications to dimensionality reduction](https://cseweb.ucsd.edu/~akmenon/HonoursThesis.pdf) section 5.1.2 for dot product preservation).

# Implementation
## Pipeline
- Calculate a large number of embeddings -- output vectors (in $R^n$).
- Define a simple Neural Network that maps these output vectors into a low dimensinal space.
- Train the Network to preserve pairwise distances and cosine similarities for each batch. The loss function is

$$\mathcal{L} = \frac{1}{N}\sum_{i<j} w_{ij} \left( d_{\text{low}}(x_i, x_j) - d_{\text{high}}(x_i, x_j) \right)^2$$
    
where

$$w_{ij} = \frac{1}{\left(d_{\text{high}}(x_i, x_j) + \epsilon\right)^m}$$

and N is the number of unique pairs in a batch.


# Prerequisites
To set up the project, create a uv environment at the project root level using the following commands:
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Then, create a `.env` file in the project root directory according to the `.env.example` file.

## Environment Configuration

To ensure the project modules are accessible, set the `PYTHONPATH` to include the `src` directory:

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

For a persistent setup, add this to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
echo 'export PYTHONPATH="$(pwd)/src:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```