# Advanced Attention & LLM Exercises (Hard & Tricky)

**Purpose**: Deepen conceptual understanding of attention mechanisms, training dynamics, and model design choices. These questions require 2-5 minutes of thinking. No answers provided—work through your reasoning on paper.

---

## Part 1: Advanced Tensor Algebra & Memory

**H1.1**: You're running attention on a batch where `(batch, seq_len, d_model) = (32, 256, 256)`. When you compute `Q @ K^T`, you get shape `(batch, seq_len, seq_len) = (32, 256, 256)`. How many **floating-point numbers** are in this single attention score matrix for the entire batch? If you use `float32`, how many GB of GPU RAM does this occupy? Is this a bottleneck on your hardware?

**H1.2**: In multi-head attention with `n_heads=4`, after splitting `d_model=256`, each head gets `d_head=64`. When you compute scaled dot-product attention per head:
- Shape before attention: `Q_head = (batch=32, n_heads=4, seq_len=256, d_head=64)`
- After `Q @ K^T`: `(32, 4, 256, 256)`
- After softmax: still `(32, 4, 256, 256)`
- After `@ V`: back to `(32, 4, 256, 64)`

Why don't you track this as a single `(batch, seq_len, seq_len, n_heads)` tensor instead? (Hint: think about which dimension changes size with different sequence lengths.)

**H1.3**: Suppose you change `d_model=256` to `d_model=512` but keep `n_heads=4`, so `d_head=128`. How does each of these scale?
- Number of parameters in Q, K, V projections
- Size of `Q @ K^T` attention score matrix
- Computational cost of softmax

Which scaling is fastest, and why?

**H1.4**: You're training with `batch_size=32` and `seq_len=256`. On epoch 2, you run out of GPU memory. A colleague suggests "just reduce seq_len to 128." Quantitatively, how much memory do you save? Is it a linear reduction or quadratic? Explain why.

**H1.5**: In your current code, `position_ids = torch.arange(seq_len, device=device)` is created inside the forward pass. If you process 1000 samples, how many times is this tensor allocated? Is this wasteful? Propose a fix.

---

## Part 2: Numerical Stability & Softmax

**H2.1**: When computing `softmax(Q @ K^T / sqrt(d_head))` with `d_head=64`, you scale by `1/sqrt(64) ≈ 0.125`. If logits range from `-10` to `+10` before scaling, what's the range after scaling? Why does this matter for numerical stability in softmax?

**H2.2**: Suppose you have two sequences:
- Seq A: 10-token context (sparse attention, most logits near 0)
- Seq B: 256-token context (very varied logits from -50 to +50)

If you compute softmax on both, will one have more "peaked" attention weights than the other? Why? What happens to gradient flow in each case?

**H2.3**: In the causal mask, masked positions get score `-inf` before softmax, ensuring `exp(-inf) = 0`. But in practice, we often use `-1e9` instead of `-inf`. 
- Why might using true `-inf` cause issues?
- Why is `-1e9` safer?
- What float precision matters here?

**H2.4**: During backpropagation through softmax, gradients flow differently depending on attention weight magnitudes. If one position dominates (attention weight ≈ 0.99) and others are sparse (0.001 each), how does gradient magnitude differ for each? Could this hurt training?

**H2.5**: You notice that loss stops decreasing after epoch 3. Your colleague suggests "the attention weights are collapsing"—most probability on a single position. How would you **inspect and verify** this? Write pseudocode to compute entropy of attention weights per head per batch element.

---

## Part 3: Multi-Head Attention Semantics

**H3.1**: You have 4 attention heads. Each head independently computes softmax over context positions. Can two heads learn **conflicting** attention patterns? (Example: Head 1 attends to position 5, Head 2 attends to position 3.) If yes, is this a problem? Why or why not?

**H3.2**: In multi-head attention, after computing each head's output, you concatenate them: `(batch, seq_len, 256)` → `head_outputs = (batch, seq_len, 64 * 4)`. Then apply an output projection `W_o`. 
- Is this projection matrix trainable? 
- What's its shape and number of parameters?
- Why not skip the projection and just concatenate?

**H3.3**: Suppose one attention head learns to reliably track "the most recent pronoun" across sequences, and another learns "the most recent noun." Is there any mechanism in multi-head attention that prevents them from learning duplicated or conflicting patterns? How does the training loss prevent such redundancy?

**H3.4**: You have `n_heads=4` and `d_model=256`, so `d_head=64`. What if someone suggests `n_heads=8` instead (keeping `d_model=256`)? 
- How do parameters and computation change?
- Would 8 heads generalizing better than 4?
- Could 8 heads be harmful? Why?

**H3.5**: In your `CausalSelfAttention.forward()`, you reshape Q, K, V from `(batch, seq_len, d_model)` to `(batch, seq_len, n_heads, d_head)`, then compute attention per head. 
- What if you didn't reshape and instead computed attention on the full `d_model` dimension?
- Would the model still work?
- What would be gained/lost?

---

## Part 4: Causal Masking Deep Dive

**H4.1**: In causal masking, position `t` cannot attend to positions `t+1, t+2, ..., T`. You implement this as:
```python
mask = torch.tril(torch.ones(seq_len, seq_len)) == 0
scores[mask] = -inf
```

If `seq_len=256`, this creates a `(256, 256)` boolean mask. How many `True` values (masked positions) are there? What fraction of the attention matrix is masked?

**H4.2**: During training, all positions are processed in parallel. During inference, you generate one token at a time. How does the causal mask behave in each case? 
- During training at position t=50: which positions can attend?
- During inference when generating token 51: which positions can attend?
- Why is this difference important?

**H4.3**: Suppose you remove the causal mask entirely (allowing position t to attend to all future positions). What happens to:
- Training loss trajectory
- Generalization to unseen data
- The learned embeddings

Why is causality crucial for language modeling?

**H4.4**: In your GPT model, tokens are processed left-to-right: `[CLS] The quick brown fox ...`. At position t=3 ("brown"), causal masking prevents attending to positions 4+ ("fox", etc.). But what if you had a **bidirectional architecture** like BERT that attends both ways? How would the computation change? What would you gain/lose for language modeling?

**H4.5**: You're debugging a bug where the model generates repetitive text. You inspect attention weights and notice: "At every position, the model attends ~100% to the previous token." This seems like the causal mask is too restrictive. Should you loosen it? Why or why not? What's the real issue?

---

## Part 5: Training Dynamics & Loss Interpretation

**H5.1**: On epoch 1, loss decreases from 12.2 → 11.5 (0.7 drop). On epoch 4, loss decreases from 4.5 → 4.4 (0.1 drop). Why does the loss decrease more slowly as training progresses? Is this good or bad?

**H5.2**: You train for 5 epochs, and loss reaches 4.2. Your colleague trains the same model for 10 epochs and gets 3.9. Is this progress guaranteed? What could go wrong if you train "forever"? (Hint: think about overfitting, especially with limited data.)

**H5.3**: Cross-entropy loss is `loss = -log(p_correct)` where `p_correct` is model's predicted probability for the correct next token.
- If `p_correct=0.5`: `loss = -log(0.5) ≈ 0.693`
- If `p_correct=0.1`: `loss = -log(0.1) ≈ 2.303`
- If `p_correct=0.01`: `loss = -log(0.01) ≈ 4.605`

Your model starts with `loss ≈ 12.2`. What does this imply about `p_correct` initially? Is the model worse than random?

**H5.4**: During training, you compute loss on the **training set**. If you had a validation set, which would have higher loss? Why? How would this gap inform your training decisions?

**H5.5**: You notice that during backpropagation:
- Gradients for embedding parameters are huge (e.g., 0.5)
- Gradients for attention parameters are tiny (e.g., 1e-5)

This suggests one part of the model is learning fast, the other slowly. Is this a problem? How would you fix it? (Hint: learning rate, initialization, weight decay.)

---

## Part 6: Embeddings & Positional Encoding

**H6.1**: Your token embedding table has shape `(vocab_size=200k, d_model=256)` ≈ 51.2M parameters. Positional embedding table has shape `(max_seq_len=256, d_model=256)` ≈ 65.5k parameters. Why does token embedding dominate? Is this asymmetry problematic?

**H6.2**: You add token and positional embeddings: `x = token_emb + pos_emb`. Both have shape `(batch, seq_len, d_model)`. This addition happens element-wise. 
- Do these embeddings interact, or are they independent signals?
- If you swap two words in a sentence, does the positional encoding change?
- What does the model learn about each component?

**H6.3**: Your position embedding uses `pos_emb = self.pos_embedding(position_ids)` where `position_ids = [0, 1, 2, ..., 255]`. This is absolute positional encoding.
- If you train on sequences up to length 256 but encounter a longer sequence at test time, what happens?
- How would relative positional encoding (RoPE or ALiBi) handle this differently?

**H6.4**: Suppose you initialize embeddings randomly with `nn.Embedding(..., from_pretrained=...)` using a pretrained embedding from a large model (e.g., OpenAI's GPT-2). 
- Would training converge faster?
- Would the model need fewer epochs?
- Why might transfer learning work here?

**H6.5**: At initialization, token embeddings are random (high entropy). After 1 epoch of training, similar tokens have similar embedding vectors (low entropy). Explain this phenomenon. What mechanism causes embeddings to cluster by semantic/syntactic similarity?

---

## Part 7: Gradient Flow & Backpropagation

**H7.1**: During backward pass through the attention softmax:
```
forward: scores → softmax(scores) → attention_weights
backward: d_loss/d_softmax ← gradient from future layers
```

If attention_weights are concentrated (one position has 0.99, others 0.01), how does gradient concentration affect weight updates in Q and K? Could this slow learning?

**H7.2**: In your model:
```
x → token_emb → + pos_emb → attention → output_head → output_logits → loss
```

Where does gradient "bottleneck" (smallest gradients)? Why? How could you use skip connections or residual networks to improve gradient flow?

**H7.3**: The causal mask applies `-inf` to future positions. During backward, these positions have `0` gradients. Does this mean the model never learns to "ignore" future positions explicitly? Why is causality enforced structurally (in the mask) rather than learned?

**H7.4**: You add L2 regularization: `loss = ce_loss + λ * ||weights||^2`. This adds `2λ*weight` to gradients during backprop. 
- How does this affect attention parameters vs. embedding parameters?
- Could it slow learning for embeddings (51.2M params) more than attention?
- Is this fair, or should you regularize differently per layer?

**H7.5**: Suppose you clip gradients: `clip_grad_norm_(model.parameters(), max_grad_norm=1.0)`. This prevents sudden gradient spikes. Does this help or hurt:
- Training speed?
- Final model quality?
- Numerical stability?

Why might gradient clipping be necessary in large models?

---

## Part 8: Architecture Design Choices

**H8.1**: Your current model is a **single-layer** GPT with one attention layer. What if you stacked **two** attention layers sequentially:
```
x → attention_1 → attention_2 → output
```

- How many parameters does this add?
- Does it help or hurt? (Think: capacity vs. overfitting on Shakespeare.)
- What's the tradeoff?

**H8.2**: In transformer blocks, there's usually a **feedforward network** after attention:
```
x → attention → LayerNorm → FFN(x) → LayerNorm → output
```

The FFN is typically `Linear(d_model, 4*d_model) → ReLU → Linear(4*d_model, d_model)`. 
- Why expand by 4x? Why the ReLU?
- Does this add nonlinearity (the model can fit more complex patterns)?
- Why not just use another attention layer instead?

**H8.3**: In your GPT model, you don't use **layer normalization**. Suppose you add `LayerNorm` before the attention layer:
```
x → LayerNorm → attention → output
```

- Does this change the model's ability to learn?
- Could it stabilize training (less gradient spiking)?
- Why do modern transformers use LayerNorm in nearly every layer?

**H8.4**: Your model uses **absolute positional encoding** (`position_ids = [0, 1, 2, ..., T]`). Alternatives include:
- Sinusoidal positional encoding (RoPE, ALiBi)
- Relative positions instead of absolute
- Learnable on any sequence length

For a model trained on sequences up to length 256, which encoding would generalize best to length 512 at test time? Why?

**H8.5**: Suppose you **remove attention** entirely and just use:
```
x → feedforward(x) → output
```

Could this model still learn Shakespeare? (Assume same total parameters.) What ability would it lose? Why is attention crucial for LLMs?

---

## Part 9: Inference & Generation Strategies

**H9.1**: During training, you process 256 tokens in parallel and compute loss. During generation (inference), you produce one token at a time. How does the causal mask change? What's the computational difference per-token?

**H9.2**: You've trained the model to predict `p(next_token | context)`. To generate text, one strategy is **greedy decoding**: always pick the token with highest probability. Another is **sampling**: randomly sample from the distribution. 
- Which produces repetitive text?
- Which produces more diverse text?
- Which reflects the model's learned distribution better?

**H9.3**: Suppose at each step, the model has 50% probability on "the" and 30% on "a", rest on others. 
- Greedy: always pick "the" → "the the the..."
- Sampling: sometimes "the", sometimes "a", sometimes other
- Beam search: explore multiple hypotheses per step

Why does beam search sometimes produce better (more natural) text than either greedy or sampling?

**H9.4**: You generate `max_tokens=100` starting from "The best way to learn". After 20 tokens, the model seems stuck in a loop. How would you **diagnose** this?
- Check attention patterns (is it stuck attending to one position?)
- Check token probabilities (is there a huge peak?)
- Check embeddings (are similar tokens far apart?)

What's the root cause, and how do you fix it?

**H9.5**: During generation, you accumulate a context of 256 tokens. If you want to generate 1000 tokens, what happens? (Hint: your `max_seq_len=256` in position embeddings). How would you handle longer generations?

---

## Part 10: Debugging, Optimization & Open-Ended Thinking

**H10.1**: You train for 5 epochs and get `test_loss=4.2`. You double the learning rate and train again: `test_loss=5.1` (worse!). You halve it: `test_loss=3.9` (better!). How would you find the **optimal** learning rate systematically?

**H10.2**: On Intel Arc GPU (your new hardware), you can process batch_size=32. Your colleague with RTX 4090 can do batch_size=256. Does batch size affect:
- Final model quality?
- Convergence speed (epochs to reach same loss)?
- Training time (wall-clock seconds)?

Why might larger batches hurt or help generalization?

**H10.3**: You notice `model.parameters()` shows 102.7M parameters, but GPU memory usage is 3GB (forward + backward + optimizer state). Roughly:
- Model weights: 102.7M * 4 bytes ≈ 410 MB
- Gradient buffer: 410 MB
- Optimizer state (Adam: momentum + variance): 2 * 410 MB ≈ 820 MB
- Activations (forward pass cache for backward): ???

Estimate activation memory. Why is it significant? How would you reduce it?

**H10.4**: You want to **distill** your trained model into a smaller model (e.g., 1 attention head instead of 4). How would you:
- Initialize the small model?
- Use the large model's knowledge during training?
- Know if distillation worked?

**H10.5**: Speculative thought: Your model achieves ~90% accuracy on predicting the next token in Shakespeare (estimated from loss). But if you count word-error-rate on generated sequences, accuracy seems lower. Why? (Hint: errors compound. If you mispredict the 5th token, the context for token 6 is wrong...)

**H10.6**: Modern LLMs use techniques like:
- Mixture of Experts (MoE): multiple sub-networks, routing between them
- Sparse Attention: not all positions attend to all others
- FlashAttention: GPU-optimized attention kernels

For your small model, are these worth the complexity? When would they become essential?

**H10.7 (Open-ended)**: Reflect on what makes large language models work:
- Is it the size (parameters)?
- The architecture (attention)?
- The data (scale)?
- The training procedure (loss function)?

Which factor matters most? Why? What could you study next to deepen your understanding?

---

## How to Use This File

1. **Pick a question** from a section above.
2. **Think for 2-5 minutes** before writing anything.
3. **Write your reasoning** in a separate file (e.g., `hard_solutions.md`), numbering each answer clearly.
4. **Ask for feedback** if unsure—I can validate your thinking.
5. **Don't look ahead**—solve in order to build intuition progressively.

---

## Tips for Success

- **Draw diagrams** for tensor shapes.
- **Work through small examples** (e.g., batch=1, seq_len=4) on paper.
- **Connect to your code**—look at actual `attention.py` and `gpt.py` while thinking.
- **Ask "why"**—not just "what happens," but "why is the design this way?"
- **Be okay with wrong answers**—the point is reasoning, not correctness.

**Good luck!**
