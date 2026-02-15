# Attention & GPT — Pen-and-Paper Exercises

Work through these by hand on paper. Use a calculator for the arithmetic.
Answers are at the bottom — try hard before checking.

---

## Part 1: Shapes & Dimensions

Your config: `d_model=8, n_heads=2, seq_len=4, batch_size=1, vocab_size=50`

**Q1.** What is `head_dim`? Show the calculation.

**Q2.** Input `x` has shape `(1, 4, 8)`. After `self.query(x)`, what is the shape? Why?

**Q3.** After `.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)`, what is the shape of Q?
Write out what each dimension represents.

**Q4.** What is the shape of `scores = Q @ K.transpose(-2, -1)`? Why does this make sense — what does each entry in this matrix represent?

**Q5.** What is the shape of `logits = self.output_head(x)` at the end of GPTModel? Why is each dimension that size?

---

## Part 2: Scaled Dot-Product Attention (by hand)

Use these tiny matrices (1 batch, 1 head, 3 tokens, head_dim=2):

```
Q = [[1, 0],       K = [[1, 0],       V = [[1, 2],
     [0, 1],            [0, 1],            [3, 4],
     [1, 1]]            [1, 1]]            [5, 6]]
```

**Q6.** Compute the raw attention scores: `scores = Q @ K^T`
Write out the full 3x3 matrix.

**Q7.** Apply scaling: divide every entry by sqrt(head_dim). What is sqrt(head_dim) here?
Write the scaled scores matrix.

**Q8.** Apply softmax to each ROW of the scaled scores.
Hint: softmax([a, b, c]) = [e^a, e^b, e^c] / (e^a + e^b + e^c)
Compute softmax for row 1 only (the others are similar).

**Q9.** Multiply your attention weights (from Q8, row 1 only) with V.
The result is the context vector for token 1.
What does this vector represent in plain English?

---

## Part 3: Causal Masking

Same Q, K, V as Part 2.

**Q10.** Write out the causal mask produced by `torch.tril(torch.ones(3, 3))`.

**Q11.** Take your scaled scores from Q7. Apply the causal mask:
replace every position where mask==0 with -inf.
Write the resulting matrix.

**Q12.** Now apply softmax to row 1 of the masked scores.
Compare this to your answer in Q8. What changed and why?

**Q13.** Apply softmax to row 2 of the masked scores.
What percentage of attention does token 2 give to token 3? Why?

**Q14.** In plain English, why does a language model NEED the causal mask?
What would go wrong without it?

---

## Part 4: Multi-Head Attention

Config: `d_model=8, n_heads=2, head_dim=4, seq_len=3`

**Q15.** Input x has shape `(1, 3, 8)`. After Q is computed by `nn.Linear(8, 8)`:
- Shape before `.view()`: ?
- Shape after `.view(1, 3, 2, 4)`: ?
- Shape after `.transpose(1, 2)`: ?

Explain what `.transpose(1, 2)` does in words.

**Q16.** After attention, the output has shape `(1, 2, 3, 4)`.
What does `.transpose(1, 2).contiguous().view(1, 3, 8)` do step by step?
Why do we need `.contiguous()`?

**Q17.** Why do we use multiple heads instead of one big head?
(Think: if d_model=256 and we use 1 head, the attention score matrix for one token pair is a dot product of two 256-dim vectors. With 4 heads, we get four dot products of 64-dim vectors instead. What's the benefit?)

---

## Part 5: GPT Forward Pass Trace

Config: `vocab_size=50, d_model=4, n_heads=2, max_length=3`
Input: `input_ids = [[7, 12, 3]]` — shape (1, 3)

**Q18.** `token_embedding` is a (50, 4) table. What happens when we do `token_embedding(input_ids)`?
What rows of the table do we look up? What is the output shape?

**Q19.** `pos_indices = torch.arange(3)` gives `[0, 1, 2]`.
`position_embedding` is a (3, 4) table.
What is `pos_emb` and what does it represent conceptually?

**Q20.** After `x = token_emb + pos_emb`, we get shape (1, 3, 4).
Why do we ADD token and position embeddings instead of concatenating them?
(Hint: what would the dimension be if we concatenated? What problems would that cause?)

**Q21.** After attention, `x` is still (1, 3, 4). Then `output_head` is `Linear(4, 50)`.
The output logits have shape (1, 3, 50). What does the entry `logits[0, 2, 7]` mean in plain English?

**Q22.** During training, we compare logits against target_ids using cross-entropy.
If `input_ids = [7, 12, 3]` and `target_ids = [12, 3, 25]`, explain the relationship.
Why is target shifted by one position?

---

## Part 6: Parameter Counting

Config: `vocab_size=200000, d_model=256, n_heads=4, max_length=256`

**Q23.** How many parameters in `token_embedding`? Show the calculation.

**Q24.** How many parameters in `position_embedding`?

**Q25.** `nn.Linear(d_model, d_model)` — how many parameters (including bias)?
Your attention class has 4 of these (query, key, value, out). Total for attention?

**Q26.** `output_head = nn.Linear(256, 200000, bias=False)` — how many parameters?

**Q27.** Add them all up. Does it roughly match the 102M parameters your model reported?
Which single component has the MOST parameters? Why?

---

## Part 7: Conceptual Questions

**Q28.** Your attention scores are computed as `Q @ K^T`. If two tokens have query and key vectors pointing in the same direction, is the attention score high or low? What does this mean — will the model "pay attention" to that token or ignore it?

**Q29.** After softmax, attention weights for one row sum to 1.0.
If there are 256 tokens in a sequence and attention is spread equally, each token gets weight 1/256 = 0.004.
Is uniform attention useful? What would you expect a trained model's attention pattern to look like?

**Q30.** Why do we divide scores by sqrt(head_dim) before softmax?
Hint: if head_dim=64 and Q, K entries are ~1.0 each, the dot product of two 64-dim vectors could be as large as ____. What happens to softmax when inputs are very large?

**Q31.** In your GPTModel, the embedding weights are random at initialization. The attention weights are random too. What kind of output do you expect from an untrained model? What is the expected initial loss for vocab_size=200000?

---

---

# Answers

Answers have been moved to a separate file. Solve the problems first, then ask me to check your answers — I'll give you the correct ones for whichever questions you've completed.
