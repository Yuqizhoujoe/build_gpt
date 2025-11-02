# %% [markdown]
# Byte pair encoding (BPE)
"""
Byte Pair Encoding (BPE) is a data compression algorithm that learns to merge
frequent pairs of tokens into single tokens.

How it works:
1. Start with individual bytes (or characters) as tokens
2. Find the most frequent pair of consecutive tokens
3. Merge that pair into a new single token
4. Repeat steps 2-3 until desired vocabulary size is reached

Example:
Starting with text: "hello hello"
- Initial tokens: ['h', 'e', 'l', 'l', 'o', ' ', 'h', 'e', 'l', 'l', 'o']
- Most frequent pair: ('l', 'l') appears 2 times
- Merge 'll' into a single token: ['h', 'e', 'll', 'o', ' ', 'h', 'e',
  'll', 'o']
- Next most frequent pair: ('h', 'e') appears 2 times
- Merge 'he' into a single token: ['he', 'll', 'o', ' ', 'he', 'll', 'o']
- Continue until vocabulary size is reached

BPE is commonly used in modern language models (like GPT) to create
efficient tokenizers that can handle any text while maintaining a reasonable
vocabulary size.
"""

# %%
text = (
    "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear "
    "and awe into the hearts of programmers worldwide. We all know we ought "
    "to "
    '"support Unicode" in our software (whatever that means—like using '
    "wchar_t for all the strings, right?). But Unicode can be abstruse, and "
    "diving into the thousand-page Unicode Standard plus its dozens of "
    "supplementary annexes, reports, and notes can be more than a little "
    "intimidating. I don't blame programmers for still finding the whole "
    "thing mysterious, even 30 years after Unicode's inception."
)

tokens_bytes = text.encode("utf-8")  # raw bytes
tokens = list(
    tokens_bytes
)  # convert to a list of integers in range 0..255 for convenience
print("---")
print(text)
print("length: ", len(text))
print("---")
print(tokens)
print("length:", len(tokens))


# %%
def get_stats(ids):
    """Count the frequency of consecutive pairs in a list of IDs.

    Args:
        ids: A list of integers representing token IDs.

    Returns:
        A dictionary mapping pairs (tuples) to their occurrence counts.

    Example:
        >>> get_stats([1, 2, 2, 3, 2, 2])
        {(1, 2): 1, (2, 2): 2, (2, 3): 1, (3, 2): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


stats = get_stats(tokens)
value_key_list = [(v, k) for k, v in stats.items()]
print(sorted(value_key_list, reverse=True))
# %%
top_pair = max(stats, key=lambda x: stats[x])
print(top_pair)


# %%
def merge(ids, pair, idx):
    """Replace all occurrences of a consecutive pair with a new token ID.

    Args:
        ids: A list of token IDs.
        pair: A tuple of two consecutive values to find and replace.
        idx: The new token ID to replace the pair with.

    Returns:
        A new list with all occurrences of the pair replaced by idx.

    Example:
        >>> merge([5, 6, 6, 7, 9, 1], (6, 7), 99)
        [5, 6, 99, 9, 1]
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))  # [5, 6, 99, 9, 1]
# %%
tokens2 = merge(tokens, top_pair, 256)
print(tokens2)
print("length:", len(tokens2))
# %%
VOCAB_SIZE = 276  # desired final vocabulary size
NUM_MERGES = VOCAB_SIZE - 256
ids = list(tokens)

merges = {}  # (int, int) -> int
for i in range(NUM_MERGES):
    stats = get_stats(ids)
    pair = max(stats.items(), key=lambda x: x[1])[0]
    idx = 256 + i
    print(f"merge {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

# %%
print("tokens length: ", len(tokens))
print("ids length: ", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

# %% [markdown]
# Note: the Tokenizer is a completely separate,
# independent module from the LLM.
#
# It has its own training dataset of text (which could be different from
# that of the LLM), on which you train the vocabulary using the Byte Pair
# Encoding (BPE) algorithm. It then translates back and forth between raw
# text and sequences of tokens. The LLM later only ever sees the tokens and
# never directly deals with any text.
#
# Diagram visualization:
#
#                     ┌─────────┐
#                     │   LLM   │
#                     └────┬────┘
#                          │
#                          ↑
#                     ┌────┴────┐
#                     │  token  │
#                     │sequence │
#                     └────┬────┘
#                          │
#                          ↑
#                     ┌────┴────┐
#                     │Tokenizer│
#                     └────┬────┘
#                          │
#                      ↑   │   ↓
#                      │   │   │
#                     ┌┴───┴───┴┐
#                     │ raw text │
#                     │(Unicode  │
#                     │code point│
#                     │sequence) │
#                     └──────────┘

# %%

# %%
