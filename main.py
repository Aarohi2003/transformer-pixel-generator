import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("✅ Imports ready!")
print(f"   PyTorch {torch.__version__}")

# ── Fixed 16-colour palette (NES-inspired) ───────────────────────────────────
PALETTE = np.array([
    [  0,   0,   0],  #  0  black        (background)
    [255, 255, 255],  #  1  white
    [220,  60,  60],  #  2  red
    [ 60, 180,  60],  #  3  green
    [ 60,  80, 220],  #  4  blue
    [220, 200,  50],  #  5  yellow
    [160,  90,  30],  #  6  brown
    [150,  50, 200],  #  7  purple
    [ 50, 200, 200],  #  8  cyan
    [255, 160,  60],  #  9  orange
    [255, 180, 180],  # 10  pink
    [130, 130, 130],  # 11  grey
    [255, 215,  80],  # 12  gold
    [ 30, 130, 130],  # 13  teal
    [240, 120,  80],  # 14  coral
    [220, 235, 220],  # 15  off-white
], dtype=np.uint8)

NUM_COLOURS = len(PALETTE)   # 16
print(f"Palette: {NUM_COLOURS} colours")

def _t(rows):
    out = np.zeros((16, 16), dtype=np.uint8)
    for r, s in enumerate(rows):
        for c, ch in enumerate(s.ljust(16)[:16]):
            out[r, c] = int(ch) if ch.isdigit() else 0
    return out

TEMPLATES = {
    "hero": _t([
        "0000011111100000",
        "0000122222210000",
        "0000122222210000",
        "0000122222210000",
        "0000012222100000",
        "0000033333300000",
        "0000333333330000",
        "0000333333330000",
        "0000333333330000",
        "0000433333340000",
        "0000440000440000",
        "0000440000440000",
        "0000440000440000",
        "0000440000440000",
        "0000550000550000",
        "0000550000550000",
    ]),
    "mushroom": _t([
        "0000000000000000",
        "0000011100000000",
        "0000111110000000",
        "0001211121000000",
        "0012111121200000",
        "0012111111200000",
        "0001211121000000",
        "0000111110000000",
        "0000022200000000",
        "0000222220000000",
        "0000232232000000",
        "0000232232000000",
        "0000022200000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
    ]),
    "tree": _t([
        "0000000000000000",
        "0000000300000000",
        "0000003330000000",
        "0000033333000000",
        "0000333333300000",
        "0003333333330000",
        "0033333133330000",
        "0333331133333000",
        "0003333333330000",
        "0000003630000000",
        "0000003630000000",
        "0000006660000000",
        "0000006660000000",
        "0000066666000000",
        "0000000000000000",
        "0000000000000000",
    ]),
    "house": _t([
        "0000000000000000",
        "0000000200000000",
        "0000002220000000",
        "0000022222000000",
        "0000222222200000",
        "0002222222220000",
        "0001111111110000",
        "0001111111110000",
        "0001113311110000",
        "0001113311110000",
        "0001114411110000",
        "0001114411110000",
        "0001114411110000",
        "0001111111110000",
        "0000000000000000",
        "0000000000000000",
    ]),
    "potion": _t([
        "0000000000000000",
        "0000000000000000",
        "0000001100000000",
        "0000001100000000",
        "0000011110000000",
        "0000122210000000",
        "0001122221000000",
        "0001133221000000",
        "0001133221000000",
        "0001122221000000",
        "0001122221000000",
        "0000122210000000",
        "0000011110000000",
        "0000001100000000",
        "0000000000000000",
        "0000000000000000",
    ]),
    "gem": _t([
        "0000000000000000",
        "0000000000000000",
        "0000011111000000",
        "0000111111100000",
        "0001111111110000",
        "0011121112110000",
        "0011211211210000",
        "0001121211100000",
        "0000112211000000",
        "0000011110000000",
        "0000001100000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
    ]),
}


def build_sprite_dataset(n_per_template=40, noise_std=5):
    """
    Generate n_per_template colour variations for every template.
    Returns a list of (16, 16, 3) uint8 RGB arrays.
    """
    sprites = []
    for name, template in TEMPLATES.items():
        unique_parts = sorted(set(template.flatten()) - {0})
        for _ in range(n_per_template):
            # Randomly assign palette colours to template parts
            palette_pool = random.sample(range(1, NUM_COLOURS), len(unique_parts))
            mapping = {0: 0, **{p: c for p, c in zip(unique_parts, palette_pool)}}
            ids  = np.vectorize(mapping.get)(template).astype(np.uint8)
            rgb  = PALETTE[ids].copy()
            # Add mild noise to simulate real-world sprites
            noise = np.random.randint(-noise_std, noise_std+1, rgb.shape, dtype=np.int16)
            rgb   = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            sprites.append(rgb)
    random.shuffle(sprites)
    return sprites


SPRITES_RGB = build_sprite_dataset(n_per_template=40)

print(f"✅ Dataset built: {len(SPRITES_RGB)} sprites")
print(f"   Each sprite shape: {SPRITES_RGB[0].shape}  (height × width × RGB)")
print(f"   Templates: {list(TEMPLATES.keys())}")

# ── Display 12 sample sprites ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
for ax, sprite in zip(axes.flat, SPRITES_RGB[:12]):
    ax.imshow(sprite, interpolation='nearest', aspect='equal')
    ax.axis('off')
fig.suptitle("Sample training sprites (RGB, with mild noise)", fontsize=11, fontweight='500')
plt.tight_layout()
plt.show()
print("These are the raw RGB images. Your next job: quantise them back to palette IDs.")


def build_colour_map(palette):
    """
    Pre-compute a lookup table from EVERY possible 24-bit RGB colour
    to its nearest palette index.

    With 256^3 = 16M possible colours this would be huge, so instead
    we return a function (closure) that computes the nearest index on demand.

    Args
    ----
    palette : np.ndarray  shape (NUM_COLOURS, 3)  dtype uint8

    Returns
    -------
    nearest_id : callable  (rgb_pixel: array-like of shape (3,)) -> int
    """
    def nearest_id(pixel):
        pixel = np.array(pixel, dtype=np.int32)

        dists = np.sum((palette.astype(np.int32) - pixel)**2, axis=1)  # shape (NUM_COLOURS,) — one distance per palette entry
        return int(np.argmin(dists))

    return nearest_id

nearest_colour = build_colour_map(PALETTE)

# Quick tests
print("Test nearest_colour:")
print(f"  Pure black   (0,0,0)     → id {nearest_colour([0,0,0])}  (expect 0)")
print(f"  Pure white   (255,255,255) → id {nearest_colour([255,255,255])}  (expect 1)")
print(f"  Near-red     (218,58,62) → id {nearest_colour([218,58,62])}  (expect 2)")


def quantise_sprite(rgb_img, palette, nearest_id_fn):
    """
    Convert a (H, W, 3) uint8 RGB image to a (H, W) array of palette indices.

    Args
    ----
    rgb_img      : np.ndarray  shape (H, W, 3)  dtype uint8
    palette      : np.ndarray  shape (NUM_COLOURS, 3)
    nearest_id_fn: the callable returned by build_colour_map()

    Returns
    -------
    ids : np.ndarray  shape (H, W)  dtype uint8
          Every value is in range [0, NUM_COLOURS)
    """
    H, W, _ = rgb_img.shape
    ids = np.zeros((H, W), dtype=np.uint8)

    
    for r in range(H):
        for c in range(W):
            ids[r, c] = nearest_id_fn(rgb_img[r, c])

    return ids

# ── Test on one sprite ────────────────────────────────────────────────────────
sample_ids = quantise_sprite(SPRITES_RGB[0], PALETTE, nearest_colour)
print(f"Quantised sprite shape : {sample_ids.shape}")
print(f"Unique palette IDs used: {sorted(set(sample_ids.flatten()))}")
print(f"All values in [0,15]   : {sample_ids.max() <= 15}")
print()
print("First 4 rows of colour IDs:")
print(sample_ids[:4])

# ── Quantise all sprites and visually verify round-trip ──────────────────────
SPRITES_IDS = [quantise_sprite(s, PALETTE, nearest_colour) for s in SPRITES_RGB]

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i in range(8):
    axes[0, i].imshow(SPRITES_RGB[i], interpolation='nearest')
    axes[0, i].axis('off')
    axes[0, i].set_title("original", fontsize=7)

    axes[1, i].imshow(PALETTE[SPRITES_IDS[i]], interpolation='nearest')
    axes[1, i].axis('off')
    axes[1, i].set_title("quantised", fontsize=7)

fig.suptitle("Original vs quantised (noise removed, palette snapped)", fontsize=10, fontweight='500')
plt.tight_layout()
plt.show()
print(f"✅ All {len(SPRITES_IDS)} sprites quantised successfully.")

def sprite_to_rows(sprite_ids):
    """
    Split a (16, 16) palette-ID array into a list of 16 row tuples.

    Args
    ----
    sprite_ids : np.ndarray  shape (16, 16)  dtype uint8

    Returns
    -------
    rows : list of 16 tuples, each tuple has 16 ints (palette IDs)

    Example
    -------
    sprite_ids[0]  = [3, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0]
    rows[0]        = (3, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0)
    """
 
    rows = [tuple(sprite_ids[r]) for r in range(16)]
    return rows

# Test
test_rows = sprite_to_rows(SPRITES_IDS[0])
print(f"Number of rows  : {len(test_rows)}")
print(f"Type of row     : {type(test_rows[0])}")
print(f"Length of row   : {len(test_rows[0])}")
print(f"Row 0           : {test_rows[0]}")

def build_row_vocab(all_row_sequences):
    """
    Collect every unique row tuple across all sprites and assign it an integer ID.

    Args
    ----
    all_row_sequences : list of lists — each inner list is the 16-row sequence
                        for one sprite (as returned by sprite_to_rows)

    Returns
    -------
    row_to_id : dict  {row_tuple: int}   — row  →  token ID
    id_to_row : dict  {int: row_tuple}   — token ID  →  row
    """

    
    unique_rows = sorted(set(row for seq in all_row_sequences for row in seq))
    row_to_id   = {row: idx for idx, row in enumerate(unique_rows)}
    id_to_row   =  {idx: row for row, idx in row_to_id.items()}

    return row_to_id, id_to_row

all_row_sequences = [sprite_to_rows(s) for s in SPRITES_IDS]
row_to_id, id_to_row = build_row_vocab(all_row_sequences)

ROW_VOCAB_SIZE = len(row_to_id)
print(f"✅ Row vocabulary built")
print(f"   Unique rows (tokens) : {ROW_VOCAB_SIZE}")
print(f"   240 sprites × 16 rows = {240*16} rows total, {ROW_VOCAB_SIZE} unique")


def encode_sprite(sprite_ids, row_to_id):
    """
    Convert a (16, 16) palette-ID array to a list of 16 row token IDs.

    Args
    ----
    sprite_ids : np.ndarray  shape (16, 16)
    row_to_id  : dict  {row_tuple: int}

    Returns
    -------
    token_ids : list of 16 ints
    """
  
    #
    rows      = sprite_to_rows(sprite_ids)
    token_ids = [row_to_id[row] for row in rows]
    return token_ids

# Test
test_encoded = encode_sprite(SPRITES_IDS[0], row_to_id)
print(f"Encoded sprite: {test_encoded}")
print(f"Type          : {type(test_encoded)}, length = {len(test_encoded)}")
print(f"All IDs valid  : {all(0 <= t < ROW_VOCAB_SIZE for t in test_encoded)}")



CONTEXT_LEN = 8   # model sees 8 rows, predicts the 9th

encoded_sprites = [encode_sprite(s, row_to_id) for s in SPRITES_IDS]

X_list, y_list = [], []
for seq in encoded_sprites:
    for i in range(len(seq) - CONTEXT_LEN):
        X_list.append(seq[i : i + CONTEXT_LEN])  # context window
        y_list.append(seq[i + CONTEXT_LEN])  # target row ID

X_all = torch.tensor(X_list, dtype=torch.long)   # (N, CONTEXT_LEN)
y_all = torch.tensor(y_list, dtype=torch.long)   # (N,)

print(f"✅ Training windows: {len(X_all)}")
print(f"   X_all shape : {X_all.shape}")
print(f"   y_all shape : {y_all.shape}")
print(f"\nExample window #0:")
print(f"  Context row IDs : {X_all[0].tolist()}")
print(f"  Target  row ID  : {y_all[0].item()}")



row_pixels = torch.zeros(ROW_VOCAB_SIZE, 16, dtype=torch.long)   # shape (VOCAB_SIZE, 16)

for row_tuple, token_id in row_to_id.items():
    row_pixels[token_id] = torch.tensor(list(row_tuple), dtype=torch.long)

print(f"row_pixels shape  : {row_pixels.shape}")
print(f"row_pixels dtype  : {row_pixels.dtype}")
print(f"Max colour ID     : {row_pixels.max().item()}  (should be ≤ 15)")
print(f"\nrow_pixels[0] = {row_pixels[0].tolist()}  ← pixel data for token 0")
print(f"id_to_row[0]  = {list(id_to_row[0])}  ← should match above")

COLOUR_DIM = 8    # embedding size per colour inside RowEncoder
D_MODEL    = 64   # main model dimension
N_HEADS    = 4    # attention heads  (D_MODEL ÷ N_HEADS = 16 per head)
D_FF       = 128  # FFN hidden size
N_LAYERS   = 2    # stacked transformer blocks
DROPOUT    = 0.1

print(f"RowEncoder  : {NUM_COLOURS} colours × {COLOUR_DIM}d → flatten {16*COLOUR_DIM}d → project {D_MODEL}d")
print(f"Transformer : d_model={D_MODEL}, {N_HEADS} heads, {N_LAYERS} layers, d_ff={D_FF}")
print(f"Output head : {D_MODEL}d → {ROW_VOCAB_SIZE} row vocab")

class RowEncoder(nn.Module):
    """
    Encodes a row of 16 colour IDs into a single d_model-dimensional vector.

    Unlike a plain nn.Embedding (one vector per token), the RowEncoder
    shares colour embeddings across all row positions, so similar rows
    produce similar output vectors.

    Forward pass:
        Input  : (B, T, 16)   — 16 colour IDs per row
        Embed  : (B, T, 16, colour_dim)
        Flatten: (B, T, 16 * colour_dim)
        Project: (B, T, d_model)
    """
    def __init__(self, num_colours, colour_dim, d_model):
        super().__init__()
        self.colour_dim = colour_dim
        # One embedding vector per palette colour
        self.colour_emb = nn.Embedding(num_colours, colour_dim)
        # Project concatenated pixel embeddings to model dimension
        self.proj       = nn.Linear(16 * colour_dim, d_model)

    def forward(self, x):
        
        B, T, P = x.shape   # P = 16 (pixels per row)

        embedded = self.colour_emb(x)

     
        flat = embedded.view(B, T, P * self.colour_dim)

        
        return self.proj(flat)

# ── Shape test ────────────────────────────────────────────────────────────────
_enc = RowEncoder(NUM_COLOURS, COLOUR_DIM, D_MODEL)
_x   = torch.randint(0, NUM_COLOURS, (2, 8, 16))   # batch=2, seq=8, pixels=16
_out = _enc(_x)
print("RowEncoder output shape:", _out.shape)
assert _out.shape == (2, 8, D_MODEL), "❌ Check your embed/view/proj steps"
print("✅ RowEncoder — shape correct!")


class PositionalEncoding(nn.Module):
    """
    Adds a fixed sin/cos position fingerprint to each row embedding.
    Identical to the music notebook — included here for completeness.
    """
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # TODO 9a: Fill even columns with sin
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

_pe = PositionalEncoding(D_MODEL)
assert _pe(torch.zeros(2, 8, D_MODEL)).shape == (2, 8, D_MODEL)
print("✅ PositionalEncoding — shape correct!")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x);  K = self.W_k(x);  V = self.W_v(x)

       
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal mask — provided, do not change
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)                     
        context = torch.matmul(weights, V)
                       


        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.W_o(context)

_mha = MultiHeadAttention(D_MODEL, N_HEADS)
assert _mha(torch.randn(2, 8, D_MODEL)).shape == (2, 8, D_MODEL)
print("✅ MultiHeadAttention — shape correct!")


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn    = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attn_out = self.attn(x)
        x   = self.norm1(x + self.dropout(attn_out))

        # TODO 11b: FFN sub-layer with residual + LayerNorm.
        ffn_out  = self.ffn(x)
        x        = self.norm2(x + self.dropout(ffn_out))
        return x

_blk = TransformerBlock(D_MODEL, N_HEADS, D_FF)
assert _blk(torch.randn(2, 8, D_MODEL)).shape == (2, 8, D_MODEL)
print("✅ TransformerBlock — shape correct!")


class PixelTransformer(nn.Module):
    """
    Decoder-only Transformer for sprite generation.

    Key difference from a text/music transformer:
    Input tokens are row IDs, but they're embedded via RowEncoder
    (pixel-aware) rather than a plain nn.Embedding (ID-lookup only).
    The row_pixels buffer bridges the two representations.
    """
    def __init__(self, vocab_size, row_pixels, num_colours, colour_dim,
                 d_model, n_heads, d_ff, n_layers, max_len=50, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('row_pixels', row_pixels)

        self.row_encoder =  RowEncoder(num_colours, colour_dim, d_model)

        self.pos_enc     = PositionalEncoding(d_model, max_len, dropout)

        self.blocks      = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout)
                                  for _ in range(n_layers)])

        self.output_proj =  nn.Linear(d_model, vocab_size)

    def forward(self, row_ids):
        # row_ids: (B, T) — sequence of row token IDs
        pixels = self.row_pixels[row_ids]                 # (B, T, 16)  lookup
        x      = self.row_encoder(pixels)                 # (B, T, D_MODEL)
        x      = x * math.sqrt(self.d_model)              # scale
        x      = self.pos_enc(x)                          # add position
        for block in self.blocks:
            x  = block(x)                                 # (B, T, D_MODEL)
        return self.output_proj(x)                        # (B, T, VOCAB_SIZE)

model = PixelTransformer(
    vocab_size  = ROW_VOCAB_SIZE,
    row_pixels  = row_pixels,
    num_colours = NUM_COLOURS,
    colour_dim  = COLOUR_DIM,
    d_model     = D_MODEL,
    n_heads     = N_HEADS,
    d_ff        = D_FF,
    n_layers    = N_LAYERS,
    dropout     = DROPOUT,
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ PixelTransformer created!  Parameters: {n_params:,}")

_test  = torch.randint(0, ROW_VOCAB_SIZE, (4, CONTEXT_LEN))
_out   = model(_test)
print(f"   Input  shape: {_test.shape}  (batch=4, context={CONTEXT_LEN})")
print(f"   Output shape: {_out.shape}  (batch=4, context={CONTEXT_LEN}, vocab={ROW_VOCAB_SIZE})")
assert _out.shape == (4, CONTEXT_LEN, ROW_VOCAB_SIZE)
print("✅ End-to-end forward pass — shape correct!")

BATCH_SIZE = 32
NUM_EPOCHS = 400
LR         = 5e-4

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
)

print(f"✅ Loss: {loss_fn}")
print(f"✅ Optimiser: Adam  lr={LR}")
print(f"   Random baseline loss: log({ROW_VOCAB_SIZE}) = {math.log(ROW_VOCAB_SIZE):.3f}")

model.train()
loss_history = []

print(f"Training for {NUM_EPOCHS} epochs  "
      f"({len(X_all)} windows, batch={BATCH_SIZE})...")
print(f"Random baseline ≈ {math.log(ROW_VOCAB_SIZE):.2f}")
print()

for epoch in range(NUM_EPOCHS):
    perm   = torch.randperm(len(X_all))
    X_shuf = X_all[perm];  y_shuf = y_all[perm]
    epoch_loss, n_batches = 0.0, 0

    for i in range(0, len(X_all), BATCH_SIZE):
        xb     = X_shuf[i : i + BATCH_SIZE]
        yb     = y_shuf[i : i + BATCH_SIZE]
        logits = model(xb)[:, -1, :]          # predict the next row
        loss   = loss_fn(logits, yb)

        optimizer.zero_grad()   # 14a
        loss.backward()         # 14b
        optimizer.step()

        epoch_loss += loss.item();  n_batches += 1

    avg = epoch_loss / n_batches
    loss_history.append(avg)
    scheduler.step()

    if (epoch + 1) % 50 == 0:
        pct = max(0, 1 - avg / math.log(ROW_VOCAB_SIZE))
        bar = '█' * int(28*pct) + '░' * (28 - int(28*pct))
        print(f"Epoch {epoch+1:>3}/{NUM_EPOCHS}  loss={avg:.4f}  [{bar}]  "
              f"lr={scheduler.get_last_lr()[0]:.1e}")

print("\n🎮 Training complete!")


fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(loss_history, color='steelblue', lw=2)
ax.axhline(math.log(ROW_VOCAB_SIZE), color='tomato', ls='--', lw=1.5,
           label=f'Random baseline  log({ROW_VOCAB_SIZE}) ≈ {math.log(ROW_VOCAB_SIZE):.2f}')
ax.set_xlabel('Epoch');  ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('PixelTransformer Training Curve', fontweight='500')
ax.legend();  ax.grid(alpha=0.3)
plt.tight_layout();  plt.show()
print(f"Final loss: {loss_history[-1]:.4f}")


def decode_sequence(row_id_sequence, id_to_row):
    """
    Convert a list of row token IDs to a 2-D numpy pixel grid.

    Args
    ----
    row_id_sequence : list of ints   — row token IDs (length = num rows)
    id_to_row       : dict           — maps int → row tuple of 16 colour IDs

    Returns
    -------
    grid : np.ndarray  shape (len(row_id_sequence), 16)  dtype uint8
           Each row contains 16 palette colour IDs.
    """
   
    rows = [list(id_to_row[rid]) for rid in row_id_sequence]
    return np.array(rows, dtype=np.uint8)

# Quick test
test_seq  = encode_sprite(SPRITES_IDS[0], row_to_id)
test_grid = decode_sequence(test_seq, id_to_row)
print(f"decode_sequence output shape : {test_grid.shape}")
print(f"Max colour ID                : {test_grid.max()}  (should be ≤ 15)")
round_trip = np.array_equal(test_grid, SPRITES_IDS[0])
print(f"Round-trip identical to original : {round_trip}")


def generate_sprite(seed_row_ids, num_generate=8, temperature=0.9):
    """
    Autoregressively generate `num_generate` new rows.

    Args
    ----
    seed_row_ids : list of ints  — row token IDs for the seed rows (top of sprite)
    num_generate : int           — how many new rows to generate
    temperature  : float         — < 1 safer, > 1 more adventurous

    Returns
    -------
    full_grid : np.ndarray  shape (len(seed)+num_generate, 16)
    """
    model.eval()
    ctx       = list(seed_row_ids[-CONTEXT_LEN:])   # sliding context window
    generated = list(seed_row_ids)                   # full sequence so far

    with torch.no_grad():
        for _ in range(num_generate):
            x      = torch.tensor([ctx], dtype=torch.long)   # (1, CONTEXT_LEN)
            logits = model(x)[0, -1, :]                       # (VOCAB_SIZE,)

            logits = logits/temperature

            probs  = F.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_id)
            ctx = ctx[1:] + [next_id]    # slide window

    return decode_sequence(generated, id_to_row)

# Quick test
seed_ids = encode_sprite(SPRITES_IDS[0], row_to_id)[:8]
result   = generate_sprite(seed_ids, num_generate=8, temperature=0.9)
print(f"generate_sprite output shape: {result.shape}   (should be (16, 16))")


def render_sprite(pixel_grid, ax=None, title="", zoom=8):
    """
    Display a 2-D palette-ID grid as a pixel art image.

    Args
    ----
    pixel_grid : np.ndarray  shape (H, W)  values 0–15
    ax         : matplotlib Axes (creates a new one if None)
    title      : string for the subplot title
    zoom       : upscale factor for the displayed image

    Returns
    -------
    ax : the matplotlib Axes used
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(zoom*0.4, zoom*0.4))

   
    
    rgb = PALETTE[pixel_grid]
    ax.imshow(rgb, interpolation='nearest', aspect='equal')
    if title:
        ax.set_title(title, fontsize=8, pad=2)
    ax.axis('off')
    return ax

# Quick test — should show a colourful sprite
fig, ax = plt.subplots(figsize=(3,3))
render_sprite(SPRITES_IDS[0], ax=ax, title="Test sprite")
plt.tight_layout()
plt.show()
print("✅ render_sprite working!")

model.eval()
categories = list(TEMPLATES.keys())
fig, axes  = plt.subplots(len(categories), 3, figsize=(7, len(categories)*2.2))
fig.suptitle("Left: original   |   Middle: seed (top 8 rows)   |   Right: model completion",
             fontsize=10, fontweight='500', y=1.01)

sprites_by_cat = {cat: [] for cat in categories}
for sprite_ids in SPRITES_IDS:
    for idx, (cat, tmpl) in enumerate(TEMPLATES.items()):
        if len(sprites_by_cat[cat]) < 1:
            row_ids = encode_sprite(sprite_ids, row_to_id)
            sprites_by_cat[cat].append((sprite_ids, row_ids))

for row_idx, cat in enumerate(categories):
    sprite_ids, row_ids = sprites_by_cat[cat][0]
    seed_ids = row_ids[:8]

    seed_grid     = decode_sequence(seed_ids, id_to_row)
    padding       = np.zeros((8, 16), dtype=np.uint8)
    seed_with_pad = np.vstack([seed_grid, padding])

    completion    = generate_sprite(seed_ids, num_generate=8, temperature=0.85)

    render_sprite(sprite_ids,    ax=axes[row_idx, 0], title=f"{cat} — original")
    render_sprite(seed_with_pad, ax=axes[row_idx, 1], title="seed (top 8 rows)")
    render_sprite(completion,    ax=axes[row_idx, 2], title="model completion")

plt.tight_layout()
plt.show()



np.random.seed(0)
fig, axes = plt.subplots(4, 10, figsize=(16, 7))

for col, ax_pair in enumerate(zip(axes[:2].T, axes[2:].T)):
    ax_orig, ax_gen = ax_pair[0][0], ax_pair[1][0]
    # pick a random sprite
    sprite_ids = random.choice(SPRITES_IDS)
    row_ids    = encode_sprite(sprite_ids, row_to_id)
    seed_ids   = row_ids[:8]
    completion = generate_sprite(seed_ids, num_generate=8, temperature=0.9)
    render_sprite(sprite_ids, ax=ax_orig, title="orig" if col==0 else "")
    render_sprite(completion, ax=ax_gen,  title="gen"  if col==0 else "")

for ax in axes.flat:
    ax.axis('off')

axes[0,0].set_title("original", fontsize=9, pad=2)
axes[2,0].set_title("generated", fontsize=9, pad=2)
fig.suptitle("20 random sprite completions  (top half seeded, bottom half generated)",
             fontsize=10, fontweight='500')
plt.tight_layout()
plt.show()

