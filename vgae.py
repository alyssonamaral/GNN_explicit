import numpy as np
from itertools import product

# --- Funções auxiliares ---

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# --- Grafo com self-loops ---
A = np.array([
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
], dtype=float)

# --- Normalização da adjacência (Â = D^{-1/2} A D^{-1/2}) ---
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A, axis=1)))
A_hat = D_inv_sqrt @ A @ D_inv_sqrt

# --- Features iniciais dos nós ---
H0 = np.array([
    [1.0, 2.0],   # nó A
    [0.5, 1.0],   # nó B
    [1.5, 0.5],   # nó C
    [2.0, 1.0]    # nó D
])

# --- Inicialização dos pesos do encoder e decoder ---
np.random.seed(42)

# Encoder
W_mu = np.random.randn(2, 2) * 0.1      # para média
W_logvar = np.random.randn(2, 2) * 0.1  # para log variância

# Decoder MLP (camada oculta + saída)
W_dec1 = np.random.randn(4, 4) * 0.1    # entrada: z_i || z_j ∈ ℝ⁴
b_dec1 = np.zeros(4)

W_dec2 = np.random.randn(4, 1) * 0.1    # saída: probabilidade
b_dec2 = np.zeros(1)

# --- Hiperparâmetros ---
lr = 0.1
epochs = 100
N = A.shape[0]

# --- Treinamento ---
for epoch in range(epochs):
    # --- ENCODER (GNN que aprende μ e logσ²) ---
    M = A_hat @ H0

    mu = M @ W_mu
    logvar = M @ W_logvar

    # Reparametrização: z = μ + σ * ε
    eps = np.random.randn(*mu.shape)
    std = np.exp(0.5 * logvar)
    Z = mu + std * eps

    # --- DECODER MLP ---
    A_pred = np.zeros_like(A)
    cache = {}  # para armazenar intermediários para backprop

    for i, j in product(range(N), repeat=2):
        z_ij = np.concatenate([Z[i], Z[j]])            # concatena z_i e z_j → shape (4,)
        h1 = relu(z_ij @ W_dec1 + b_dec1)              # primeira camada MLP
        out = sigmoid(h1 @ W_dec2 + b_dec2)            # saída do decoder
        A_pred[i, j] = out
        cache[(i, j)] = (z_ij, h1, out)                # salva para usar no gradiente

    # --- CÁLCULO DA PERDA ---
    bce = -np.mean(A * np.log(A_pred + 1e-10) + (1 - A) * np.log(1 - A_pred + 1e-10))
    kl = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))
    loss = bce + kl

    # --- BACKPROPAGATION DECODER ---
    dW_dec1 = np.zeros_like(W_dec1)
    db_dec1 = np.zeros_like(b_dec1)
    dW_dec2 = np.zeros_like(W_dec2)
    db_dec2 = np.zeros_like(b_dec2)
    dZ = np.zeros_like(Z)

    for i, j in product(range(N), repeat=2):
        z_ij, h1, out = cache[(i, j)]

        # Gradiente da loss wrt saída do decoder
        dL_dout = - (A[i, j] / (out + 1e-10) - (1 - A[i, j]) / (1 - out + 1e-10)) / (N * N)

        # Sigmoid e ReLU backward
        dout_dh1 = sigmoid_deriv(h1 @ W_dec2 + b_dec2) * W_dec2.T
        dh1_dz = relu_deriv(z_ij @ W_dec1 + b_dec1) * W_dec1.T

        # Gradientes da MLP decoder
        dW_dec2 += np.outer(h1, dL_dout * sigmoid_deriv(h1 @ W_dec2 + b_dec2))
        db_dec2 += dL_dout * sigmoid_deriv(h1 @ W_dec2 + b_dec2)

        delta_h1 = dL_dout * dout_dh1
        dW_dec1 += np.outer(z_ij, delta_h1)
        db_dec1 += delta_h1.reshape(-1)

        # Propagando para Z (metade para i, metade para j)
        dz_ij = (delta_h1 @ W_dec1.T).flatten()  # Garante shape (4,)
        dZ[i] += dz_ij[:2]  # Primeiras 2 dimensões
        dZ[j] += dz_ij[2:]  # Últimas 2 dimensões

    # --- BACKPROP ENCODER (usando dZ) ---
    dZ_dmu = 1
    dZ_dlogvar = 0.5 * std * eps

    dL_dmu = dZ * dZ_dmu + mu / mu.size
    dL_dlogvar = dZ * dZ_dlogvar + 0.5 * (np.exp(logvar) - 1) / logvar.size

    dW_mu = (A_hat @ H0).T @ dL_dmu
    dW_logvar = (A_hat @ H0).T @ dL_dlogvar

    # --- Atualização dos pesos ---
    W_mu -= lr * dW_mu
    W_logvar -= lr * dW_logvar
    W_dec1 -= lr * dW_dec1
    b_dec1 -= lr * db_dec1
    W_dec2 -= lr * dW_dec2
    b_dec2 -= lr * db_dec2

    # --- Log ---
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss:.5f} | BCE: {bce:.5f} | KL: {kl:.5f}")

# --- Resultados finais ---
print("\nMatriz de adjacência real (A):")
print(A.astype(int))

print("\nMatriz de adjacência reconstruída (A_pred):")
print(np.round(A_pred, 2))

print("\nEmbeddings finais (Z):")
print(np.round(Z, 3))