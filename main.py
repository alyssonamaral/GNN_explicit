import numpy as np

# --- Funções auxiliares ---

def relu(x):
    # ReLU: ativa apenas valores positivos (zera os negativos)
    return np.maximum(0, x)

def relu_deriv(x):
    # Derivada da ReLU: 1 onde x > 0, senão 0
    return (x > 0).astype(float)

def sigmoid(x):
    # Sigmoide: função logística para saída entre 0 e 1
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    # Derivada da sigmoide: s(x) * (1 - s(x))
    s = sigmoid(x)
    return s * (1 - s)

# --- Grafo (adjacência com self-loops já incluídos) ---
A = np.array([
    [1, 1, 1, 0],  # conexões do nó A
    [1, 1, 0, 1],  # conexões do nó B
    [1, 0, 1, 0],  # conexões do nó C
    [0, 1, 0, 1]   # conexões do nó D
], dtype=float)

# --- Normalização da adjacência (A_hat = D^{-1/2} * A * D^{-1/2}) ---
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A, axis=1)))
A_hat = D_inv_sqrt @ A @ D_inv_sqrt

# --- Features iniciais de cada nó (2 dimensões por nó) ---
H0 = np.array([
    [1.0, 2.0],   # nó A
    [0.5, 1.0],   # nó B
    [1.5, 0.5],   # nó C
    [2.0, 1.0]    # nó D
])

# --- Rótulos verdadeiros para cada nó (0 ou 1) ---
Y_target = np.array([
    [0.0],  # A
    [0.0],  # B
    [1.0],  # C
    [0.0]   # D
])

# --- Inicialização aleatória dos pesos das camadas ---
np.random.seed(42)
W1 = np.random.randn(2, 2) * 0.1  # pesos da GNN camada 1
W2 = np.random.randn(2, 2) * 0.1  # pesos da GNN camada 2
W3 = np.random.randn(2, 1) * 0.1  # perceptron final

# --- Hiperparâmetros de treinamento ---
lr = 0.1       # taxa de aprendizado
epochs = 100   # número de épocas

# --- Loop de treinamento ---
for epoch in range(epochs):
    # --- FORWARD PASS ---

    # Camada 1 da GNN: agregação + transformação linear + ReLU
    M1 = A_hat @ H0        # Agrega as features dos vizinhos
    Z1 = M1 @ W1           # Aplica pesos da 1ª camada
    H1 = relu(Z1)          # Ativação ReLU

    # Camada 2 da GNN: mesma lógica
    M2 = A_hat @ H1
    Z2 = M2 @ W2
    H2 = relu(Z2)          # Nova representação dos nós

    # Perceptron final: aplica pesos finais + sigmoide (classificação binária)
    Z3 = H2 @ W3           # Produto entre features finais e pesos do MLP
    Y_hat = sigmoid(Z3)    # Probabilidades entre 0 e 1 (uma por nó)

    # --- CÁLCULO DA PERDA (Loss) ---

    # Usamos Binary Cross-Entropy (BCE)
    loss = -np.mean(Y_target * np.log(Y_hat + 1e-10) + (1 - Y_target) * np.log(1 - Y_hat + 1e-10))

    # --- BACKPROPAGATION ---

   # Derivada da loss em relação à saída (Y_hat) - CORREÇÃO PARA BCE:
    dL_dYhat = - (Y_target / (Y_hat + 1e-10) - (1 - Y_target) / (1 - Y_hat + 1e-10)) / Y_target.size

    # Derivada da saída em relação à pré-ativação Z3 (sigmoid)
    dYhat_dZ3 = sigmoid_deriv(Z3)
    dL_dZ3 = dL_dYhat * dYhat_dZ3  

    # Gradiente dos pesos do perceptron (W3)
    dL_dW3 = H2.T @ dL_dZ3                          # shape: (2, 1)

    # Retropropagação do erro da saída para H2
    dL_dH2 = dL_dZ3 @ W3.T                          # shape: (4, 2)

    # Backprop na camada 2 da GNN
    dZ2 = relu_deriv(Z2)                            # derivada da ReLU
    dL_dZ2 = dL_dH2 * dZ2                           # gradiente total em Z2
    dL_dW2 = (A_hat @ H1).T @ dL_dZ2                # gradiente dos pesos W2

    # Backprop na camada 1 da GNN
    dL_dH1 = A_hat.T @ (dL_dZ2 @ W2.T)              # erro vindo da camada 2
    dZ1 = relu_deriv(Z1)                            # derivada da ReLU
    dL_dZ1 = dL_dH1 * dZ1                           # gradiente total em Z1
    dL_dW1 = (A_hat @ H0).T @ dL_dZ1                # gradiente dos pesos W1

    # --- Atualização dos pesos com gradiente descendente ---
    W3 -= lr * dL_dW3
    W2 -= lr * dL_dW2
    W1 -= lr * dL_dW1

    # --- Log da perda e saída atual (a cada 10 épocas) ---
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss:.5f} | Saídas: {Y_hat.T}")

# --- Resultados finais ---
print("\nSaídas finais dos nós (após treinamento):")
for i, y in enumerate(Y_hat):
    print(f"Nó {chr(65 + i)}: saída = {y[0]:.4f} | alvo = {Y_target[i][0]}")