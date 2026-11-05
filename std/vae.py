"""
VAE (Variational Autoencoder) explícito em NumPy, seguindo o capítulo 17 de
"Understanding Deep Learning" (Simon J.D. Prince).

Modelo (seção 17.2):
    Prior:          Pr(z)       = Norm_z[0, I]                      (eq. 17.5)
    Verossimilh.:   Pr(x|z,ϕ)   = Norm_x[f[z,ϕ], σ²I]               (eq. 17.6)
    Aprox. variac.: q(z|x,θ)    = Norm_z[g_µ[x,θ], g_Σ[x,θ]]        (eq. 17.20)

    f[z,ϕ] -> decoder (MLP), g[x,θ] -> encoder (MLP)

Loss = -ELBO, estimado com UMA amostra z* (eq. 17.23):
    ELBO ≈ log Pr(x|z*,ϕ) - D_KL[ q(z|x,θ) || Pr(z) ]

    -log Pr(x|z*,ϕ) = ||x - f[z*,ϕ]||² / (2σ²) + (D_x/2)·log(2πσ²)
    D_KL            = ½ (Tr[Σ] + µᵀµ - D_z - log det[Σ])            (eq. 17.24)

Truque da reparametrização (eq. 17.25):
    z* = µ + Σ^{1/2} ε*,   ε* ~ Norm[0, I]

Testado no Iris (4 features -> espaço latente 2D), para visualizar como as
3 espécies se organizam no espaço latente sem o modelo nunca ver os rótulos.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# --- Funções auxiliares ---

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# --- Dados: Iris ---
iris = load_iris()
X_raw = iris.data                 # shape (150, 4): sepal len, sepal wid, petal len, petal wid
y = iris.target                   # 0, 1, 2 -> usado só para colorir o gráfico, nunca no treino
nomes_especies = iris.target_names
nomes_features = iris.feature_names

# Padronização (média 0, desvio 1) -> combina com a variância fixa σ²I do decoder
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = (X_raw - X_mean) / X_std

N, D_x = X.shape   # 150 exemplos, 4 features
D_z = 2            # dimensão latente (2 para conseguir plotar)
D_h = 16           # neurônios na camada oculta do encoder e do decoder

# --- Inicialização dos pesos ---
np.random.seed(42)

def init(fan_in, fan_out):
    # inicialização de He (boa para ReLU)
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

params = {
    # Encoder g[x,θ]: x -> h1 -> (µ, log σ²)
    "W1":   init(D_x, D_h),  "b1":   np.zeros(D_h),
    "W_mu": init(D_h, D_z),  "b_mu": np.zeros(D_z),
    "W_lv": init(D_h, D_z) * 0.1,  "b_lv": np.zeros(D_z),   # começa com log σ² ≈ 0

    # Decoder f[z,ϕ]: z -> h2 -> x̂ (média da Gaussiana Pr(x|z,ϕ))
    "W2":   init(D_z, D_h),  "b2":   np.zeros(D_h),
    "W3":   init(D_h, D_x),  "b3":   np.zeros(D_x),
}

# --- Hiperparâmetros ---
sigma2 = 0.1      # variância σ² da verossimilhança (eq. 17.6). Menor -> reconstrução pesa mais que o KL
lr = 0.01
epochs = 2000
batch_size = 10

gerar_gif = True  # salva vae_iris.gif com o espaço latente se organizando durante o treino
n_frames = 150    # quantos "retratos" do espaço latente guardar para o GIF


# =====================================================================
# FORWARD: calcula -ELBO para um mini-batch
# =====================================================================
def forward(params, Xb, eps):
    # --- ENCODER g[x,θ] -> parâmetros da q(z|x,θ) (eq. 17.20) ---
    a1 = Xb @ params["W1"] + params["b1"]           # (B, D_h)
    h1 = relu(a1)
    mu = h1 @ params["W_mu"] + params["b_mu"]       # (B, D_z) média µ
    logvar = h1 @ params["W_lv"] + params["b_lv"]   # (B, D_z) log da diagonal de Σ
    # (prever log σ² em vez de σ² garante Σ > 0 sem restrição nenhuma)

    # --- REPARAMETRIZAÇÃO: z* = µ + Σ^{1/2} ε* (eq. 17.25) ---
    std = np.exp(0.5 * logvar)                      # Σ^{1/2} (diagonal)
    z = mu + std * eps                              # (B, D_z)

    # --- DECODER f[z,ϕ] -> média da Pr(x|z,ϕ) (eq. 17.6) ---
    a2 = z @ params["W2"] + params["b2"]            # (B, D_h)
    h2 = relu(a2)
    x_hat = h2 @ params["W3"] + params["b3"]        # (B, D_x) saída linear (é uma média Gaussiana)

    # --- LOSS = -ELBO (eq. 17.23) ---
    # Termo 1: -log Pr(x|z*,ϕ) da Gaussiana esférica
    recon = np.sum((Xb - x_hat) ** 2, axis=1) / (2 * sigma2) \
            + 0.5 * D_x * np.log(2 * np.pi * sigma2)                     # (B,)

    # Termo 2: KL entre Norm[µ, Σ] e Norm[0, I] (eq. 17.24)
    #   Tr[Σ] = Σ exp(logvar),  µᵀµ = Σ µ²,  log det[Σ] = Σ logvar
    kl = 0.5 * np.sum(np.exp(logvar) + mu ** 2 - 1 - logvar, axis=1)    # (B,)

    loss = np.mean(recon + kl)

    cache = (Xb, a1, h1, mu, logvar, std, eps, z, a2, h2, x_hat)
    return loss, np.mean(recon), np.mean(kl), cache


# =====================================================================
# BACKWARD: gradientes de -ELBO em relação a todos os parâmetros
# =====================================================================
def backward(params, cache):
    Xb, a1, h1, mu, logvar, std, eps, z, a2, h2, x_hat = cache
    B = Xb.shape[0]
    g = {}

    # --- Decoder ---
    # d(recon)/d(x_hat) = -(x - x_hat) / σ²   (dividido por B por causa da média)
    dL_dxhat = -(Xb - x_hat) / sigma2 / B                 # (B, D_x)

    g["W3"] = h2.T @ dL_dxhat
    g["b3"] = dL_dxhat.sum(axis=0)
    dL_dh2 = dL_dxhat @ params["W3"].T                    # (B, D_h)

    dL_da2 = dL_dh2 * relu_deriv(a2)
    g["W2"] = z.T @ dL_da2
    g["b2"] = dL_da2.sum(axis=0)
    dL_dz = dL_da2 @ params["W2"].T                       # (B, D_z) erro que chega no z*

    # --- Reparametrização: z = µ + exp(½ logvar)·ε ---
    #   dz/dµ = 1
    #   dz/dlogvar = ½ exp(½ logvar)·ε = ½·std·ε
    # É AQUI que o truque funciona: ε é constante, então o gradiente passa para µ e Σ.
    # Somamos também a derivada do KL:
    #   dKL/dµ = µ
    #   dKL/dlogvar = ½ (exp(logvar) - 1)
    dL_dmu = dL_dz * 1.0 + mu / B
    dL_dlogvar = dL_dz * 0.5 * std * eps + 0.5 * (np.exp(logvar) - 1) / B

    # --- Encoder ---
    g["W_mu"] = h1.T @ dL_dmu
    g["b_mu"] = dL_dmu.sum(axis=0)
    g["W_lv"] = h1.T @ dL_dlogvar
    g["b_lv"] = dL_dlogvar.sum(axis=0)

    # h1 alimenta as duas cabeças (µ e logvar) -> os gradientes se somam
    dL_dh1 = dL_dmu @ params["W_mu"].T + dL_dlogvar @ params["W_lv"].T

    dL_da1 = dL_dh1 * relu_deriv(a1)
    g["W1"] = Xb.T @ dL_da1
    g["b1"] = dL_da1.sum(axis=0)

    return g


# =====================================================================
# CHECAGEM DE GRADIENTE (diferenças finitas) — para confiar no backprop
# =====================================================================
def checar_gradientes(params, Xb, h=1e-5):
    eps = np.random.randn(Xb.shape[0], D_z)     # ε fixo: assim a loss é determinística
    _, _, _, cache = forward(params, Xb, eps)
    g = backward(params, cache)

    print("Checagem de gradiente (erro relativo, analítico vs numérico):")
    for nome, P in params.items():
        num = np.zeros_like(P)
        for idx in np.ndindex(P.shape):
            original = P[idx]
            P[idx] = original + h
            L_mais = forward(params, Xb, eps)[0]
            P[idx] = original - h
            L_menos = forward(params, Xb, eps)[0]
            P[idx] = original
            num[idx] = (L_mais - L_menos) / (2 * h)
        erro = np.linalg.norm(g[nome] - num) / (np.linalg.norm(g[nome]) + np.linalg.norm(num) + 1e-12)
        print(f"  {nome:5s}: {erro:.2e}")
    print()

checar_gradientes(params, X[:10])


# =====================================================================
# TREINAMENTO (Adam explícito, mini-batches)
# =====================================================================
beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
m = {k: np.zeros_like(v) for k, v in params.items()}   # 1º momento
v = {k: np.zeros_like(v) for k, v in params.items()}   # 2º momento
t = 0

hist_loss, hist_recon, hist_kl = [], [], []

# Para o GIF: loss a cada passo de gradiente + µ de todas as flores em alguns passos.
# Os passos são espaçados em escala log, porque a maior parte da organização acontece no começo.
passos_por_epoca = int(np.ceil(N / batch_size))
passos_snapshot = set(np.unique(np.geomspace(1, epochs * passos_por_epoca, n_frames).astype(int)))
passo_recon, passo_kl = [], []
snapshots = []   # lista de (passo, µ de todo o dataset)

for epoch in range(epochs):
    perm = np.random.permutation(N)
    loss_ep = recon_ep = kl_ep = 0.0

    for inicio in range(0, N, batch_size):
        idx = perm[inicio:inicio + batch_size]
        Xb = X[idx]

        # ε* ~ Norm[0, I]: o "ramo estocástico" fora do caminho do gradiente (fig. 17.11)
        eps = np.random.randn(len(idx), D_z)

        loss, recon, kl, cache = forward(params, Xb, eps)
        grads = backward(params, cache)

        # --- Atualização Adam ---
        t += 1
        for k in params:
            m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
            v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
            m_hat = m[k] / (1 - beta1 ** t)
            v_hat = v[k] / (1 - beta2 ** t)
            params[k] -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)

        passo_recon.append(recon)
        passo_kl.append(kl)
        if gerar_gif and t in passos_snapshot:
            mu_snap = forward(params, X, np.zeros((N, D_z)))[3][3]   # cache[3] = µ
            snapshots.append((t, mu_snap))

        n_b = len(idx) / N
        loss_ep += loss * n_b
        recon_ep += recon * n_b
        kl_ep += kl * n_b

    hist_loss.append(loss_ep)
    hist_recon.append(recon_ep)
    hist_kl.append(kl_ep)

    if (epoch + 1) % 200 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:4d} | -ELBO: {loss_ep:8.4f} | Recon: {recon_ep:8.4f} | KL: {kl_ep:.4f}")


# =====================================================================
# RESULTADOS
# =====================================================================
# Passa o dataset inteiro pelo encoder (sem amostrar: usamos µ como "posição" de cada flor)
_, _, _, cache = forward(params, X, np.zeros((N, D_z)))
_, _, _, mu_all, logvar_all, std_all, _, _, _, _, x_hat_all = cache

erro_recon = np.mean((X - x_hat_all) ** 2)
print(f"\nMSE de reconstrução (usando z = µ, dados padronizados): {erro_recon:.4f}")
print(f"σ médio da q(z|x) por dimensão latente: {np.round(std_all.mean(axis=0), 3)}")

print("\nCentro (média de µ) de cada espécie no espaço latente:")
for c, nome in enumerate(nomes_especies):
    print(f"  {nome:10s}: {np.round(mu_all[y == c].mean(axis=0), 3)}")

# --- Geração (seção 17.2.1): z* ~ Pr(z) -> f[z*,ϕ] -> x* ---
n_gerados = 300
z_prior = np.random.randn(n_gerados, D_z)
h2_gen = relu(z_prior @ params["W2"] + params["b2"])
x_gen = h2_gen @ params["W3"] + params["b3"]
x_gen = x_gen + np.sqrt(sigma2) * np.random.randn(*x_gen.shape)   # amostra de Norm[f[z*], σ²I]
x_gen_raw = x_gen * X_std + X_mean                                 # volta para centímetros


# --- Gráficos ---
cores = ["#2a78d6", "#eb6834", "#1baf7a"]   # setosa, versicolor, virginica
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 1) Espaço latente: µ de cada flor + elipse de 1σ da q(z|x)
ax = axes[0]
for c, nome in enumerate(nomes_especies):
    sel = y == c
    for (mx, my), (sx, sy) in zip(mu_all[sel], std_all[sel]):
        ax.add_patch(plt.matplotlib.patches.Ellipse(
            (mx, my), 2 * sx, 2 * sy, color=cores[c], alpha=0.06, lw=0))
    ax.scatter(mu_all[sel, 0], mu_all[sel, 1], s=22, color=cores[c],
               edgecolor="white", linewidth=0.6, label=nome)
ax.set_title("Espaço latente (µ de q(z|x), elipses = 1σ)")
ax.set_xlabel("z₁")
ax.set_ylabel("z₂")
ax.set_aspect("equal")
ax.legend(frameon=False)

# 2) Curvas de treino
ax = axes[1]
ax.plot(hist_recon, color=cores[0], lw=2, label="Reconstrução  -log Pr(x|z)")
ax.plot(hist_kl, color=cores[1], lw=2, label="KL[q(z|x) || Pr(z)]")
ax.set_title("Termos da -ELBO durante o treino")
ax.set_xlabel("Época")
ax.set_ylim(min(min(hist_recon), 0) - 0.5, 8)   # corta o pico das primeiras épocas
ax.legend(frameon=False)

# 3) Dados reais vs gerados a partir do prior
ax = axes[2]
ax.scatter(x_gen_raw[:, 2], x_gen_raw[:, 3], s=14, color="#9a9994", alpha=0.5,
           label="gerados (z ~ Norm[0, I])")
for c, nome in enumerate(nomes_especies):
    sel = y == c
    ax.scatter(X_raw[sel, 2], X_raw[sel, 3], s=22, color=cores[c],
               edgecolor="white", linewidth=0.6, label=f"real: {nome}")
ax.set_title("Reais vs gerados pelo decoder")
ax.set_xlabel(nomes_features[2])
ax.set_ylabel(nomes_features[3])
ax.legend(frameon=False)

for ax in axes:
    ax.grid(alpha=0.2)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

plt.tight_layout()
plt.savefig("vae_iris.png", dpi=130)
print("\nFigura salva em vae_iris.png")


# =====================================================================
# GIF: espaço latente se organizando + evolução da Loss e do KL
# =====================================================================
if gerar_gif:
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import FancyBboxPatch

    fundo, tinta = "#000000", "#f0efec"
    cores_gif = ["#3987e5", "#d95926", "#199e70"]   # mesmas espécies, tons para fundo escuro
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})

    passo_recon = np.array(passo_recon)
    passo_kl = np.array(passo_kl)
    total_passos = len(passo_recon)
    passo_eixo = np.arange(1, total_passos + 1)

    # Limite fixo dos eixos latentes (para os pontos "se mexerem" dentro do mesmo quadro)
    lim = np.ceil(max(np.abs(mu).max() for _, mu in snapshots) * 1.1)

    fig_gif = plt.figure(figsize=(13, 5.5), facecolor=fundo)

    # --- Painel principal: espaço latente com eixos em cruz e setas ---
    ax_z = fig_gif.add_axes([0.2, 0.03, 0.45, 0.76], facecolor=fundo)
    ax_z.set_xlim(-lim * 1.08, lim * 1.08)
    ax_z.set_ylim(-lim * 1.08, lim * 1.08)
    ax_z.set_aspect("equal")
    ax_z.axis("off")
    seta = dict(arrowstyle="-|>", color=tinta, lw=1.5, mutation_scale=18)
    ax_z.annotate("", xy=(lim, 0), xytext=(-lim, 0), arrowprops=seta)
    ax_z.annotate("", xy=(0, lim), xytext=(0, -lim), arrowprops=seta)
    for tick in np.arange(-lim + 1, lim):
        if tick != 0:
            ax_z.plot([tick, tick], [-0.06 * lim / 3, 0.06 * lim / 3], color=tinta, lw=1.2)
            ax_z.plot([-0.06 * lim / 3, 0.06 * lim / 3], [tick, tick], color=tinta, lw=1.2)
    ax_z.text(lim, -0.12 * lim, "$z_1$", color=tinta, fontsize=14, ha="right", va="top")
    ax_z.text(0.08 * lim, lim, "$z_2$", color=tinta, fontsize=14, va="top")

    pontos = ax_z.scatter(snapshots[0][1][:, 0], snapshots[0][1][:, 1], s=22,
                          c=[cores_gif[c] for c in y], edgecolors="none", alpha=0.9)

    # --- Títulos e legenda (caixinhas coloridas, como na referência) ---
    fig_gif.text(0.425, 0.9, "Espaço Latente  $\\mu$ de $q(z|x)$", color=tinta,
                 fontsize=22, ha="center", va="center")
    fig_gif.add_artist(plt.Line2D([0.3, 0.55], [0.85, 0.85], color=tinta, lw=2))
    txt_passo = fig_gif.text(0.02, 0.9, "", color=tinta, fontsize=18, va="center")

    for c, nome in enumerate(nomes_especies):
        y0 = 0.66 - c * 0.11
        fig_gif.add_artist(FancyBboxPatch((0.03, y0), 0.022, 0.05, boxstyle="square,pad=0",
                                          fc=cores_gif[c], ec="#3a3a38", lw=2,
                                          transform=fig_gif.transFigure))
        fig_gif.text(0.06, y0 + 0.025, nome, color=tinta, fontsize=15, va="center")

    # --- Inset: evolução da Loss (reconstrução) e do KL ---
    ax_l = fig_gif.add_axes([0.72, 0.5, 0.25, 0.3], facecolor=fundo)
    ax_l.set_xscale("log")   # eixo log: os frames também são espaçados em log
    ax_l.set_xlim(1, total_passos)
    ax_l.set_ylim(0, 10)     # o pico inicial (~130) sai do quadro de propósito
    for lado in ("top", "right"):
        ax_l.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax_l.spines[lado].set_color(tinta)
    ax_l.tick_params(colors=tinta, labelsize=9)
    ax_l.set_xlabel("passo (log)", color=tinta, fontsize=10)
    ax_l.set_title("Evolução da Loss", color=tinta, fontsize=15, pad=24)
    ax_l.text(0.55, 1.04, "$\\mathcal{L}_{KL}$", color=cores_gif[1], fontsize=14,
              transform=ax_l.transAxes)
    ax_l.text(0.80, 1.04, "$\\mathcal{L}_{recon}$", color=cores_gif[0], fontsize=14,
              transform=ax_l.transAxes)
    linha_recon, = ax_l.plot([], [], color=cores_gif[0], lw=1.5)
    linha_kl, = ax_l.plot([], [], color=cores_gif[1], lw=1.5)

    # Média móvel curta: a loss por mini-batch é ruidosa (z* é amostrado)
    def suavizar(a, janela=10):
        c = np.cumsum(np.insert(a, 0, 0.0))
        out = np.empty_like(a)
        for i in range(len(a)):
            ini = max(0, i + 1 - janela)
            out[i] = (c[i + 1] - c[ini]) / (i + 1 - ini)
        return out

    recon_suave = suavizar(passo_recon)
    kl_suave = suavizar(passo_kl)

    def atualizar(frame):
        passo, mu_snap = snapshots[frame]
        pontos.set_offsets(mu_snap)
        txt_passo.set_text(f"Passo de Gradiente: {passo}")
        linha_recon.set_data(passo_eixo[:passo], recon_suave[:passo])
        linha_kl.set_data(passo_eixo[:passo], kl_suave[:passo])
        return pontos, txt_passo, linha_recon, linha_kl

    # Repete o último frame por ~2 s para o GIF "parar" no resultado final antes de reiniciar
    fps = 15
    ordem_frames = list(range(len(snapshots))) + [len(snapshots) - 1] * (2 * fps)
    anim = FuncAnimation(fig_gif, atualizar, frames=ordem_frames, interval=1000 / fps)
    anim.save("vae_iris.gif", writer=PillowWriter(fps=fps), dpi=90,
              savefig_kwargs={"facecolor": fundo})
    plt.close(fig_gif)
    print("GIF salvo em vae_iris.gif")

plt.show()
