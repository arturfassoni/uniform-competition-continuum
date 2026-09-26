"""Figures for "Fitness is asymptotically irrelevant under uniform competition
in phenotype-structured populations".

Produces
  fig1_mechanism.pdf  (Figure 1: mechanism of the proof, Section 1.3)
  fig2_numerics.pdf   (Figure 2: numerical illustration, Section 6.2)
  fig3_examples.pdf   (Figure 3: whole line, jumps, OU without rate, Section 7.3)
and prints the numbers quoted in the text (spectral gap, distances, memory times).

Run with: python3 make_figures.py   (about a minute)
Dependencies: numpy, scipy, matplotlib.

All PDEs are discretised by the finite-volume scheme of Section 6.2,
    du_i/dt = (J_{i+1/2} - J_{i-1/2}) / h_i,
    J_{i+1/2} = a_{i+1/2} (u_{i+1}/psi_{i+1} - u_i/psi_i) / (x_{i+1} - x_i),
with a = D psi, zero flux at both ends, which conserves mass, preserves
positivity and has psi as exact stationary state (also on non-uniform grids).
"""
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp, quad
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import expm_multiply
from scipy.special import erf, expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 8.5,
    "text.color": "0.15", "axes.labelcolor": "0.15", "axes.edgecolor": "0.4",
    "xtick.color": "0.3", "ytick.color": "0.3",
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.fontsize": 7, "legend.frameon": False,
    "lines.linewidth": 1.4, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})
C_LIGHT, C_MED, C_DARK = "#8db8e8", "#3b82d6", "#0b2350"
C_ORANGE, C_SHADE, C_GREY = "#e8663a", "#f0eeea", "#8a8a8a"


# ---------------------------------------------------------------- numerics
def fv_generator(edges, D, logpsi):
    """Finite-volume switching generator on the cells defined by `edges`.
    D(x) diffusivity, logpsi(x) = int v/D (unnormalised). Returns
    (L sparse, centres, widths, psi normalised at centres)."""
    c = 0.5 * (edges[1:] + edges[:-1]); h = np.diff(edges)
    lp_c = logpsi(c); lp_e = logpsi(edges[1:-1]); m = lp_c.max()
    psi_c = np.exp(lp_c - m); a = D(edges[1:-1]) * np.exp(lp_e - m)
    w = a / np.diff(c)                          # a_{i+1/2}/(x_{i+1}-x_i)
    n = len(c); rows, cols, vals = [], [], []
    for i in range(n - 1):                      # flux between cells i, i+1
        for (r_, c_, s) in [(i, i + 1, 1), (i, i, -1), (i + 1, i + 1, -1), (i + 1, i, 1)]:
            rows.append(r_); cols.append(c_)
            vals.append(s * w[i] / h[r_] / psi_c[c_])
    L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    psi = psi_c / (h * psi_c).sum()
    return L, c, h, psi


def simulate(L, h, r, u0, T, t_eval, additive=False, rtol=1e-9):
    """u' = L u + g(U) r u with g(U) = 1 - U (uniform competition), or
    u' = L u + (r - U) u (additive competition)."""
    L = sp.csr_matrix(L)

    def rhs(t, u):
        U = h @ u
        return L @ u + ((r - U) * u if additive else (1 - U) * r * u)

    def jac(t, u):                              # rank-one part omitted
        U = h @ u
        return L + sp.diags((r - U) if additive else (1 - U) * r)

    s = solve_ivp(rhs, [0, T], u0, t_eval=t_eval, method="BDF",
                  jac=jac, rtol=rtol, atol=1e-13)
    assert s.success, s.message
    return s.t, s.y


def gap(L, psi, h):
    """Spectral gap of a generator that is self-adjoint for the weighted
    scalar product sum_i h_i f_i g_i / psi_i (true for all generators here)."""
    s = np.sqrt(h / psi)
    B = (sp.diags(s) @ sp.csr_matrix(L) @ sp.diags(1 / s)).toarray()
    ev = np.sort(eigvalsh(0.5 * (B + B.T)))
    return -ev[-2]


# double-well example of Section 6.2
D0 = 0.02
P = lambda x: 0.03 * np.cos(4 * np.pi * x) + 0.03 * x
r_dw = lambda x: 0.1 + 1.9 * expit((x - 0.5) / 0.02)
N = 400
edges = np.linspace(0, 1, N + 1)


def dw_generator(Dval):
    return fv_generator(edges, lambda x: Dval + 0 * x, lambda x: -P(x) / Dval)


L_dw, x_dw, h_dw, psi_dw = dw_generator(D0)
r = r_dw(x_dw)
lam1 = gap(L_dw, psi_dw, h_dw)
print("double well: lambda1 = %.4f" % lam1)

t1 = np.linspace(0, 150, 3001)
tu, yu = simulate(L_dw, h_dw, r, 0.01 * psi_dw, 150, t1)
ta, ya = simulate(L_dw, h_dw, r, 0.01 * psi_dw, 150, t1, additive=True)
Uu, Ua = h_dw @ yu, h_dw @ ya
dist_u = h_dw @ np.abs(yu / Uu - psi_dw[:, None])
dist_a = h_dw @ np.abs(ya / Ua - psi_dw[:, None])
print("uniform: max dist %.3f, dist(150) = %.2e" % (dist_u.max(), dist_u[-1]))


def shade_fast(ax, x0=0.5, x1=1.0, label=True, y=0.97):
    ax.axvspan(x0, x1, color=C_SHADE, zorder=0, lw=0)
    if label:
        ax.text(0.03, y, r"$r\approx0.1$", transform=ax.transAxes, fontsize=6.5,
                color=C_GREY, va="top")
        ax.text(0.97, y, r"$r\approx2$", transform=ax.transAxes, fontsize=6.5,
                color=C_GREY, va="top", ha="right")


# =================================================== Figure: mechanism
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.3))
# (a) total population and its growth rate
ax = axs[0]
tt = np.linspace(0, 12, 1201)
_, yy = simulate(L_dw, h_dw, r, 0.01 * psi_dw, 12, tt)
UU = h_dw @ yy
dU = (1 - UU) * (r @ (h_dw[:, None] * yy))
ax.fill_between(tt, 0, dU, color=C_LIGHT, alpha=0.55, lw=0)
ax.plot(tt, dU, color=C_MED, lw=1.0)
ax.plot(tt, UU, color=C_DARK)
ax.axhline(1, color=C_GREY, lw=0.6, ls=":")
ax.text(12, 1.03, r"$U^*$", ha="right", va="bottom", fontsize=7, color=C_GREY)
ax.text(6.3, 0.80, r"$U(t)$", color=C_DARK, fontsize=7.5)
ax.text(4.2, 0.36, r"$|U'(t)|$", color=C_MED, fontsize=7.5)
ax.annotate(r"area $=|U^*-U_0|$", xy=(2.9, 0.12), xytext=(5.6, 0.2),
            fontsize=7, color=C_DARK,
            arrowprops=dict(arrowstyle="-", color=C_DARK, lw=0.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 1.15)
ax.set_xlabel(r"time $t$")
ax.set_title(r"(a) $U$ is monotone, $\int|U'|<\infty$", loc="left")
# (b) new cells vs stationary density
ax = axs[1]
ib = np.argmax(dU); tb = tt[ib]; ub = yy[:, ib]
newborn = r * ub / (r @ (h_dw * ub))            # r u / rho : density of new cells
wpert = newborn - psi_dw                         # = phi / U'  (zero mass)
shade_fast(ax)
ax.fill_between(x_dw, psi_dw, newborn, where=wpert > 0, color=C_MED, alpha=0.35, lw=0)
ax.fill_between(x_dw, psi_dw, newborn, where=wpert < 0, color=C_ORANGE, alpha=0.35, lw=0)
ax.plot(x_dw, psi_dw, "k--", lw=1.1)
ax.plot(x_dw, newborn, color=C_DARK, lw=1.2)
ax.annotate(r"$\varphi>0$", xy=(0.755, 3.0), xytext=(0.9, 3.3), ha="center", va="center", fontsize=7.5,
            color=C_DARK, arrowprops=dict(arrowstyle="-", color=C_DARK, lw=0.6))
ax.text(0.25, 1.1, r"$\varphi<0$", ha="center", fontsize=7, color="#a33a14")
ax.text(0.84, 4.8, r"$ru/\rho$", fontsize=7.5, color=C_DARK)
ax.text(0.36, 2.9, r"$\psi$", fontsize=7.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 6.6); ax.set_xticks([0, 0.5, 1])
ax.set_xlabel(r"phenotype $x$")
ax.set_title(r"(b) forcing $\varphi=U'(ru/\rho-\psi)$", loc="left")
print("mechanism (b): t_b = %.2f, ||ru/rho - psi|| = %.3f" % (tb, h_dw @ np.abs(wpert)))
# (c) Doeblin: switching cancels the common part of w+ and w-
ax = axs[2]
wp, wm = np.maximum(wpert, 0), np.maximum(-wpert, 0)
tau = 10.0
Swp = expm_multiply(tau * L_dw, wp); Swm = expm_multiply(tau * L_dw, wm)
common = np.minimum(Swp, Swm)
ratio = (h_dw @ np.abs(Swp - Swm)) / (h_dw @ np.abs(wpert))
print("mechanism (c): tau = %g, ||S w||/||w|| = %.3f" % (tau, ratio))
shade_fast(ax, label=False)
ax.plot(x_dw, wp, color=C_MED, lw=0.7, alpha=0.45, label=r"$w^\pm$")
ax.plot(x_dw, wm, color=C_ORANGE, lw=0.7, alpha=0.45)
ax.fill_between(x_dw, 0, common, color="0.55", alpha=0.45, lw=0, label="common part")
ax.plot(x_dw, Swp, color=C_MED, label=r"$S(\tau)w^+$")
ax.plot(x_dw, Swm, color=C_ORANGE, label=r"$S(\tau)w^-$")
ax.legend(loc="upper center", fontsize=6.8, handlelength=1.4, borderaxespad=0.0,
          bbox_to_anchor=(0.5, 1.02), labelspacing=0.25)
ax.text(0.98, 0.98, r"$\tau=%g$" % tau, transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color=C_GREY)
ax.set_xlim(0, 1); ax.set_ylim(0, 5.2); ax.set_xticks([0, 0.5, 1])
ax.set_xlabel(r"phenotype $x$")
ax.set_title(r"(c) the overlap cancels", loc="left")
fig.savefig("fig1_mechanism.pdf", bbox_inches="tight"); plt.close(fig)


# ============================== Figure: numerical illustration (Fig. numerics)
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
Ds = np.geomspace(0.008, 0.1, 22)
lams = np.array([gap(*[dw_generator(Dv)[k] for k in (0, 3, 2)]) for Dv in Ds])
xs = np.linspace(0, 1, 20001); Ps = P(xs)
iR = np.argmin(np.where(xs > 0.5, Ps, np.inf)); iS = np.argmax(np.where((xs > 0.3) & (xs < 0.7), Ps, -np.inf))
dP = Ps[iS] - Ps[iR]
print("Kramers: barrier from the shallow well dP = %.4f" % dP)

fig = plt.figure(figsize=(6.8, 7.0))
gs = GridSpec(3, 1, height_ratios=[0.75, 1.3, 0.95], hspace=0.5, figure=fig)
row0 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.25)
row1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.12)
row2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.42)
# (a) landscape, (b) proliferation rate
ax = fig.add_subplot(row0[0])
ax.plot(xs, Ps, color="0.15", lw=1.2)
ax.set_xlim(0, 1); ax.set_xlabel(r"phenotype $x$"); ax.set_ylabel(r"$P(x)$")
ax.set_title(r"(a) landscape $P$ (tilted double well)", loc="left")
ax = fig.add_subplot(row0[1])
shade_fast(ax, label=False)
ax.plot(x_dw, r, color="0.35", lw=1.2)
ax.set_xlim(0, 1); ax.set_ylim(0, 2.1); ax.set_xlabel(r"phenotype $x$"); ax.set_ylabel(r"$r(x)$")
ax.set_title(r"(b) proliferation rate $r$", loc="left")
# (c), (d) kymographs of the composition
T_show = 50.0
m = t1 <= T_show
pu, pa = yu[:, m] / Uu[m], ya[:, m] / Ua[m]
vmax = max(pu.max(), pa.max())
norm = PowerNorm(0.6, vmin=0, vmax=vmax)
axc = fig.add_subplot(row1[0])
axd = fig.add_subplot(row1[1], sharey=axc)
for ax, pp, ttl in [(axc, pu, r"(c) $u/U$, uniform competition"),
                    (axd, pa, r"(d) $u/U$, additive competition")]:
    im = ax.imshow(pp.T, extent=[0, 1, 0, T_show], origin="lower", aspect="auto",
                   cmap="Greys", norm=norm)
    ax.set_ylim(T_show, 0); ax.set_xlabel(r"phenotype $x$"); ax.set_title(ttl, loc="left")
    ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
axc.set_ylabel(r"time $t$")
plt.setp(axd.get_yticklabels(), visible=False)
cb = fig.colorbar(im, ax=[axc, axd], pad=0.015, fraction=0.03)
cb.set_label(r"$u/U$"); cb.ax.tick_params(labelsize=7)
# (e) total population
ax = fig.add_subplot(row2[0])
m20 = t1 <= 20
ax.plot(t1[m20], Ua[m20], color=C_ORANGE); ax.plot(t1[m20], Uu[m20], color=C_MED)
ax.axhline(1, color=C_GREY, lw=0.6, ls=":")
ax.text(0.3, 1.04, r"$U^*$", fontsize=7.5, color=C_GREY)
ax.text(19.5, 1.85, "additive", ha="right", va="top", fontsize=8, color=C_ORANGE)
ax.text(19.5, 0.92, "uniform", ha="right", va="top", fontsize=8, color=C_MED)
ax.set_xlim(0, 20); ax.set_ylim(0, 2.1)
ax.set_xlabel(r"time $t$"); ax.set_title(r"(e) total population $U$", loc="left")
# (f) distance to psi
ax = fig.add_subplot(row2[1])
m100 = t1 <= 100
ax.semilogy(t1[m100], dist_a[m100], color=C_ORANGE); ax.semilogy(t1[m100], dist_u[m100], color=C_MED)
k40 = np.argmin(abs(t1 - 40)); tt_ref = np.linspace(25, 100, 10)
ax.semilogy(tt_ref, 4 * dist_u[k40] * np.exp(-lam1 * (tt_ref - 40)), ":", color=C_GREY, lw=1)
ax.text(50, 0.62, r"$\propto e^{-\lambda_1t}$", fontsize=8, color=C_GREY, va="center")
ax.text(98, 1.6, "additive", ha="right", va="bottom", fontsize=8, color=C_ORANGE)
ax.text(4, 4e-3, "uniform", fontsize=8, color=C_MED)
ax.set_xlim(0, 100); ax.set_ylim(1e-3, 4)
ax.set_xlabel(r"time $t$"); ax.set_title(r"(f) $\|u/U-\psi\|_{L^1}$", loc="left")
# (g) memory time 1/lambda_1 versus 1/D (Kramers)
ax = fig.add_subplot(row2[2])
ax.semilogy(1 / Ds, 1 / lams, "o-", color=C_DARK, ms=2.3, lw=1.0)
ref = (1 / lams[0]) * np.exp(dP * (1 / Ds - 1 / Ds[0]))
ax.semilogy(1 / Ds, ref, ":", color=C_GREY, lw=1)
ax.text(80, ref[np.argmin(abs(1 / Ds - 80))] * 3, r"$\propto e^{\Delta P/D}$", fontsize=8,
        color=C_GREY, ha="right")
ax.plot([1 / D0], [1 / lam1], "o", mfc="white", mec=C_MED, mew=1.2, ms=5.5, zorder=5)
ax.annotate("(a)\u2013(f)", xy=(1 / D0, 1 / lam1), xytext=(1 / D0 + 32, 1 / lam1 / 5),
            fontsize=7.5, color=C_MED, ha="center",
            arrowprops=dict(arrowstyle="-", color=C_MED, lw=0.6))
ax.axhline(5, color=C_ORANGE, lw=0.8, ls="--")
ax.text(126, 5.8, "growth", fontsize=7.5, color=C_ORANGE, ha="right", va="bottom")
ax.set_xlabel(r"$1/D$"); ax.set_title(r"(g) memory time $1/\lambda_1$", loc="left")
ax.set_xlim(0, 128)
fig.savefig("fig2_numerics.pdf", bbox_inches="tight"); plt.close(fig)
print("memory times: D=%.3f -> %.1f, D=%.3f -> %.3g" % (Ds[-1], 1 / lams[-1], Ds[0], 1 / lams[0]))


# ================================================== Figure: other examples
def run_example(L, x, h, psi, rr, T, nt=301):
    t = np.linspace(0, T, nt)
    _, y = simulate(L, h, rr, 0.01 * psi, T, t, rtol=1e-8)
    U = h @ y
    p = y / U
    d = h @ np.abs(p - psi[:, None])
    return t, p, d, U


examples = {}
# (a) Ornstein-Uhlenbeck on R (truncated to [-8, 8])
th, Dou = 1.0, 0.5
L, x, h, psi = fv_generator(np.linspace(-8, 8, 641), lambda z: Dou + 0 * z,
                            lambda z: -th * z ** 2 / (2 * Dou))
rr = 0.1 + 1.9 * expit((x - 0.5) / 0.1)
examples["a"] = (x, psi, rr) + run_example(L, x, h, psi, rr, 15)
print("OU: gap %.3f (theta = %g)" % (gap(L, psi, h), th))
# (b) state-dependent noise, heavy tails: v = -theta x, D = D0 (1 + x^2)
thh, Dh = 0.65, 0.3
xi = np.linspace(-np.arcsinh(1000 / 2), np.arcsinh(1000 / 2), 901)
L, x, h, psi = fv_generator(2 * np.sinh(xi), lambda z: Dh * (1 + z ** 2),
                            lambda z: -thh / (2 * Dh) * np.log1p(z ** 2))
rr = 0.1 + 1.9 * expit((x - 2) / 0.2)
examples["b"] = (x, psi, rr) + run_example(L, x, h, psi, rr, 40)
print("heavy tails: gap %.3f" % gap(L, psi, h))
# (c) Metropolis jumps on (0,1) with pi = psi of Section 6.2 (no diffusion)
Nj = 200; xj = (np.arange(Nj) + 0.5) / Nj; hj = np.full(Nj, 1 / Nj)
pij = np.exp(-P(xj) / D0); pij /= hj @ pij
sq, kj = 0.2, 1.0
q = np.exp(-(xj[:, None] - xj[None, :]) ** 2 / (2 * sq ** 2)) / (np.sqrt(2 * np.pi) * sq)
K = kj * q * np.minimum(1, pij[:, None] / pij[None, :])   # K[i, j]: rate density j -> i
np.fill_diagonal(K, 0)
G = K * hj[None, :]                                       # gains: sum_j K_ij u_j h
G = G - np.diag(G.sum(0))                                 # losses: kappa_j u_j
G = sp.csr_matrix(G)
print("jumps: max |G pi| = %.1e, gap %.4f" % (abs(G @ pij).max(), gap(G, pij, hj)))
rr = r_dw(xj)
examples["c"] = (xj, pij, rr) + run_example(G, xj, hj, pij, rr, 60)

fig = plt.figure(figsize=(6.4, 6.3))
outer = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)
spec = {"a": dict(xl=(-3.5, 3.5), T=15, title="(a) Ornstein\u2013Uhlenbeck on $\\mathbb{R}$",
                  fast=(0.5, 3.5), logy=False),
        "b": dict(xl=(-12, 12), T=40, title=r"(b) $D=D_0(1+x^2)$: heavy tails",
                  fast=(2, 12), logy=True),
        "c": dict(xl=(0, 1), T=60, title="(c) Metropolis jumps, no diffusion",
                  fast=(0.5, 1), logy=False)}
pos = {"a": outer[0, 0], "b": outer[0, 1], "c": outer[1, 0]}
for key in "abc":
    x, psi, rr, t, p, d, U = examples[key]
    s = spec[key]
    inner = pos[key].subgridspec(2, 1, height_ratios=[1.25, 1], hspace=0.08)
    ak = fig.add_subplot(inner[0]); ap = fig.add_subplot(inner[1], sharex=ak)
    sel = (x >= s["xl"][0]) & (x <= s["xl"][1])
    ak.pcolormesh(x[sel], t, p[sel].T, cmap="Greys", shading="auto",
                  norm=PowerNorm(0.55, vmin=0, vmax=p[sel].max()), rasterized=True)
    ak.set_ylim(s["T"], 0); ak.set_ylabel(r"time $t$")
    ak.tick_params(labelbottom=False)
    ak.set_title(s["title"], loc="left")
    kmax = np.argmax(d)
    ap.axvspan(*s["fast"], color=C_SHADE, lw=0, zorder=0)
    ap.plot(x[sel], p[sel, kmax], color=C_LIGHT, label=r"$t=%.0f$" % t[kmax])
    ap.plot(x[sel], p[sel, -1], color=C_DARK, label=r"$t=%.0f$" % t[-1])
    ap.plot(x[sel], psi[sel], "k--", lw=1.1, label=r"$\psi$")
    if s["logy"]:
        ap.set_yscale("log"); ap.set_ylim(3e-4, 1)
    else:
        ap.set_ylim(0, 1.15 * max(p[sel, kmax].max(), psi.max()))
    ap.set_xlim(*s["xl"]); ap.set_xlabel(r"phenotype $x$")
    ap.set_ylabel(r"$u/U$")
    print("example %s: max dist %.3f at t=%.1f, final dist %.2e, U(T)=%.4f"
          % (key, d.max(), t[kmax], d[-1], U[-1]))
    if key == "a":
        ap.legend(loc="upper left", fontsize=6.3, handlelength=1.4, borderaxespad=0.1, labelspacing=0.2)
# (d) OU without uniform rate: ||S(t)h_y - psi||_1 for shifted Gaussians
ax = fig.add_subplot(outer[1, 1])
sig = np.sqrt(Dou / th)
tt = np.linspace(0, 12, 1201)
cols = [C_LIGHT, C_MED, "#1f5aa6", C_DARK]
for k, yk in enumerate([1, 10, 100, 1000]):
    mshift = yk * sig * np.exp(-th * tt)
    ax.semilogy(tt, 2 * erf(mshift / (2 * sig * np.sqrt(2))), color=cols[k],
                label=r"$y=%d\,\sigma$" % yk)
ax.set_ylim(1e-4, 9); ax.set_xlim(0, 12)
for k in range(3):
    t0 = np.log(10) * (k + 1) / th
    ax.annotate("", xy=(t0 + 0.6, 2.9), xytext=(t0 - np.log(10) / th + 0.6, 2.9),
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.7))
ax.text(4.1, 4.0, r"shift $\ln 10/\theta$ per decade of $y$", fontsize=6.8, color=C_GREY,
        ha="center")
ax.set_xlabel(r"time $t$")
ax.set_ylabel(r"$\|S(t)h_y-\psi\|_{L^1}$")
ax.set_title(r"(d) OU: no uniform rate", loc="left")
ax.legend(loc="lower left", fontsize=6.5, handlelength=1.6)
fig.savefig("fig3_examples.pdf", bbox_inches="tight", dpi=300); plt.close(fig)
