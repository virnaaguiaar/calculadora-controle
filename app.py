"""
Calculadora de Controle: projeto de PD, PI, PID e a/(s+b) pelo lugar das raízes,
com resposta ao degrau e discretização (Tustin, Euler, ZOH).

Rodar localmente:  streamlit run app.py
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from controle import (discretizar, equacao_diferencas, ler_polinomio, malha_fechada,
                      metricas, poli_para_texto, polo_desejado, projetar,
                      resposta_degrau, tabela_angulos, zeta_de_mp, zeta_wn_minimo)

st.set_page_config(page_title="Calculadora de Controle", page_icon="📐", layout="wide")


# ----------------------------------------------------------------------
# utilidades de exibição
# ----------------------------------------------------------------------
def latex_poli(p, var='s'):
    txt = poli_para_texto(p, var)
    return re.sub(r'\^(\d+)', r'^{\1}', txt)


def latex_fracao(num, den, var='s'):
    return r'\dfrac{' + latex_poli(num, var) + '}{' + latex_poli(den, var) + '}'


def fmt_c(z, casas=4):
    z = complex(z)
    if abs(z.imag) < 1e-9:
        return f'{z.real:.{casas}f}'
    sinal = '+' if z.imag >= 0 else '−'
    return f'{z.real:.{casas}f} {sinal} j{abs(z.imag):.{casas}f}'


METODOS = {
    "Tustin (bilinear)": "tustin",
    "Euler de avanço (forward): s = (z−1)/T": "euler_avanco",
    "Euler de atraso (backward): s = (z−1)/(Tz)": "euler_atraso",
    "ZOH (segurador de ordem zero)": "zoh",
}


def bloco_discretizacao(num, den, chave, rotulo='C', mostrar_eq=True):
    c1, c2 = st.columns(2)
    T = c1.number_input("Período de amostragem T (s)", min_value=1e-6, value=1.0,
                        step=0.1, format="%.4f", key=f"T_{chave}")
    metodo_nome = c2.selectbox("Método", list(METODOS), key=f"m_{chave}")
    try:
        nz, dz = discretizar(num, den, T, METODOS[metodo_nome])
    except ValueError as e:
        st.error(str(e))
        return
    st.latex(rf'{rotulo}(z) = ' + latex_fracao(nz, dz, 'z'))
    polos_z = np.roots(dz.coeffs) if dz.order > 0 else []
    zeros_z = np.roots(nz.coeffs) if nz.order > 0 else []
    c1, c2 = st.columns(2)
    c1.markdown("**Polos em z:** " + (", ".join(fmt_c(p) for p in polos_z) or "—"))
    c2.markdown("**Zeros em z:** " + (", ".join(fmt_c(z) for z in zeros_z) or "—"))
    fora = [p for p in polos_z if abs(p) > 1 + 1e-9]
    if fora:
        st.warning("Há polos fora do círculo unitário (|z| > 1). "
                   "Com esse T e esse método, a versão discreta é instável.")
    if mostrar_eq:
        st.markdown("**Equação a diferenças:**")
        st.code(equacao_diferencas(nz, dz), language=None)


# ----------------------------------------------------------------------
# barra lateral
# ----------------------------------------------------------------------
st.sidebar.title("📐 Calculadora de Controle")
modo = st.sidebar.radio("O que você quer fazer?",
                        ["Projetar controlador", "Só discretizar uma função"])

with st.sidebar.expander("Como digitar os polinômios"):
    st.markdown(
        "- Forma fatorada: `5(s+3)`, `s(s+4)`, `(s+1)^2`\n"
        "- Forma expandida: `s^3 + 4s^2 + 4s`\n"
        "- Coeficientes separados por espaço: `1 4 4 0`\n"
        "- Constante: `1` ou `0,4`\n"
        "- Vírgula decimal é aceita: `1,1772`"
    )

# ======================================================================
# MODO 2: só discretizar
# ======================================================================
if modo == "Só discretizar uma função":
    st.title("Discretização de F(s)")
    c1, c2 = st.columns(2)
    try:
        num = ler_polinomio(c1.text_input("Numerador de F(s)", "1"))
        den = ler_polinomio(c2.text_input("Denominador de F(s)", "s(s+1)"))
    except Exception as e:
        st.error(f"Erro ao ler o polinômio: {e}")
        st.stop()
    st.latex(r'F(s) = ' + latex_fracao(num, den))
    bloco_discretizacao(num, den, "solo", rotulo='F')
    st.stop()

# ======================================================================
# MODO 1: projeto
# ======================================================================
st.title("Projeto pelo lugar das raízes")

# ---------- planta ----------
st.header("1. Planta e realimentação")
c1, c2, c3, c4 = st.columns(4)
try:
    N_G = ler_polinomio(c1.text_input("Numerador de G(s)", "4(s+4)"))
    D_G = ler_polinomio(c2.text_input("Denominador de G(s)", "s^3 + 4s^2 + 4s"))
    N_H = ler_polinomio(c3.text_input("Numerador de H(s)", "1"))
    D_H = ler_polinomio(c4.text_input("Denominador de H(s)", "1"))
except Exception as e:
    st.error(f"Erro ao ler o polinômio: {e}")
    st.stop()

c1, c2 = st.columns(2)
c1.latex(r'G(s) = ' + latex_fracao(N_G, D_G))
c2.latex(r'H(s) = ' + latex_fracao(N_H, D_H))
N_GH, D_GH = N_G * N_H, D_G * D_H

# ---------- controlador ----------
st.header("2. Controlador")
tipos = {
    "PD  →  Kp + Kd·s": "PD",
    "PI  →  Kp + Ki/s": "PI",
    "PID com zeros reais e iguais  →  Kd(s − z)²/s": "PID",
    "Polo puro  →  a/(s + b)": "POLO",
}
tipo = tipos[st.radio("Tipo", list(tipos), horizontal=True)]

# ---------- especificação ----------
st.header("3. Especificação")
espec = st.radio("Como a questão dá o polo desejado?",
                 ["Mp (%) e tempo de acomodação", "ξ e ωn", "Polo s_d direto"],
                 horizontal=True)

mp_alvo = ts_alvo = criterio = None
if espec == "Mp (%) e tempo de acomodação":
    c1, c2, c3 = st.columns(3)
    mp_alvo = c1.number_input("Mp máximo (%)", min_value=0.01, max_value=99.0, value=10.0, step=1.0)
    ts_alvo = c2.number_input("Tempo de acomodação máximo (s)", min_value=0.001, value=4.0, step=0.5)
    criterio = c3.selectbox("Critério do ts", [5, 2], format_func=lambda x: f"{x}%  (ts = {3 if x == 5 else 4}/ξωn)")

    zeta_exato = zeta_de_mp(mp_alvo)
    arred = st.checkbox("Arredondar ξ para cima com 1 casa decimal (como no caderno)", value=True)
    zeta = np.ceil(zeta_exato * 10 - 1e-9) / 10 if arred else zeta_exato
    zwn = zeta_wn_minimo(ts_alvo, criterio)
    wn_min = zwn / zeta

    c1, c2, c3 = st.columns(3)
    c1.metric("ξ calculado", f"{zeta_exato:.4f}")
    c2.metric("ξ usado", f"{zeta:.4f}")
    c3.metric("ωn mínimo", f"{wn_min:.4f} rad/s")
    wn = st.number_input("ωn de projeto (rad/s). Dica: use um valor acima do mínimo para ter folga.",
                         min_value=1e-6, value=float(np.round(wn_min, 4)), step=0.1, format="%.4f")
    if wn < wn_min - 1e-9:
        st.warning("ωn abaixo do mínimo: o tempo de acomodação estimado não atende.")
    s_d = polo_desejado(zeta, wn)

elif espec == "ξ e ωn":
    c1, c2 = st.columns(2)
    zeta = c1.number_input("ξ", min_value=0.001, max_value=0.999, value=0.7, step=0.05, format="%.4f")
    wn = c2.number_input("ωn (rad/s)", min_value=1e-6, value=0.5, step=0.1, format="%.4f")
    s_d = polo_desejado(zeta, wn)

else:
    c1, c2 = st.columns(2)
    re_sd = c1.number_input("Parte real de s_d", value=-4.0, step=0.5, format="%.4f")
    im_sd = c2.number_input("Parte imaginária de s_d (positiva)", min_value=1e-6, value=4.0,
                            step=0.5, format="%.4f")
    s_d = complex(re_sd, im_sd)

st.info(f"Polo desejado:  s_d = {fmt_c(s_d)}")

# ---------- cálculo ----------
try:
    r = projetar(tipo, N_GH, D_GH, s_d)
except Exception as e:
    st.error(f"Não foi possível projetar: {e}")
    st.stop()

st.header("4. Critério de ângulo")
st.dataframe(pd.DataFrame(tabela_angulos(s_d, r["polos"], r["zeros"])),
             hide_index=True)
if tipo in ('PI', 'PID'):
    st.caption("O polo em 0 é o integrador do controlador e já está incluído.")

c1, c2, c3 = st.columns(3)
c1.metric("Σ ângulos dos polos", f'{r["soma_polos"]:.3f}°')
c2.metric("Σ ângulos dos zeros", f'{r["soma_zeros"]:.3f}°')
c3.metric("φ = Σpolos − Σzeros", f'{r["phi"]:.3f}°')

if tipo in ('PD', 'PI'):
    st.markdown(f"O zero do controlador deve contribuir com **θ = φ − 180° = {r['theta']:.3f}°** "
                f"e fica em **s = {r['z_c']:.4f}**.")
elif tipo == 'PID':
    st.markdown(f"Os dois zeros juntos contribuem com **φ − 180° = {r['theta_total']:.3f}°**, "
                f"ou seja, **{r['theta']:.3f}° cada**. Zeros duplos em **s = {r['z_c']:.4f}**.")
else:
    st.markdown(f"O polo do controlador deve contribuir com **θ = 180° − φ = {r['theta']:.3f}°** "
                f"e fica em **s = {r['p_c']:.4f}**.")

st.header("5. Critério de módulo e resultado")
st.markdown(f"|G(s_d)H(s_d)| {'/ |s_d| ' if tipo in ('PI', 'PID') else ''}= **{r['modulo_GH']:.6f}**")

if tipo == 'PD':
    st.latex(rf"K_d = \frac{{1}}{{|s_d - z_c|\,|GH|}} = {r['Kd']:.4f} \qquad K_p = -z_c K_d = {r['Kp']:.4f}")
    st.latex(rf"C(s) = {r['Kp']:.4f} + {r['Kd']:.4f}\,s")
elif tipo == 'PI':
    st.latex(rf"K_p = \frac{{|s_d|}}{{|s_d - z_c|\,|GH|}} = {r['Kp']:.4f} \qquad K_i = -z_c K_p = {r['Ki']:.4f}")
    st.latex(rf"C(s) = {r['Kp']:.4f} + \frac{{{r['Ki']:.4f}}}{{s}}")
elif tipo == 'PID':
    st.latex(rf"K_d = {r['Kd']:.4f} \qquad K_p = -2 z_c K_d = {r['Kp']:.4f} \qquad K_i = z_c^2 K_d = {r['Ki']:.4f}")
    st.latex(rf"C(s) = {r['Kp']:.4f} + \frac{{{r['Ki']:.4f}}}{{s}} + {r['Kd']:.4f}\,s")
else:
    st.latex(rf"a = \frac{{|s_d + b|}}{{|GH|}} = {r['a']:.4f} \qquad b = {r['b']:.4f}")
    st.latex(rf"G_c(s) = \frac{{{r['a']:.4f}}}{{s + {r['b']:.4f}}}")

# ---------- malha fechada ----------
st.header("6. Verificação em malha fechada")
num_mf, den_mf = malha_fechada(r["C_num"], r["C_den"], N_G, D_G, N_H, D_H)
polos_mf = np.roots(den_mf.coeffs)
st.markdown("**Polos de malha fechada:** " + ", ".join(fmt_c(p) for p in polos_mf))

resp = resposta_degrau(num_mf, den_mf)
if resp is None:
    st.error("Malha fechada instável: não há resposta ao degrau para mostrar.")
else:
    t, y = resp
    m = metricas(t, y)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor final", f"{m['yf']:.4f}")
    c2.metric("Mp simulado", f"{m['Mp']:.2f}%",
              delta=None if mp_alvo is None else ("atende" if m['Mp'] <= mp_alvo else "NÃO atende"),
              delta_color="normal" if mp_alvo is None or m['Mp'] <= mp_alvo else "inverse")
    c3.metric("ts (5%)", f"{m['ts5']:.3f} s",
              delta=None if criterio != 5 else ("atende" if m['ts5'] < ts_alvo else "NÃO atende"),
              delta_color="normal" if criterio != 5 or m['ts5'] < ts_alvo else "inverse")
    c4.metric("ts (2%)", f"{m['ts2']:.3f} s",
              delta=None if criterio != 2 else ("atende" if m['ts2'] < ts_alvo else "NÃO atende"),
              delta_color="normal" if criterio != 2 or m['ts2'] < ts_alvo else "inverse")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, y, lw=2, label="y(t)")
    ax.axhline(m['yf'], color='gray', ls='--', lw=1, label="valor final")
    pct = criterio or 5
    ax.axhspan(m['yf'] * (1 - pct / 100), m['yf'] * (1 + pct / 100), color='green', alpha=0.08,
               label=f"faixa de ±{pct}%")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("saída")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    st.pyplot(fig)
    if num_mf.order >= den_mf.order:
        st.warning("T(s) tem numerador e denominador de mesmo grau: a saída salta em t = 0 "
                   "e o Mp simulado inclui esse salto.")
    st.caption("A simulação considera todos os polos e zeros de malha fechada. Se ela não atender "
               "à especificação, aumente o ωn de projeto: o método supõe que o par s_d domina.")

# ---------- discretização ----------
st.header("7. Discretização (opcional)")
alvo = st.radio("O que discretizar?",
                ["Nada", "Só o controlador C(s)", "G(s)·H(s)·C(s) (malha aberta completa)"],
                horizontal=True)
if alvo == "Só o controlador C(s)":
    st.latex(r'C(s) = ' + latex_fracao(r["C_num"], r["C_den"]))
    bloco_discretizacao(r["C_num"], r["C_den"], "ctrl", rotulo='C')
elif alvo.startswith("G(s)"):
    L_num, L_den = N_GH * r["C_num"], D_GH * r["C_den"]
    st.latex(r'G(s)H(s)C(s) = ' + latex_fracao(L_num, L_den))
    bloco_discretizacao(L_num, L_den, "malha", rotulo='GHC', mostrar_eq=False)
