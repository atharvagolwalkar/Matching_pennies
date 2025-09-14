# matching_pennies_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

st.set_page_config(page_title="Matching Pennies Simulator", layout="wide")

# -------------------------
# Helper math functions
# -------------------------
def effective_prob_intended_to_actual(p_intend, alpha):
    """
    Given intended probability p_intend of playing H (Heads),
    and 'reliability' alpha (probability the intended action remains the same),
    return effective prob of actual H.
    alpha in [0,1]. If alpha = 1 => no noise. If alpha = 0.5 => fully random independent of intent.
    Formula:
        p_eff = p_intend*alpha + (1-p_intend)*(1-alpha)
              = (2*alpha -1)*p_intend + (1-alpha)
    """
    return (2 * alpha - 1) * p_intend + (1 - alpha)

def expected_payoff_A(p_eff, q_eff):
    """
    Payoff to Player A (Matching Pennies): +1 if actions match, -1 otherwise.
    Given effective probabilities p_eff (A plays H) and q_eff (B plays H).
    E = P(match) - P(mismatch) = 2*P(match) - 1
    P(match) = p_eff*q_eff + (1-p_eff)*(1-q_eff)
    """
    p = p_eff
    q = q_eff
    p_match = p * q + (1 - p) * (1 - q)
    return 2 * p_match - 1

def payoff_matrix_table(alpha=1.0, beta=1.0):
    """
    Build the payoff matrix for intended actions H/T x H/T
    But consider noise alpha (A) and beta (B) by assuming:
    - If A intends H, actual p_eff = alpha (since p_intend=1)
    - If A intends T, actual p_eff = 1-alpha (since p_intend=0)
    likewise for B with beta.
    Returns matrix of shape (2,2) with payoff to A.
    Rows: A intends H (0) or T (1)
    Cols: B intends H (0) or T (1)
    """
    mat = np.zeros((2,2))
    for i, a_intend in enumerate([1, 0]):  # 1 => H intended, 0 => T intended
        p_eff = a_intend * alpha + (1 - a_intend) * (1 - alpha)
        for j, b_intend in enumerate([1, 0]):
            q_eff = b_intend * beta + (1 - b_intend) * (1 - beta)
            mat[i, j] = expected_payoff_A(p_eff, q_eff)
    return mat

# Solve for mixed strategy equilibrium under noisy flips (alpha, beta)
def find_mixed_equilibrium(alpha=1.0, beta=1.0, tol=1e-6):
    """
    Solve for intended mixed strategies (p for A intends Heads, q for B intends Heads)
    such that each player is indifferent between their two intended actions.
    That implies:
        U_A(H | q) = U_A(T | q)  -> equation in q only -> solve for q*
        U_B(H | p) = U_B(T | p)  -> equation in p only -> solve for p*
    We solve each with a 1D root-finding on q and p in [0,1].
    If the equation has no sign change (rare), fallback to 0.5.
    """
    # Define U_A(H|q) and U_A(T|q)
    def diff_A(q):
        # A intends H -> p_eff = alpha
        q_eff_h = effective_prob_intended_to_actual(q, beta)
        u_h = expected_payoff_A(alpha, q_eff_h)
        # A intends T -> p_eff = 1-alpha
        u_t = expected_payoff_A(1 - alpha, q_eff_h)
        return u_h - u_t

    def diff_B(p):
        # B intends H -> q_eff = beta
        p_eff_h = effective_prob_intended_to_actual(p, alpha)
        # B payoff is negative of A's payoff (zero-sum), but indifference eqn is same structure:
        # U_B(H) = -U_A(... when B intends H). We'll compute B's payoff directly:
        # But easiest: compute U_B(H) - U_B(T) = -(U_A given B intends H) + (U_A given B intends T)
        # So diff_B = U_B(H) - U_B(T) = -U_A(p_eff_h, beta) + U_A(p_eff_h, 1-beta)
        u_b_h = -expected_payoff_A(p_eff_h, beta)
        u_b_t = -expected_payoff_A(p_eff_h, 1 - beta)
        return u_b_h - u_b_t

    # Solve via bisection on [0,1]
    def solve_1d(f):
        a, b = 0.0, 1.0
        fa, fb = f(a), f(b)
        if abs(fa) < tol:
            return a
        if abs(fb) < tol:
            return b
        # If no sign change, fallback to 0.5
        if fa * fb > 0:
            # Try midpoint as fallback, but better fallback - choose 0.5
            return 0.5
        for _ in range(60):
            m = 0.5 * (a + b)
            fm = f(m)
            if abs(fm) < tol:
                return float(np.clip(m, 0, 1))
            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
        return float(np.clip(0.5 * (a + b), 0, 1))

    q_star = solve_1d(diff_A)
    p_star = solve_1d(diff_B)
    return p_star, q_star

# -------------------------
# AI opponents
# -------------------------
def ai_random_equilibrium(p_eq, q_eq, for_player='B'):
    # If AI is player B, it uses q_eq. If AI is player A (rare), uses p_eq.
    if for_player == 'B':
        q = q_eq
        return 'H' if np.random.rand() < q else 'T'
    else:
        p = p_eq
        return 'H' if np.random.rand() < p else 'T'

def ai_frequency_opponent(history, for_player='B'):
    """
    If AI is B, it inspects history of A's actual choices (H/T as strings)
    and plays the opposite choice that would win more often.
    We'll implement: estimate prob of A playing H and best response for B is:
       if P(A plays H) > 0.5 -> choose H (to match) ??? Wait, in matching pennies,
       B wins when mismatch. So best response is to play opposite of A's most probable actual action.
    """
    if len(history) == 0:
        return 'H' if np.random.rand() < 0.5 else 'T'
    # Get freq of A_actual
    a_actuals = [h[0] for h in history]  # tuple stored as (A_actual, B_actual, winner)
    p_H = a_actuals.count('H') / len(a_actuals)
    # B wants mismatch -> play opposite of the most likely A action
    if p_H > 0.5:
        return 'T'
    elif p_H < 0.5:
        return 'H'
    else:
        return 'H' if np.random.rand() < 0.5 else 'T'

def ai_markov_opponent(history, for_player='B'):
    """
    1-step Markov: look at last A actual move and use frequencies of A_next | A_last.
    If nothing to go on, random.
    """
    if len(history) < 2:
        return 'H' if np.random.rand() < 0.5 else 'T'
    # Build transition counts for A actual moves
    trans = {'H': {'H': 0, 'T': 0}, 'T': {'H': 0, 'T': 0}}
    a_actuals = [h[0] for h in history]
    for i in range(len(a_actuals) - 1):
        prev = a_actuals[i]
        nxt = a_actuals[i + 1]
        trans[prev][nxt] += 1
    last = a_actuals[-1]
    total = trans[last]['H'] + trans[last]['T']
    if total == 0:
        prob_next_H = 0.5
    else:
        prob_next_H = trans[last]['H'] / total
    # B chooses opposite of predicted next A action to cause mismatch
    if prob_next_H > 0.5:
        return 'T'
    elif prob_next_H < 0.5:
        return 'H'
    else:
        return 'H' if np.random.rand() < 0.5 else 'T'

# -------------------------
# Utility: apply noise to an intended choice
# -------------------------
def actual_from_intended(intended_choice, reliability):
    """Given intended 'H' or 'T' and reliability alpha, return actual action."""
    if np.random.rand() < reliability:
        return intended_choice
    else:
        return 'H' if intended_choice == 'T' else 'T'

# -------------------------
# Session state initialization
# -------------------------
if 'history' not in st.session_state:
    # store tuples: (A_intended, A_actual, B_intended, B_actual, winner (A/B/Tie))
    st.session_state.history = []
if 'scores' not in st.session_state:
    st.session_state.scores = {'A': 0, 'B': 0, 'Rounds': 0}

# -------------------------
# Layout
# -------------------------
st.title("🎯 Matching Pennies Simulator — Interactive + Theory")
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Game Settings")
    mode = st.selectbox("Mode", ["Human vs Human", "Human vs Computer"])
    ai_type = None
    if mode == "Human vs Computer":
        ai_type = st.selectbox("AI strategy", ["Random (play equilibrium)", "Frequency (exploit)", "1-step Markov"])
    st.markdown("**Noise / Biased coin** (optional):\n- Reliability = probability the intended action stays the same.\n- 1.0 => no noise; 0.5 => totally unrelated to intent (random).")
    alpha = st.slider("Player A reliability (alpha)", 0.5, 1.0, 1.0, 0.01)
    beta = st.slider("Player B reliability (beta)", 0.5, 1.0, 1.0, 0.01)
    st.markdown("---")
    st.write("Controls:")
    if mode == "Human vs Human":
        st.info("Both players will press their chosen buttons below each round.")
    else:
        st.info("You are Player A (left). AI is Player B (right).")

    if st.button("Reset Game"):
        st.session_state.history = []
        st.session_state.scores = {'A': 0, 'B': 0, 'Rounds': 0}
        st.experimental_rerun()

with col2:
    st.header("Play Round")
    # Two columns for player choices
    pcol1, pcol2 = st.columns(2)
    # DEFAULT intents
    A_intent = None
    B_intent = None

    if mode == "Human vs Human":
        with pcol1:
            st.subheader("Player A")
            A_choice = st.radio("Choose (Player A)", ['H', 'T'], key="A_choice")
        with pcol2:
            st.subheader("Player B")
            B_choice = st.radio("Choose (Player B)", ['H', 'T'], key="B_choice")
        if st.button("Play Round (Both)"):
            A_intent = A_choice
            B_intent = B_choice
    else:
        # Human vs AI: Human chooses, AI chooses automatically
        with pcol1:
            st.subheader("You — Player A")
            A_choice = st.radio("Choose (You)", ['H', 'T'], key="A_choice_human")
            if st.button("Play Round (You vs AI)"):
                A_intent = A_choice
                # determine AI intended action based on strategy
                # First compute equilibrium (for AI Random)
                p_eq, q_eq = find_mixed_equilibrium(alpha=alpha, beta=beta)
                # Use historical actuals to feed smarter AIs
                history_for_ai = [(h[1], h[3], h[4]) for h in st.session_state.history]  # (A_actual, B_actual, winner)
                # transform to our simpler history format for AI funcs: list of (A_actual, B_actual, winner)
                hist_simplified = [(h[1], h[3], h[4]) for h in st.session_state.history]
                if ai_type == "Random (play equilibrium)":
                    B_intent = 'H' if np.random.rand() < q_eq else 'T'
                elif ai_type == "Frequency (exploit)":
                    B_intent = ai_frequency_opponent(hist_simplified, for_player='B')
                elif ai_type == "1-step Markov":
                    B_intent = ai_markov_opponent(hist_simplified, for_player='B')
                else:
                    B_intent = 'H' if np.random.rand() < 0.5 else 'T'

    # If a round is triggered (A_intent not None and B_intent not None)
    if A_intent is not None and B_intent is not None:
        # Determine actual actions after noise
        A_actual = actual_from_intended(A_intent, alpha)
        B_actual = actual_from_intended(B_intent, beta)

        # Determine winner
        if A_actual == B_actual:
            winner = 'A'
            st.session_state.scores['A'] += 1
        else:
            winner = 'B'
            st.session_state.scores['B'] += 1
        st.session_state.scores['Rounds'] += 1

        # Save to history (store both intended and actual for clarity)
        st.session_state.history.append((A_intent, A_actual, B_intent, B_actual, winner))

        st.success(f"Round played — Actual: A={A_actual}, B={B_actual}. Winner: {winner}")

# -------------------------
# Display payoff matrix and theoretical equilibrium
# -------------------------
st.markdown("---")
st.header("Theory: Payoff Matrix & Nash Equilibrium (with noise)")

p_eq, q_eq = find_mixed_equilibrium(alpha=alpha, beta=beta)
col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("Payoff matrix (to Player A)")
    mat = payoff_matrix_table(alpha=alpha, beta=beta)
    df_mat = pd.DataFrame(mat, index=["A intend H", "A intend T"], columns=["B intend H", "B intend T"])
    st.table(df_mat.style.format("{:.2f}"))

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    cmap = plt.cm.viridis
    norm = colors.TwoSlopeNorm(vmin=mat.min(), vcenter=0.0, vmax=mat.max())
    im = ax.imshow(mat, cmap=cmap, norm=norm)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['B intend H','B intend T'])
    ax.set_yticks([0,1])
    ax.set_yticklabels(['A intend H','A intend T'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i,j]:+.2f}", ha='center', va='center', color='white', fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

with col4:
    st.subheader("Mixed-strategy Nash equilibrium (intended probabilities)")
    st.markdown(f"- Player A intends **Heads** with probability **p\*** = `{p_eq:.3f}`")
    st.markdown(f"- Player B intends **Heads** with probability **q\*** = `{q_eq:.3f}`")
    st.markdown("Interpretation:")
    st.markdown(" - With perfect reliability (alpha=beta=1), p* = q* = 0.5 (the classic result).")
    st.markdown(" - With noisy actions, the equilibrium shifts to keep opponents indifferent.")

# -------------------------
# History, stats, and visualizations
# -------------------------
st.markdown("---")
st.header("Game History & Empirical Statistics")

hist = st.session_state.history
if len(hist) == 0:
    st.info("No rounds played yet. Play a round to see results and statistics.")
else:
    # Build a DataFrame showing the rounds
    df = pd.DataFrame(hist, columns=["A_intended","A_actual","B_intended","B_actual","Winner"])
    df.index = np.arange(1, len(df)+1)
    st.subheader("Round-by-round")
    st.dataframe(df)

    # Empirical frequencies and win rates
    total = len(df)
    A_actual_counts = df['A_actual'].value_counts().reindex(['H','T']).fillna(0)
    B_actual_counts = df['B_actual'].value_counts().reindex(['H','T']).fillna(0)
    wins_A = (df['Winner'] == 'A').sum()
    wins_B = (df['Winner'] == 'B').sum()
    st.write(f"Rounds played: **{total}** — Wins: **A={wins_A}**, **B={wins_B}**")

    # Frequency bar chart
    fig2, ax2 = plt.subplots(figsize=(5,3))
    bar_width = 0.35
    indices = np.arange(2)
    ax2.bar(indices - bar_width/2, A_actual_counts.values/total, bar_width, label='A actual freq')
    ax2.bar(indices + bar_width/2, B_actual_counts.values/total, bar_width, label='B actual freq')
    ax2.set_xticks(indices)
    ax2.set_xticklabels(['H','T'])
    ax2.set_ylabel('Empirical frequency')
    ax2.set_title('Actual action frequencies (empirical)')
    ax2.legend()
    st.pyplot(fig2)

    # Running cumulative win rate plot for A
    cumulative_A = df['Winner'].eq('A').cumsum() / np.arange(1, total+1)
    fig3, ax3 = plt.subplots(figsize=(6,2.5))
    ax3.plot(np.arange(1, total+1), cumulative_A, marker='o')
    ax3.axhline(0.5, linestyle='--', label='50% baseline')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Cumulative win rate of A')
    ax3.set_ylim(0,1)
    ax3.set_title('Cumulative win rate (Player A)')
    ax3.legend()
    st.pyplot(fig3)

# -------------------------
# Show final notes and tips
# -------------------------
st.markdown("---")
st.header("Notes & Extension Ideas")
st.markdown("""
- The app models *noisy actions* via reliability parameters `alpha` (A) and `beta` (B).
- The Nash equilibrium displayed are **intended** mixed strategies (i.e., the probabilities with which players choose Heads or Tails before noise).
- **Extensions you can add**:
  - Let AI learn the opponent's intended **mixed** strategy directly (Bayesian update).
  - Add option to change payoff values (instead of +1/-1) to study different zero-sum variants.
  - Add multi-round tournaments with multiple AIs and visualize which strategy wins overall.
  - Export history to CSV for report/analysis (`st.download_button`).
""")
st.caption("Built for a student project: interactive, theoretical, and extendable. Good luck! 🎓")
