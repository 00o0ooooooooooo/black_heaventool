
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy

# ===================== Constants (fixed model) =====================
BIG_PAY_DEFAULT = 210
REG_PAY_DEFAULT = 90
BIG_G_DEFAULT   = 59
REG_G_DEFAULT   = 24
COIN_PER_G_DEFAULT = 3.0
HEAVEN_AVG_GAIN_PER_ROUND_DEFAULT = 120

APP_TITLE = "沖ドキ！BLACK 押し引き v0.46（固定モデル＋スマホUI）"

# ===================== Page setup & CSS =====================
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(f"## {APP_TITLE}")
st.caption("学習は一度きりの固定モデル。ホールでは「履歴＋現在ハマりG＋（任意で区切り線）」だけを入れればOK。")

st.markdown("""
<style>
:root { --btn-h: 52px; --btn-fs: 18px; }
.stButton>button { height: var(--btn-h); font-size: var(--btn-fs); border-radius: 12px; }
.stNumberInput input { font-size: 18px; }
[data-testid="stMetricDelta"] { font-size: 14px !important; }

/* Container bottom padding to avoid sticky keypad overlap */
.block-container { padding-bottom: 260px; }

/* Sticky keypad footer */
.sticky-pad {
  position: fixed;
  left: 0; right: 0; bottom: 0;
  background: rgba(24, 24, 28, .98);
  border-top: 1px solid rgba(255,255,255,.08);
  padding: .6rem .8rem 1.0rem;
  z-index: 999;
}
.pad-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; }
.pad-nums { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; }
.pad-actions { display: grid; grid-template-columns: repeat(2, 1fr); gap: .5rem; }

/* Mobile tweaks */
@media (max-width: 880px) {
  :root { --btn-h: 56px; --btn-fs: 18px; }
  .pad-grid { grid-template-columns: 1fr 1fr; }
  .block-container { padding-bottom: 300px; }
}

/* Badges & small text */
.badge-warn { background:#5b1e1e; color:#ffb3b3; padding:4px 8px; border-radius:10px; font-weight:600; display:inline-block; }
.small-note { color:#aaa; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ===================== Model =====================
def load_model_table():
    here = Path(__file__).resolve().parent
    for p in [here/"model_bins_v1.csv", Path.cwd()/"model_bins_v1.csv"]:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

def bin_short_pct(p):
    if p < 40: return "<40"
    if p < 50: return "40-50"
    if p < 60: return "50-60"
    if p < 65: return "60-65"
    if p < 70: return "65-70"
    if p < 80: return "70-80"
    return "80+"

def bin_advg(a):
    if a <= 800: return "<=800"
    if a <= 1200: return "800-1200"
    if a <= 1600: return "1200-1600"
    if a <= 1800: return "1600-1800"
    if a <= 2200: return "1800-2200"
    return ">2200"

def regional_adjust(cum_adv, short_pct_model, model65, model50, pending_g, flags):
    dP = dE = dLe2 = 0.0
    # 1600帯（65%換算で説明可能なよう model65 を使う）
    if 1580 <= cum_adv <= 1680 and short_pct_model >= model65: dP -= 3.0
    if 1550 <= cum_adv <= 1800 and short_pct_model < model50:  dP += 2.0
    if 1720 <= cum_adv <= 1820:                                dP += 1.0
    # 3500帯（据え置き）
    if 3300 <= cum_adv <= 3450: dP -= 2.0; dE -= 0.3; dLe2 += 2.0
    if 3450 <= cum_adv <= 3550: dP += 1.0; dE += 1.0; dLe2 -= 5.0; 
    if 3450 <= cum_adv <= 3550 and short_pct_model < model50:  dE += 0.5
    if 3550 <= cum_adv <= 3700: dP += 1.0; dE -= 1.0; dLe2 += 4.0
    # 現在ハマり
    if pending_g >= 540: dP += 2.0
    elif pending_g >= 230: dP += 1.0
    # 例外
    if flags.get("post_black", False): dP -= 3.0
    if flags.get("huge_reg", False):   dP -= 2.0
    return dP, dE, dLe2

def clip_pct(x, lo=5.0, hi=85.0): return float(max(lo, min(hi, x)))
def clip01(x): return float(max(0.0, min(100.0, x)))

def compute_segment_stats(seg_df, pending_g, params):
    """空区間でも現在ハマりだけで計算する"""
    big_pay, reg_pay = params["big_pay"], params["reg_pay"]
    big_g, reg_g = params["big_g"], params["reg_g"]
    coin_per_g = params["coin_per_g"]
    if seg_df is None or len(seg_df)==0:
        cum_adv = float(pending_g)
        coin_in = float(pending_g) * coin_per_g
        payout  = 0.0
        short_pct_model = 0.0 if coin_in>0 else np.nan
        return {"cum_adv":cum_adv, "coin_in":coin_in, "payout":payout, "short_pct_model":short_pct_model}

    df = seg_df.copy()
    df["bonus_g"] = np.where(df["Type"]=="BIG", big_g, reg_g)
    df["payout"]  = np.where(df["Type"]=="BIG", big_pay, reg_pay)
    df["usual_g"] = pd.to_numeric(df["IntervalG"], errors="coerce").fillna(0).astype(float)
    df["adv_g_inc"] = df["usual_g"] + df["bonus_g"]
    cum_adv = df["adv_g_inc"].sum() + pending_g
    coin_in = (df["usual_g"].sum() + pending_g) * coin_per_g
    payout  = df["payout"].sum()
    short_pct_model = (payout/coin_in*100.0) if coin_in>0 else np.nan
    return {"cum_adv":cum_adv, "coin_in":coin_in, "payout":payout, "short_pct_model":short_pct_model}

def lookup_model(bin_tbl, short_pct_model, cum_adv):
    b1, b2 = bin_short_pct(short_pct_model), bin_advg(cum_adv)
    row = bin_tbl[(bin_tbl["bin_pct"]==b1) & (bin_tbl["bin_adv"]==b2)]
    if len(row)==0: return None, b1, b2
    r = row.iloc[0]
    return {"p_trig": float(r["p_trig_sm"])*100.0, "E_len": float(r["E_len_sm"]), "p_le2": float(r["p_le2_sm"])*100.0, "n": int(r["n"]), "n_pos": int(r["n_pos"]) if not pd.isna(r["n_pos"]) else 0}, b1, b2

def decide_action(p, ev, ple2, mode="標準", loss_tol=100):
    if mode=="保守":
        if (ev>0) and (p>=40.0) and (ple2<=45.0): return "押す"
        return "様子見" if abs(ev)<=20 else "引く"
    if mode=="攻め":
        if (ev>-abs(loss_tol)) and (p>=28.0): return "押す"
        return "様子見" if abs(ev)<=20 else "引く"
    if (ev>0) and (p>=32.0): return "押す"
    return "様子見" if abs(ev)<=20 else "引く"

# ===================== Sidebar =====================
bin_tbl = load_model_table()
if bin_tbl is None:
    st.error("model_bins_v1.csv が見つかりません。同じフォルダに置いてください。")

with st.sidebar:
    mode = st.radio("判定モード", ["保守","標準","攻め"], index=1, horizontal=True)
    base_per_50 = st.number_input("ベース（50枚で回るG）", min_value=25, max_value=40, value=32, step=1)
    with st.expander("上級設定（普段は不要）", expanded=False):
        big_pay = st.number_input("BIG平均枚数", 150, 280, BIG_PAY_DEFAULT, 5)
        reg_pay = st.number_input("REG平均枚数", 50, 150, REG_PAY_DEFAULT, 5)
        big_g   = st.number_input("BIG中の有利G (+)", 40, 80, BIG_G_DEFAULT, 1)
        reg_g   = st.number_input("REG中の有利G (+)", 15, 40, REG_G_DEFAULT, 1)
        coin_per_g = st.number_input("通常時の投入（枚/G）", 1.0, 5.0, COIN_PER_G_DEFAULT, 0.5)
        heaven_gain = st.number_input("天国1回あたり平均差枚（固定）", 60, 200, HEAVEN_AVG_GAIN_PER_ROUND_DEFAULT, 5)
        loss_tol = st.number_input("攻め：許容損失X（枚）", 0, 1000, 100, 10)
    st.subheader("例外補正")
    post_black = st.checkbox("バレ/黒後の可能性", value=False)
    huge_reg   = st.checkbox("超大ハマREGを直近で引いた", value=False)

# 65%換算（model%に写像）と 50%換算
c = 50.0 / base_per_50  # 純投資/G
offset = (1.0 - c/3.0) * 100.0
model65 = 65.0 - offset
model50 = 50.0 - offset

# ===================== Session: history & keypad state =====================
if "pad_value" not in st.session_state: st.session_state.pad_value = ""
if "hist_rows" not in st.session_state:
    st.session_state.hist_rows = [{"IntervalG":120,"Type":"BIG","セグ開始":False}]
if "undo" not in st.session_state: st.session_state.undo = []
if "redo" not in st.session_state: st.session_state.redo = []
if "pending_g_override" not in st.session_state: st.session_state.pending_g_override = 0

def push_undo():
    st.session_state.undo.append(deepcopy(st.session_state.hist_rows))
    st.session_state.redo.clear()

def do_undo():
    if st.session_state.undo:
        st.session_state.redo.append(deepcopy(st.session_state.hist_rows))
        st.session_state.hist_rows = st.session_state.undo.pop()
        st.rerun()

def do_redo():
    if st.session_state.redo:
        st.session_state.undo.append(deepcopy(st.session_state.hist_rows))
        st.session_state.hist_rows = st.session_state.redo.pop()
        st.rerun()

# ---- keypad helpers (force immediate rerun for no-lag) ----
def _push_digit(d):
    st.session_state.pad_value = (st.session_state.pad_value + d)[:4]
    st.rerun()

def _backspace():
    st.session_state.pad_value = st.session_state.pad_value[:-1]
    st.rerun()

def _clear():
    st.session_state.pad_value = ""
    st.rerun()

def _get_pad_int():
    try: return int(st.session_state.pad_value or "0")
    except: return 0

def _apply_to_pending():
    n = _get_pad_int()
    st.session_state.pending_g_override = n
    st.session_state.pad_value = ""
    st.rerun()

def _add_row(t):
    n = _get_pad_int()
    push_undo()
    st.session_state.hist_rows.append({"IntervalG": n, "Type": t, "セグ開始": False})
    st.session_state.pad_value = ""
    st.rerun()

def _del_last():
    if st.session_state.hist_rows:
        push_undo()
        st.session_state.hist_rows = st.session_state.hist_rows[:-1]
        st.rerun()

def _reset_all():
    push_undo()
    st.session_state.hist_rows = []
    st.rerun()

def _delete_one_row(idx):
    if 0 <= idx < len(st.session_state.hist_rows):
        push_undo()
        st.session_state.hist_rows.pop(idx)
        st.rerun()

def _mark_latest_seg():
    if len(st.session_state.hist_rows)>0:
        push_undo()
        for r in st.session_state.hist_rows: r["セグ開始"] = False
        st.session_state.hist_rows[-1]["セグ開始"] = True
        st.rerun()

# ===================== History Editor =====================
st.markdown("### 履歴")
hist_df = pd.DataFrame(st.session_state.hist_rows)
grid = st.data_editor(
    hist_df, num_rows="dynamic", use_container_width=True,
    column_config={
        "IntervalG": st.column_config.NumberColumn("当たり間G(通常)", min_value=0, max_value=5000, step=1),
        "Type": st.column_config.SelectboxColumn("種別", options=["BIG","REG"]),
        "セグ開始": st.column_config.CheckboxColumn("ここから区切る")
    },
    key="hist_editor_v046"
)
st.session_state.hist_rows = grid.to_dict(orient="records")

col_btn = st.columns([1,1,1,1,1])
with col_btn[0]:
    if st.button("リセット（履歴を空に）", use_container_width=True): _reset_all()
with col_btn[1]:
    if st.button("1つ削除（末尾）", use_container_width=True): _del_last()
with col_btn[2]:
    idx = st.number_input("削除する行#", min_value=0, max_value=max(0,len(st.session_state.hist_rows)-1), value=0, step=1, key="del_idx")
with col_btn[3]:
    if st.button("↑ 行を削除", use_container_width=True): _delete_one_row(st.session_state.get("del_idx",0))
with col_btn[4]:
    if st.button("最新に区切る", use_container_width=True): _mark_latest_seg()

# Pending G
pending_default = int(st.session_state.get("pending_g_override", 0))
pending_g = st.number_input("現在のハマりG（未当選）", min_value=0, max_value=5000, value=pending_default, step=10)

# ===================== Compute =====================
df = pd.DataFrame(st.session_state.hist_rows).dropna(subset=["IntervalG","Type"])
df["IntervalG"] = pd.to_numeric(df["IntervalG"], errors="coerce").fillna(0).astype(int)
true_idx = [i for i, v in enumerate(df.get("セグ開始", [])) if bool(v)]
seg_start_pos = true_idx[-1] if true_idx else None
seg_df = df.iloc[seg_start_pos+1:] if seg_start_pos is not None else df

params = dict(big_pay=BIG_PAY_DEFAULT, reg_pay=REG_PAY_DEFAULT, big_g=BIG_G_DEFAULT, reg_g=REG_G_DEFAULT, coin_per_g=COIN_PER_G_DEFAULT)
stats = compute_segment_stats(seg_df, pending_g, params)
cum_adv, short_pct_model = stats["cum_adv"], stats["short_pct_model"]

# model% → equiv%（ベース戻りを加えた見え方）
equiv_pct = (short_pct_model + (1.0 - (50.0/base_per_50)/3.0) * 100.0) if np.isfinite(short_pct_model) else np.nan

# Distances
dist1600 = max(0, int(1600 - cum_adv))
dist3500 = max(0, int(3500 - cum_adv))

# 2Dビン
if bin_tbl is not None and np.isfinite(short_pct_model):
    mrow, b1, b2 = lookup_model(bin_tbl, short_pct_model, cum_adv)
    if mrow is None:
        p, E_len, p_le2 = 28.0, 3.2, 50.0
        n_info = "(fallback)"
    else:
        p, E_len, p_le2 = mrow["p_trig"], mrow["E_len"], mrow["p_le2"]
        n_info = f"n={mrow['n']}/pos={mrow['n_pos']}"
else:
    p, E_len, p_le2, b1, b2, n_info = 28.0, 3.2, 50.0, "n/a", "n/a", ""

# 補正（model%_65 / _50 使用）
dP, dE, dLe2 = regional_adjust(cum_adv, short_pct_model if np.isfinite(short_pct_model) else 60.0, model65, model50, pending_g, {"post_black":post_black,"huge_reg":huge_reg})
p_adj = clip_pct(p + dP); E_len_adj = max(1.0, E_len + dE); p_le2_adj = clip01(p_le2 + dLe2)

# Horizon auto choose
candidates = [None, 1600, 1750, 2000, 3300, 3500, 3700]
def ev_with_h(h):
    add_g = 0 if h is None else max(0, int(h - cum_adv))
    invest = (pending_g + add_g) * COIN_PER_G_DEFAULT  # 掛け枚数ベース
    heaven_expect = (p_adj/100.0) * (E_len_adj * HEAVEN_AVG_GAIN_PER_ROUND_DEFAULT)
    return heaven_expect - invest, add_g
best_ev, best_h, best_addg = -1e9, None, 0
for h in candidates:
    ev, addg = ev_with_h(h)
    if ev > best_ev: best_ev, best_h, best_addg = ev, h, addg

decision = ("押す" if (best_ev>0 and p_adj>=32.0) else ("様子見" if abs(best_ev)<=20 else "引く"))
warn_badge = (p_le2_adj >= 40.0 and E_len_adj <= 3.2)

# ===================== Output =====================
st.markdown("### サマリー")
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    st.metric("P(天国)", f"{p_adj:.1f} %", help=f"bin: {b1}×{b2} {n_info}")
with c2:
    if np.isfinite(short_pct_model):
        st.metric("model% / equiv%", f"{short_pct_model:.1f} % / {equiv_pct:.1f} %",
                  help=f"65%換算の目安: model%≧{model65:.1f}%（ベース{base_per_50}G/50枚）")
    else:
        st.metric("model% / equiv%", "—", help="データ不足")
with c3:
    st.metric("E[連] / P(≤2連)", f"{E_len_adj:.2f} / {p_le2_adj:.1f} %")
    if warn_badge: st.markdown('<span class="badge-warn">短連リスク↑</span>', unsafe_allow_html=True)
with c4:
    st.metric("EV（最良horizon）", f"{best_ev:.0f} 枚", help=f"horizon: {best_h or '次当たり'} / 追加G:{best_addg}")
    st.subheader(decision)

st.markdown("### 距離インジケータ")
colx, coly, colz = st.columns([1,1,1])
with colx: st.metric("累積有利G（未当選込み）", f"{cum_adv:.0f} G")
with coly: st.metric("1600まで", f"{dist1600} G")
with colz: st.metric("3500まで", f"{dist3500} G")

with st.expander("根拠（適用補正とビン）", expanded=False):
    st.write(f"- 2Dビン: **{b1} × {b2}** → p={p:.1f}%, E={E_len:.2f}, ≤2={p_le2:.1f}%")
    applied = []
    # 1600帯（文言を65%換算で説明）
    if 1580 <= cum_adv <= 1680 and short_pct_model >= model65: applied.append(f"1600帯: model%≥{model65:.1f}%（65%相当） → P-3pt")
    if 1550 <= cum_adv <= 1800 and short_pct_model < model50:  applied.append(f"1600帯: model%<{model50:.1f}%（50%相当） → P+2pt")
    if 1720 <= cum_adv <= 1820:                                applied.append("1600帯: 近傍 → P+1pt")
    # 3500帯
    if 3300 <= cum_adv <= 3450: applied.append("3500帯: 3300–3450 → P-2/E-0.3/≤2+2pt")
    if 3450 <= cum_adv <= 3550: applied.append("3500帯: 3450–3550 → P+1/E+1.0/≤2-5pt (+低model%ならE+0.5)")
    if 3550 <= cum_adv <= 3700: applied.append("3500帯: 3550–3700 → P+1/E-1.0/≤2+4pt")
    # 現在ハマり
    if pending_g >= 540: applied.append("現在ハマり≥540 → P+2pt")
    elif pending_g >= 230: applied.append("現在ハマり≥230 → P+1pt")
    # 例外
    if post_black: applied.append("バレ/黒後 → P-3pt")
    if huge_reg:   applied.append("超大ハマREG → P-2pt")
    for a in applied: st.write("・"+a)
    if np.isfinite(short_pct_model):
        st.write(f"model%={short_pct_model:.1f}% / equiv%={equiv_pct:.1f}%（offset={offset:.1f}pt）")

with st.expander("アクティブ区間（表）", expanded=False):
    st.dataframe(seg_df)

# ===================== Sticky keypad (bottom fixed) =====================
st.markdown("---")
with st.container():
    st.markdown('<div class="sticky-pad">', unsafe_allow_html=True)
    st.markdown("**クイック入力（テンキー）**　"
                f"<span class='small-note'>入力値: {st.session_state.pad_value or '—'}</span>",
                unsafe_allow_html=True)
    colK = st.columns([2,2,2,2])
    with colK[0]:
        # keypad numbers
        cols = st.columns(3)
        if cols[0].button("1", key="k1"): _push_digit("1")
        if cols[1].button("2", key="k2"): _push_digit("2")
        if cols[2].button("3", key="k3"): _push_digit("3")
        cols = st.columns(3)
        if cols[0].button("4", key="k4"): _push_digit("4")
        if cols[1].button("5", key="k5"): _push_digit("5")
        if cols[2].button("6", key="k6"): _push_digit("6")
        cols = st.columns(3)
        if cols[0].button("7", key="k7"): _push_digit("7")
        if cols[1].button("8", key="k8"): _push_digit("8")
        if cols[2].button("9", key="k9"): _push_digit("9")
        cols = st.columns(3)
        if cols[0].button("⌫", key="kbs"): _backspace()
        if cols[1].button("0", key="k0"): _push_digit("0")
        if cols[2].button("C", key="kcl"): _clear()
    with colK[1]:
        if st.button("行を追加：BIG", key="add_big"): _add_row("BIG")
        if st.button("行を追加：REG", key="add_reg"): _add_row("REG")
    with colK[2]:
        if st.button("現在ハマりに適用", key="apply_pending"): _apply_to_pending()
        if st.button("最新に区切る", key="seg_latest"): _mark_latest_seg()
    with colK[3]:
        if st.button("UNDO", key="undo_btn"): do_undo()
        if st.button("REDO", key="redo_btn"): do_redo()
        if st.button("末尾を削除", key="del_last_btn"): _del_last()

    st.markdown('</div>', unsafe_allow_html=True)
