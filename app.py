
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy

# ===================== Constants =====================
BIG_PAY_DEFAULT = 210
REG_PAY_DEFAULT = 90
BIG_G_DEFAULT   = 59
REG_G_DEFAULT   = 24
COIN_PER_G_DEFAULT = 3.0
HEAVEN_AVG_GAIN_PER_ROUND_DEFAULT = 120

APP_TITLE = "沖ドキ！BLACK 押し引き v0.50（固定モデル＋スマホUI）"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"## {APP_TITLE}")
st.caption("入力は「履歴＋現在ハマりG＋（任意で区切り線）」だけ。テンキー位置は上/下を切替可能。")

# ---------- Keep scroll position across reruns (Cloud対策) ----------
st.markdown("""
<script>
(() => {
  const KEY='__okiblack_scrollY';
  const y = sessionStorage.getItem(KEY);
  if (y !== null) { try { window.scrollTo(0, parseInt(y)); } catch(e){} }
  const save = () => sessionStorage.setItem(KEY, String(window.scrollY||0));
  window.addEventListener('scroll', save, {passive:true});
  document.addEventListener('click', save, true);
  window.addEventListener('beforeunload', save);
})();
</script>
""", unsafe_allow_html=True)

# ------------------------ Styles ------------------------
st.markdown("""
<style>
:root { --btn-h: 48px; --btn-fs: 18px; }
.stButton>button { height: var(--btn-h); font-size: var(--btn-fs); border-radius: 12px; }
.stNumberInput input { font-size: 18px; }
[data-testid="stMetricDelta"] { font-size: 14px !important; }

/* Sticky header/footer for keypad */
.sticky-top {
  position: sticky; top: 0; z-index: 999;
  background: rgba(24,24,28,.98);
  border-bottom: 1px solid rgba(255,255,255,.08);
  padding: .6rem .6rem .8rem .6rem;
  margin: -0.6rem -0.6rem .6rem -0.6rem;
}
.sticky-bottom {
  position: fixed; left:0; right:0; bottom:0; z-index: 999;
  background: rgba(24,24,28,.98);
  border-top: 1px solid rgba(255,255,255,.08);
  padding: .6rem .8rem 1.0rem .8rem;
}
/* Reserve bottom space when bottom keypad is used */
body::after { content:""; display:block; height: var(--reserve, 0px); }
.reserve-bottom { --reserve: 270px; }

/* small note */
.small-note { color:#aaa; font-size:12px; }
.badge-warn { background:#5b1e1e; color:#ffb3b3; padding:4px 8px; border-radius:10px; font-weight:600; display:inline-block; }

/* tighten table spacing on mobile */
@media (max-width: 540px) {
  :root { --btn-h: 56px; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------ Model helpers ------------------------
# 置換：モデル読み込みを強化
def load_model_table():
<<<<<<< HEAD
    from pathlib import Path
    import pandas as pd
=======
    """Try multiple locations and tolerate BOM/TSV mistakes."""
>>>>>>> okidoki black helper v0.49: fix keys, scroll, empty-history, mobile keypad
    candidates = [
        Path(__file__).resolve().parent / "model_bins_v1.csv",
        Path.cwd() / "model_bins_v1.csv",
        Path.cwd() / "data" / "model_bins_v1.csv",
        Path.cwd() / "models" / "model_bins_v1.csv",
    ]
    for p in candidates:
        if p.exists():
            # try normal CSV with utf-8-sig first
            try:
<<<<<<< HEAD
                return pd.read_csv(p, encoding="utf-8-sig")
=======
                df = pd.read_csv(p, encoding="utf-8-sig")
                if "bin_pct" in df.columns and "bin_adv" in df.columns:
                    return df
            except Exception:
                pass
            # if looks like TSV, try tab
            try:
                df = pd.read_csv(p, sep="\t", encoding="utf-8-sig")
                if "bin_pct" in df.columns and "bin_adv" in df.columns:
                    return df
>>>>>>> okidoki black helper v0.49: fix keys, scroll, empty-history, mobile keypad
            except Exception:
                # 次の候補を試す
                pass
    return None

<<<<<<< HEAD
# 読込後すぐ（bin_tbl = load_model_table() の直後あたり）
status = "OK" if bin_tbl is not None else "NG"
rows = len(bin_tbl) if bin_tbl is not None else 0
st.caption(f"📦 モデルCSV: {status}  / rows={rows}  / 期待ファイル名: model_bins_v1.csv")
if bin_tbl is None:
    st.error("model_bins_v1.csv が見つかりません。app.py と同じ階層に置いてコミットしてください。"
             "（.gitignoreの *.csv を外すか `git add -f model_bins_v1.csv`）")

=======
bin_tbl = load_model_table()

# 読込ステータスを表示（デバッグ用）
status = "OK" if bin_tbl is not None else "NG"
rows = len(bin_tbl) if bin_tbl is not None else 0
cols = ", ".join(bin_tbl.columns) if bin_tbl is not None else "-"
st.caption(f"📦 モデルCSV: {status} / rows={rows} / cols=[{cols}] / 期待名: model_bins_v1.csv")
if bin_tbl is None:
    st.error("model_bins_v1.csv が読めていません。app.py と同じ階層に置き、.gitignoreに弾かれていないか確認。"
             "必要なら `git add -f model_bins_v1.csv` を実行してください。")
>>>>>>> okidoki black helper v0.49: fix keys, scroll, empty-history, mobile keypad

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
    if 1580 <= cum_adv <= 1680 and short_pct_model >= model65: dP -= 3.0
    if 1550 <= cum_adv <= 1800 and short_pct_model < model50:  dP += 2.0
    if 1720 <= cum_adv <= 1820:                                dP += 1.0
    if 3300 <= cum_adv <= 3450: dP -= 2.0; dE -= 0.3; dLe2 += 2.0
    if 3450 <= cum_adv <= 3550: dP += 1.0; dE += 1.0; dLe2 -= 5.0
    if 3450 <= cum_adv <= 3550 and short_pct_model < model50:  dE += 0.5
    if 3550 <= cum_adv <= 3700: dP += 1.0; dE -= 1.0; dLe2 += 4.0
    if pending_g >= 540: dP += 2.0
    elif pending_g >= 230: dP += 1.0
    if flags.get("post_black", False): dP -= 3.0
    if flags.get("huge_reg", False):   dP -= 2.0
    return dP, dE, dLe2

def clip_pct(x, lo=5.0, hi=85.0): return float(max(lo, min(hi, x)))
def clip01(x): return float(max(0.0, min(100.0, x)))

def compute_segment_stats(seg_df, pending_g, params):
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

# ------------------------ Sidebar ------------------------
with st.sidebar:
    mode = st.radio("判定モード", ["保守","標準","攻め"], index=1, horizontal=True, key="mode")
    base_per_50 = st.number_input("ベース（50枚で回るG）", min_value=25, max_value=40, value=32, step=1, key="baseG")
    keypad_pos = st.selectbox("テンキーの位置", ["上（推奨）","下（親指派）"], index=0, key="keypad_pos")
    with st.expander("上級設定（普段は不要）", expanded=False):
        st.number_input("BIG平均枚数", 150, 280, BIG_PAY_DEFAULT, 5, key="big_pay")
        st.number_input("REG平均枚数", 50, 150, REG_PAY_DEFAULT, 5, key="reg_pay")
        st.number_input("BIG中の有利G (+)", 40, 80, BIG_G_DEFAULT, 1, key="big_g")
        st.number_input("REG中の有利G (+)", 15, 40, REG_G_DEFAULT, 1, key="reg_g")
        st.number_input("通常時の投入（枚/G）", 1.0, 5.0, COIN_PER_G_DEFAULT, 0.5, key="coin_per_g")
        st.number_input("天国1回あたり平均差枚（固定）", 60, 200, HEAVEN_AVG_GAIN_PER_ROUND_DEFAULT, 5, key="heaven_gain")
        st.number_input("攻め：許容損失X（枚）", 0, 1000, 100, 10, key="loss_tol")
    st.subheader("例外補正")
    post_black = st.checkbox("バレ/黒後の可能性", value=False, key="post_black")
    huge_reg   = st.checkbox("超大ハマREGを直近で引いた", value=False, key="huge_reg")

# 65%換算（model%に写像）と 50%換算
c = 50.0 / st.session_state["baseG"]
offset = (1.0 - c/3.0) * 100.0
model65 = 65.0 - offset
model50 = 50.0 - offset

# ------------------------ Session ------------------------
if "pad_value" not in st.session_state: st.session_state.pad_value = ""
if "hist_rows" not in st.session_state:
    st.session_state.hist_rows = [{"IntervalG":120,"Type":"BIG","セグ開始":False}]
if "undo" not in st.session_state: st.session_state.undo = []
if "redo" not in st.session_state: st.session_state.redo = []
if "pending_g_override" not in st.session_state: st.session_state.pending_g_override = 0

# helpers
def push_undo():
    st.session_state.undo.append(deepcopy(st.session_state.hist_rows))
    st.session_state.redo.clear()

# keypad handlers
def _push_digit(d): st.session_state.pad_value = (st.session_state.pad_value + d)[:4]; st.rerun()
def _backspace(): st.session_state.pad_value = st.session_state.pad_value[:-1]; st.rerun()
def _clear(): st.session_state.pad_value = ""; st.rerun()
def _get_pad_int():
    try: return int(st.session_state.pad_value or "0")
    except: return 0
def _apply_to_pending():
    st.session_state.pending_g_override = _get_pad_int()
    st.session_state.pad_value = ""
    st.rerun()
def _add_row(t):
    n = _get_pad_int()
    if n <= 0:
        st.toast("G数が未入力です。テンキーで入れてください。", icon="⚠️")
        return
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

# ------------------------ Keypad (fragment) ------------------------
@st.fragment
def keypad_fragment(position="top"):
    klass = "sticky-top" if position=="top" else "sticky-bottom"
    if position=="bottom":
        st.markdown('<div class="reserve-bottom"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{klass}">', unsafe_allow_html=True)
    st.markdown(f"**クイック入力（テンキー）**　<span class='small-note'>入力値: <b>{st.session_state.pad_value or '—'}</b>　｜　現在ハマり: <b>{int(st.session_state.get('pending_g_override',0))}G</b></span>", unsafe_allow_html=True)

    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        if st.button("1", use_container_width=True, key="k1"): _push_digit("1")
        if st.button("4", use_container_width=True, key="k4"): _push_digit("4")
        if st.button("7", use_container_width=True, key="k7"): _push_digit("7")
        if st.button("⌫", use_container_width=True, key="kbs"): _backspace()
    with colB:
        if st.button("2", use_container_width=True, key="k2"): _push_digit("2")
        if st.button("5", use_container_width=True, key="k5"): _push_digit("5")
        if st.button("8", use_container_width=True, key="k8"): _push_digit("8")
        if st.button("0", use_container_width=True, key="k0"): _push_digit("0")
    with colC:
        if st.button("3", use_container_width=True, key="k3"): _push_digit("3")
        if st.button("6", use_container_width=True, key="k6"): _push_digit("6")
        if st.button("9", use_container_width=True, key="k9"): _push_digit("9")
        if st.button("C", use_container_width=True, key="kcl"): _clear()
    with colD:
        if st.button("行を追加：BIG", use_container_width=True, key="k_add_big"): _add_row("BIG")
        if st.button("行を追加：REG", use_container_width=True, key="k_add_reg"): _add_row("REG")
        if st.button("現在ハマりに適用", use_container_width=True, key="k_apply_pending"): _apply_to_pending()
        cL, cR = st.columns(2)
        with cL:
            if st.button("末尾を削除", use_container_width=True, key="k_del_last"): _del_last()
        with cR:
            if st.button("最新に区切る", use_container_width=True, key="k_seg_latest"): _mark_latest_seg()
    st.markdown('</div>', unsafe_allow_html=True)

# Show keypad at chosen position
keypad_fragment("top" if st.session_state["keypad_pos"].startswith("上") else "bottom")

# ------------------------ History ------------------------
st.markdown("### 履歴")
hist_rows = st.session_state.hist_rows or []
df_init = pd.DataFrame(hist_rows, columns=["IntervalG","Type","セグ開始"])
grid = st.data_editor(
    df_init, num_rows="dynamic", use_container_width=True,
    column_config={
        "IntervalG": st.column_config.NumberColumn("当たり間G(通常)", min_value=0, max_value=5000, step=1),
        "Type": st.column_config.SelectboxColumn("種別", options=["BIG","REG"]),
        "セグ開始": st.column_config.CheckboxColumn("ここから区切る")
    },
    key="hist_editor_v050"
)
st.session_state.hist_rows = grid.to_dict(orient="records")

cols = st.columns([1,1,1,1,1])
with cols[0]:
    if st.button("リセット（履歴を空に）", use_container_width=True, key="h_reset"): _reset_all()
with cols[1]:
    if st.button("1つ削除（末尾）", use_container_width=True, key="h_del_last"): _del_last()
with cols[2]:
    idx = st.number_input("削除する行#", min_value=0, max_value=max(0,len(st.session_state.hist_rows)-1), value=0, step=1, key="del_idx")
with cols[3]:
    if st.button("↑ 行を削除", use_container_width=True, key="h_del_row"): _delete_one_row(st.session_state.get("del_idx",0))
with cols[4]:
    if st.button("最新に区切る", use_container_width=True, key="h_seg_latest"): _mark_latest_seg()

pending_default = int(st.session_state.get("pending_g_override", 0))
pending_g = st.number_input("現在のハマりG（未当選）", min_value=0, max_value=5000, value=pending_default, step=10, key="pending_input")

# ------------------------ Compute ------------------------
df = pd.DataFrame(st.session_state.hist_rows, columns=["IntervalG","Type","セグ開始"]).dropna(subset=["IntervalG","Type"])
df["IntervalG"] = pd.to_numeric(df["IntervalG"], errors="coerce").fillna(0).astype(int)
true_idx = [i for i, v in enumerate(df.get("セグ開始", [])) if bool(v)]
seg_start_pos = true_idx[-1] if true_idx else None
seg_df = df.iloc[seg_start_pos+1:] if seg_start_pos is not None else df

params = dict(
    big_pay=BIG_PAY_DEFAULT, reg_pay=REG_PAY_DEFAULT,
    big_g=BIG_G_DEFAULT, reg_g=REG_G_DEFAULT,
    coin_per_g=COIN_PER_G_DEFAULT
)
stats = compute_segment_stats(seg_df, pending_g, params)
cum_adv, short_pct_model = stats["cum_adv"], stats["short_pct_model"]
equiv_pct = (short_pct_model + (1.0 - (50.0/st.session_state["baseG"])/3.0) * 100.0) if np.isfinite(short_pct_model) else np.nan

dist1600 = max(0, int(1600 - cum_adv))
dist3500 = max(0, int(3500 - cum_adv))

if bin_tbl is not None and np.isfinite(short_pct_model):
    mrow, b1, b2 = lookup_model(bin_tbl, short_pct_model, cum_adv)
    if mrow is None:
        p, E_len, p_le2 = 28.0, 3.2, 50.0; n_info="(fallback)"
    else:
        p, E_len, p_le2 = mrow["p_trig"], mrow["E_len"], mrow["p_le2"]
        n_info=f"n={mrow['n']}/pos={mrow['n_pos']}"
else:
    p, E_len, p_le2, b1, b2, n_info = 28.0, 3.2, 50.0, "n/a", "n/a", ""

dP, dE, dLe2 = regional_adjust(
    cum_adv,
    short_pct_model if np.isfinite(short_pct_model) else 60.0,
    model65, model50, pending_g,
    {"post_black": st.session_state["post_black"], "huge_reg": st.session_state["huge_reg"]}
)
p_adj = clip_pct(p + dP); E_len_adj = max(1.0, E_len + dE); p_le2_adj = clip01(p_le2 + dLe2)

candidates = [None, 1600, 1750, 2000, 3300, 3500, 3700]
def ev_with_h(h):
    add_g = 0 if h is None else max(0, int(h - cum_adv))
    invest = (pending_g + add_g) * COIN_PER_G_DEFAULT
    heaven_expect = (p_adj/100.0) * (E_len_adj * HEAVEN_AVG_GAIN_PER_ROUND_DEFAULT)
    return heaven_expect - invest, add_g
best_ev, best_h, best_addg = -1e9, None, 0
for h in candidates:
    ev, addg = ev_with_h(h)
    if ev > best_ev: best_ev, best_h, best_addg = ev, h, addg

decision = ("押す" if (best_ev>0 and p_adj>=32.0) else ("様子見" if abs(best_ev)<=20 else "引く"))
warn_badge = (p_le2_adj >= 40.0 and E_len_adj <= 3.2)

# ------------------------ Output ------------------------
st.markdown("### サマリー")
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    st.metric("P(天国)", f"{p_adj:.1f} %", help=f"bin: {b1}×{b2} {n_info}")
with c2:
    if np.isfinite(short_pct_model):
        st.metric("model% / equiv%", f"{short_pct_model:.1f} % / {equiv_pct:.1f} %", help=f"65%換算の目安: model%≧{model65:.1f}%（ベース{st.session_state['baseG']}G/50枚）")
    else:
        st.metric("model% / equiv%", "—", help="データ不足")
with c3:
    st.metric("E[連] / P(≤2連)", f"{E_len_adj:.2f} / {p_le2_adj:.1f} %")
    if warn_badge: st.markdown('<span class="badge-warn">短連リスク↑</span>', unsafe_allow_html=True)
with c4:
    st.metric("EV（最良horizon）", f"{best_ev:.0f} 枚", help=f"horizon: {best_h or '次当たり'} / 追加G:{best_addg}")
    st.subheader(decision)

st.markdown("### 距離インジケータ")
cx, cy, cz = st.columns([1,1,1])
with cx: st.metric("累積有利G（未当選込み）", f"{cum_adv:.0f} G")
with cy: st.metric("1600まで", f"{dist1600} G")
with cz: st.metric("3500まで", f"{dist3500} G")
