# app.py — 沖ドキ！BLACK 押し引き補助 v0.51+ (テンキー復活 & 空セル安全化)
from __future__ import annotations
import math, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------ Page & CSS ------------------------
st.set_page_config(page_title="沖ドキ！BLACK 押し引き補助", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
div[data-testid="stMetricValue"]{
  overflow: visible !important; text-overflow: clip !important; white-space: nowrap !important;
  font-variant-numeric: tabular-nums;
}
.summary-grid{ display:grid; gap:14px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
.scard{ padding:12px 14px; border:1px solid #333; border-radius:14px; background:rgba(255,255,255,.03); }
.slabel{ font-size:12px; opacity:.75; margin-bottom:4px; white-space:nowrap; }
.sval{ font-size: clamp(18px, 5.2vw, 38px); font-weight:700; line-height:1.05; white-space: nowrap; overflow: visible; text-overflow: clip; font-variant-numeric: tabular-nums; }
.small-note { color:#aaa; font-size:12px; }
.badge-warn { background:#5b1e1e; color:#ffb3b3; padding:3px 8px; border-radius:10px; font-weight:600; display:inline-block; }
.kb {display:grid; grid-template-columns:repeat(3,minmax(72px,1fr)); gap:8px; }
.kbtn{ padding:14px 0; border:1px solid #444; border-radius:14px; text-align:center; font-size:20px; }
.knote{ opacity:.7; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ------------------------ Model CSV loader ------------------------
def load_model_table() -> Optional[pd.DataFrame]:
    cands = [
        Path(__file__).resolve().parent / "model_bins_v1.csv",
        Path.cwd() / "model_bins_v1.csv",
        Path.cwd() / "data" / "model_bins_v1.csv",
        Path.cwd() / "models" / "model_bins_v1.csv",
    ]
    df = None
    for p in cands:
        if p.exists():
            try: df = pd.read_csv(p, encoding="utf-8-sig")
            except Exception: df = pd.read_csv(p)
            break
    if df is None: return None

    df = df.rename(columns={
        "pct_bin":"bin_pct","adv_bin":"bin_adv",
        "p_trig":"p_trig_sm","p_trigger":"p_trig_sm",
        "elen":"E_len_sm","p_le2":"p_le2_sm",
    })
    need = {"bin_pct","bin_adv","p_trig_sm","E_len_sm","p_le2_sm"}
    if not need.issubset(df.columns): return None

    for c in ["p_trig_sm","p_le2_sm"]:
        mx = pd.to_numeric(df[c], errors="coerce").max()
        if mx is not None and mx > 1.01:
            df[c] = pd.to_numeric(df[c], errors="coerce")/100.0

    def pct_range(lbl:str)->Tuple[float,float]:
        mp={"<40":(-math.inf,40),"40-50":(40,50),"50-60":(50,60),"60-65":(60,65),
            "65-70":(65,70),"70-80":(70,80),"80+":(80,math.inf)}
        if lbl in mp: return mp[lbl]
        t=str(lbl).replace("％","%").replace("〜","-").replace("–","-").replace(" ","")
        m=re.findall(r"[0-9]+(?:\.[0-9]+)?",t)
        if len(m)==2: a,b=map(float,m); return (min(a,b),max(a,b))
        if len(m)==1: v=float(m[0]); return (v,v)
        return (80,math.inf)

    def adv_range(lbl:str)->Tuple[float,float]:
        mp={"<=800":(-math.inf,800),"800-1200":(800,1200),"1200-1600":(1200,1600),
            "1600-1800":(1600,1800),"1800-2200":(1800,2200),">2200":(2200,math.inf)}
        if lbl in mp: return mp[lbl]
        t=str(lbl).replace("〜","-").replace("–","-").replace(" ","")
        if t.startswith("<") or t.startswith("≤"): return (-math.inf,800)
        if t.endswith("+") or t.startswith(">"): return (2200,math.inf)
        m=re.findall(r"[0-9]+(?:\.[0-9]+)?",t)
        if len(m)==2: a,b=map(float,m); return (min(a,b),max(a,b))
        if len(m)==1: v=float(m[0]); return (v,v)
        return (2200,math.inf)

    if "pct_lo" not in df.columns or "pct_hi" not in df.columns:
        lo,hi=zip(*[pct_range(x) for x in df["bin_pct"]]); df["pct_lo"]=lo; df["pct_hi"]=hi
    if "adv_lo" not in df.columns or "adv_hi" not in df.columns:
        lo,hi=zip(*[adv_range(x) for x in df["bin_adv"]]); df["adv_lo"]=lo; df["adv_hi"]=hi
    return df

BIN_TBL = load_model_table()
if BIN_TBL is not None:
    st.caption(f"🗃️ モデルCSV: **OK** / rows={len(BIN_TBL)} / 読込: model_bins_v1.csv")
else:
    st.error("モデルCSV `model_bins_v1.csv` が見つからない/列不足です。リポ直下に配置してください。")

def pick_model(short_pct: float, adv_g: float) -> Dict:
    if BIN_TBL is None or len(BIN_TBL)==0:
        return {"p_trig_sm":0.28,"E_len_sm":3.2,"p_le2_sm":0.50,"bin":"n/a×n/a (fallback)"}
    df=BIN_TBL
    m=df[(df["pct_lo"]<=short_pct)&(short_pct<=df["pct_hi"])&
         (df["adv_lo"]<=adv_g)&(adv_g<=df["adv_hi"])]
    if len(m)>0:
        r=m.iloc[0].to_dict(); r["bin"]=f"{r['bin_pct']}×{r['bin_adv']} (interval)"; return r
    df=df.copy()
    df["d"]=(np.maximum(0,df["pct_lo"]-short_pct)+np.maximum(0,short_pct-df["pct_hi"]) +
             np.maximum(0,df["adv_lo"]-adv_g)+np.maximum(0,adv_g-df["adv_hi"]))
    r=df.sort_values("d",kind="mergesort").iloc[0].to_dict()
    r["bin"]=f"{r['bin_pct']}×{r['bin_adv']} (nearest)"; return r

# ------------------------ Session state ------------------------
if "hist_rows" not in st.session_state: st.session_state.hist_rows: List[Dict] = []
if "undo_stack" not in st.session_state: st.session_state.undo_stack: List[List[Dict]] = []
if "redo_stack" not in st.session_state: st.session_state.redo_stack: List[List[Dict]] = []
if "num_buf" not in st.session_state: st.session_state.num_buf = ""  # テンキーの入力バッファ
if "now_hamari" not in st.session_state: st.session_state.now_hamari = 0

def push_undo():
    st.session_state.undo_stack.append([dict(r) for r in st.session_state.hist_rows])
    st.session_state.redo_stack.clear()
def undo():
    if st.session_state.undo_stack:
        st.session_state.redo_stack.append([dict(r) for r in st.session_state.hist_rows])
        st.session_state.hist_rows = st.session_state.undo_stack.pop()
def redo():
    if st.session_state.redo_stack:
        st.session_state.undo_stack.append([dict(r) for r in st.session_state.hist_rows])
        st.session_state.hist_rows = st.session_state.redo_stack.pop()

# ------------------------ Sidebar: 設定 ------------------------
DEFAULT_CONF = {
    "base_50": 32, "coin_in_g": 3.00, "big_pay": 210, "reg_pay": 90,
    "big_adv_g": 59, "reg_adv_g": 24, "heaven_avg_diff": 100, "ev_risk_limit": 300,
}
if "conf_active" not in st.session_state:
    st.session_state.conf_active = DEFAULT_CONF.copy()
    st.session_state.conf_mode = "標準"
    st.session_state.conf_applied_label = "初期値"
for k,v in DEFAULT_CONF.items():
    st.session_state.setdefault(f"ui_{k}", st.session_state.conf_active.get(k,v))
st.session_state.setdefault("ui_mode", st.session_state.conf_mode)

def collect_ui()->Dict: return {k: st.session_state[f"ui_{k}"] for k in DEFAULT_CONF.keys()}
def is_dirty()->bool:
    ui=collect_ui(); act=st.session_state.conf_active
    if st.session_state.ui_mode != st.session_state.conf_mode: return True
    return any(ui[k]!=act.get(k) for k in ui.keys())

with st.sidebar:
    st.markdown("### 判定モード")
    st.radio("",["保守","標準","攻め"],index=["保守","標準","攻め"].index(st.session_state.conf_mode),
             key="ui_mode",horizontal=True,label_visibility="collapsed")
    st.number_input("ベース（50枚で回るG）",10,60,key="ui_base_50",step=1)
    c1,c2=st.columns(2)
    with c1: st.number_input("BIG平均枚数",0,500,key="ui_big_pay",step=5)
    with c2: st.number_input("REG平均枚数",0,300,key="ui_reg_pay",step=5)
    c3,c4=st.columns(2)
    with c3: st.number_input("BIG中の有利G(+)",0,300,key="ui_big_adv_g",step=1)
    with c4: st.number_input("REG中の有利G(+)",0,300,key="ui_reg_adv_g",step=1)
    st.number_input("通常時の投入（枚/G）",0.0,5.0,key="ui_coin_in_g",step=0.05,format="%.2f")
    st.number_input("天国1回あたり平均差枚（EV用）",0,1000,key="ui_heaven_avg_diff",step=5)
    st.number_input("攻め：許容損失（枚）",0,5000,key="ui_ev_risk_limit",step=50)
    if is_dirty(): st.warning("未反映の変更があります。「変更を反映」を押してください。", icon="⚠️")
    if st.button("変更を反映", type="primary", use_container_width=True):
        st.session_state.conf_active = collect_ui()
        st.session_state.conf_mode   = st.session_state.ui_mode
        st.session_state.conf_applied_label = "反映済み"
        st.toast("設定を反映しました。", icon="✅"); st.rerun()

conf = st.session_state.conf_active
st.caption(
    f"適用設定: ベース {conf['base_50']}G/50枚, 通常 {conf['coin_in_g']:.2f}枚/G, "
    f"BIG {conf['big_pay']}枚(+{conf['big_adv_g']}G), REG {conf['reg_pay']}枚(+{conf['reg_adv_g']}G) "
    f"| モード: {st.session_state.conf_mode} | {st.session_state.conf_applied_label}"
)

# ------------------------ テンキー（任意） ------------------------
st.markdown("### クイック入力（テンキー）")
st.write(f"入力値: **{st.session_state.num_buf or '—'}**　<span class='knote'>テンキー → 追加 or 現在ハマリに適用</span>", unsafe_allow_html=True)

def tap(d:str):
    s=st.session_state.num_buf
    if d=="C": st.session_state.num_buf=""
    elif d=="⌫": st.session_state.num_buf=s[:-1]
    else: st.session_state.num_buf=(s + d)[:6]  # 6桁まで
def buf_int()->int:
    try: return int(st.session_state.num_buf) if st.session_state.num_buf!="" else 0
    except: return 0

kb_rows=[["1","2","3"],["4","5","6"],["7","8","9"],["⌫","0","C"]]
for row in kb_rows:
    cols=st.columns(3)
    for i,k in enumerate(row):
        if cols[i].button(k, use_container_width=True, key=f"kb_{k}_{i}", help="テンキー"):
            tap(k); st.rerun()

cA,cB,cC,cD=st.columns([1.2,1.2,1.2,1])
with cA:
    if st.button("行を追加：BIG", use_container_width=True):
        push_undo()
        st.session_state.hist_rows.append({"IntervalG": buf_int(), "Type":"BIG", "セグ開始": False})
        st.session_state.num_buf=""; st.rerun()
with cB:
    if st.button("行を追加：REG", use_container_width=True):
        push_undo()
        st.session_state.hist_rows.append({"IntervalG": buf_int(), "Type":"REG", "セグ開始": False})
        st.session_state.num_buf=""; st.rerun()
with cC:
    if st.button("現在ハマリに適用", use_container_width=True):
        st.session_state.now_hamari = buf_int()
        st.session_state.num_buf=""; st.rerun()
with cD:
    if st.button("末尾を削除", use_container_width=True) and st.session_state.hist_rows:
        push_undo(); st.session_state.hist_rows.pop(); st.rerun()

# ------------------------ 履歴エディタ（チェックボックス無し） ------------------------
st.markdown("### 履歴")
hist_rows = st.session_state.hist_rows or []
df_init = pd.DataFrame(hist_rows)
seg_flags = df_init["セグ開始"].astype(bool).tolist() if "セグ開始" in df_init.columns else [False]*len(df_init)

if len(df_init)==0:
    df_show = pd.DataFrame(columns=["IntervalG","Type"])
else:
    df_show = df_init.reindex(columns=["IntervalG","Type"]).copy()
    df_show["IntervalG"] = pd.to_numeric(df_show["IntervalG"], errors="coerce").fillna(0).astype(int)
    df_show["Type"]      = df_show["Type"].fillna("BIG")

grid = st.data_editor(
    df_show, num_rows="dynamic", use_container_width=True,
    column_config={
        "IntervalG": st.column_config.NumberColumn("当たり間G(通常)", min_value=0, max_value=5000, step=1),
        "Type":      st.column_config.SelectboxColumn("種別", options=["BIG","REG"]),
    },
    column_order=["IntervalG","Type"], key="hist_editor_v051_noseg",
)

c_del1,c_del2,_ = st.columns([1,1,2])
with c_del1:
    if st.button("リセット（履歴を空に）", use_container_width=True):
        push_undo(); st.session_state.hist_rows=[]; st.rerun()
with c_del2:
    if st.button("1つ削除（末尾）", use_container_width=True) and st.session_state.hist_rows:
        push_undo(); st.session_state.hist_rows.pop(); st.rerun()

# エディタ結果 → state へ戻す（空セル安全化）
def safe_int_cell(x) -> int:
    try:
        v = pd.to_numeric(x, errors="coerce")
        # pandasが返すNaNはfloat。math.isnanで判定
        f = float(v) if not isinstance(v, pd.Series) else float(v.iloc[0])
        return 0 if math.isnan(f) else int(round(f))
    except Exception:
        return 0

df_after = pd.DataFrame(grid)
m = len(df_after)
seg_flags = (seg_flags + [False]*(m-len(seg_flags)))[:m]
new_rows=[]
for i in range(m):
    iv = safe_int_cell(df_after.iloc[i].get("IntervalG", 0))
    tp = str(df_after.iloc[i].get("Type", "BIG")) or "BIG"
    new_rows.append({"IntervalG": iv, "Type": tp, "セグ開始": bool(seg_flags[i])})
st.session_state.hist_rows = new_rows

# ------------------------ 区切り：スライダー＋決定 ------------------------
st.markdown("#### 区切り操作（指一本）")
N = len(st.session_state.hist_rows)
st.session_state.seg_slider_v051 = max(1, min(int(st.session_state.get("seg_slider_v051", 1)), max(1, N)))
if N>0:
    st.slider("区切り行（1〜N）", 1, N, value=st.session_state.seg_slider_v051,
              key="seg_slider_v051", help="この行の“直後”から現在までをアクティブ区間にします。")
    cL,cC,cR=st.columns([1,1,2])
    with cL:
        if st.button("−", use_container_width=True):
            st.session_state.seg_slider_v051=max(1,st.session_state.seg_slider_v051-1); st.rerun()
    with cC:
        if st.button("+", use_container_width=True):
            st.session_state.seg_slider_v051=min(N,st.session_state.seg_slider_v051+1); st.rerun()
    with cR:
        if st.button("ここに区切る", type="primary", use_container_width=True):
            pos0=int(st.session_state.seg_slider_v051)-1
            push_undo()
            for r in st.session_state.hist_rows: r["セグ開始"]=False
            st.session_state.hist_rows[pos0]["セグ開始"]=True
            st.toast(f"{st.session_state.seg_slider_v051} 行目に区切りを設定しました", icon="✅"); st.rerun()
    if st.button("区切りを解除", use_container_width=True):
        push_undo(); [r.update({"セグ開始":False}) for r in st.session_state.hist_rows]
        st.toast("区切りを解除しました", icon="🧹"); st.rerun()
else:
    st.info("履歴が空です。行を追加してから区切りを設定してください。")

# ------------------------ 集計 ------------------------
def active_segment_rows(rows: List[Dict]) -> List[Dict]:
    if not rows: return []
    pos=-1
    for i,r in enumerate(rows):
        if bool(r.get("セグ開始")): pos=i
    return rows[pos+1:] if pos>=0 else rows

def calc_metrics(rows: List[Dict], now_hamari_g: int, conf: Dict) -> Dict:
    total_iv=sum(int(r.get("IntervalG",0)) for r in rows)
    big_n   =sum(1 for r in rows if str(r.get("Type","BIG"))=="BIG")
    reg_n   =sum(1 for r in rows if str(r.get("Type","REG"))=="REG")

    coin_in = float(conf["coin_in_g"]) * (total_iv + now_hamari_g)
    payout  = big_n*int(conf["big_pay"]) + reg_n*int(conf["reg_pay"])
    short_pct = (payout/coin_in*100.0) if coin_in>0 else 0.0

    adv_g = total_iv + big_n*int(conf["big_adv_g"]) + reg_n*int(conf["reg_adv_g"]) + int(now_hamari_g)

    rowm   = pick_model(short_pct, adv_g)
    p_trig = float(rowm.get("p_trig_sm",0.28))
    e_len  = float(rowm.get("E_len_sm",3.2))
    p_le2  = float(rowm.get("p_le2_sm",0.50))
    binlab = str(rowm.get("bin","n/a×n/a"))

    ev = p_trig * (e_len * float(conf["heaven_avg_diff"])) - (int(now_hamari_g)*float(conf["coin_in_g"]))
    return {"coin_in":int(round(coin_in)),"payout":int(payout),"short_pct":float(short_pct),
            "adv_g":int(adv_g),"p_trig":p_trig,"e_len":e_len,"p_le2":p_le2,"bin":binlab,"ev":int(round(ev))}

now_hamari = st.number_input("現在のハマりG（未当選）", min_value=0, max_value=5000,
                             value=int(st.session_state.now_hamari), step=1, key="now_hamari")
seg_rows = active_segment_rows(st.session_state.hist_rows)
res = calc_metrics(seg_rows, int(st.session_state.now_hamari), conf)

# ------------------------ サマリー ------------------------
st.markdown("### サマリー")
st.markdown(f"<div class='small-note'>bin: {res['bin']} / 累積有利G(未当選込み) {res['adv_g']} G / 短期% {res['short_pct']:.1f}%</div>", unsafe_allow_html=True)
st.markdown(
    f"""
<div class="summary-grid">
  <div class="scard"><div class="slabel">P(天国)</div><div class="sval">{res['p_trig']*100:.1f} %</div></div>
  <div class="scard"><div class="slabel">E[連] / P(≤2連)</div><div class="sval">{res['e_len']:.2f} / {res['p_le2']*100:.1f} %</div></div>
  <div class="scard"><div class="slabel">短期機械割（払出/投入）</div><div class="sval">{res['short_pct']:.1f} %</div></div>
  <div class="scard"><div class="slabel">EV（ざっくり）</div><div class="sval">{res['ev']:+,} 枚</div></div>
</div>
""", unsafe_allow_html=True
)
dist1600=max(0,1600-res["adv_g"]); dist3500=max(0,3500-res["adv_g"])
st.markdown(f"<div class='small-note'>1600まで: {dist1600} G / 3500まで: {dist3500} G</div>", unsafe_allow_html=True)

# ざっくり判定
mode=st.session_state.conf_mode; judge="様子見"
if mode=="保守":
    if res["ev"]>0 and res["p_trig"]>=0.40 and res["p_le2"]<=0.45: judge="押す"
elif mode=="標準":
    if res["ev"]>0 and res["p_trig"]>=0.32: judge="押す"
elif mode=="攻め":
    if res["ev"]>-abs(conf["ev_risk_limit"]) and res["p_trig"]>=0.28: judge="押す"
st.subheader(f"判定：{judge}")

# ------------------------ UNDO/REDO ------------------------
c_ur1,c_ur2,_=st.columns([1,1,3])
with c_ur1:
    if st.button("UNDO", use_container_width=True): undo(); st.rerun()
with c_ur2:
    if st.button("REDO", use_container_width=True): redo(); st.rerun()

st.markdown("<span class='small-note'>※本ツールは実戦データの傾向と一般的な試験枠組みを元にしたヒューリスティックです。最終判断は自己責任で。</span>", unsafe_allow_html=True)
