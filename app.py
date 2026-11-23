"""City Hospital: live ML training dashboard.   Run:  streamlit run app.py

Educational project. Never use it to diagnose real patients.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.config import DATA_PATH, FEATURE_NAMES, HIDDEN_OPTIONS, LR_OPTIONS, MEAN_COLS, SAMPLE_PATH, TARGET
from src.data_loader import load_data, prepare
from src.evaluate import confusion_figure, loss_curve, loss_curve_grid, metrics_bar_figure
from src.experiments import experiment_plan, run_experiment
from src.model import make_cfg
from src.predict import load_bundle, score_dataframe

st.set_page_config(page_title="City Hospital Training Lab", page_icon="🏥", layout="wide")
PAL = {"B": "#2a9d8f", "M": "#e63946"}


# ---------------------------------------------------------------- data
def get_df():
    if "df" in st.session_state:
        return st.session_state.df
    if DATA_PATH.exists():
        df = load_data(DATA_PATH)
    else:
        st.warning("`data/data.csv` was not found. Upload the Kaggle file below, or run `python -m src.make_data`.")
        up = st.file_uploader("Upload data.csv", type="csv")
        if up is None:
            st.stop()
        try:
            df = load_data(up)
        except ValueError as e:
            st.error(str(e))
            st.stop()
    st.session_state.df = df
    return df


st.title("🏥 City Hospital: Live Training Lab")
st.caption("Breast Cancer Wisconsin (Diagnostic) · PyTorch · educational demo, never for real diagnosis")

df = get_df()
if "prep" not in st.session_state:
    st.session_state.prep = prepare(df)
prep = st.session_state.prep
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)
st.session_state.setdefault("exp", None)

with st.sidebar:
    st.header("Dataset")
    sp = prep.split
    st.write(f"**{len(df)}** patients · **{len(FEATURE_NAMES)}** features")
    st.write(f"Train **{len(sp['y_train'])}** · Validation **{len(sp['y_val'])}** · Test **{len(sp['y_test'])}**")
    st.caption("Experiments here use train + validation only. The test set is kept for the final model (`python -m src.train`).")
    if st.button("Clear run history"):
        st.session_state.history, st.session_state.last, st.session_state.exp = [], None, None
        st.rerun()

tab_data, tab_lab, tab_exp, tab_pred = st.tabs(["📊 Data explorer", "🧪 Training lab", "🏆 Experiments", "🔍 Predict"])

# ---------------------------------------------------------------- data explorer
with tab_data:
    counts = df[TARGET].value_counts().reindex(["B", "M"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(df))
    c2.metric("Benign", int(counts["B"]))
    c3.metric("Malignant", int(counts["M"]), f"{counts['M'] / len(df):.1%} of all")
    c4.metric("Missing values", int(df[FEATURE_NAMES].isnull().sum().sum()))

    left, right = st.columns(2)
    with left:
        st.subheader("Distribution of one feature")
        feat = st.selectbox("Feature", FEATURE_NAMES, index=FEATURE_NAMES.index("radius_mean"))
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.histplot(data=df, x=feat, hue=TARGET, kde=True, palette=PAL, element="step", stat="density",
                     common_norm=False, ax=ax)
        st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader("Correlation with malignant")
        corr = df[FEATURE_NAMES].corrwith((df[TARGET] == "M").astype(int)).sort_values()
        fig, ax = plt.subplots(figsize=(6, 6.4))
        ax.barh(corr.index, corr.values, color=[PAL["B"] if v < 0 else PAL["M"] for v in corr.values])
        ax.tick_params(labelsize=7)
        st.pyplot(fig); plt.close(fig)

    st.subheader("Average of each mean feature per class")
    st.dataframe(df.groupby(TARGET)[MEAN_COLS].mean().T.round(3), width="stretch")
    st.subheader("Scale differences (why scaling is needed)")
    rng = (df[FEATURE_NAMES].max() - df[FEATURE_NAMES].min()).sort_values()
    st.write(f"`{rng.index[-1]}` spans **{rng.iloc[-1]:,.1f}**, while `{rng.index[0]}` spans only **{rng.iloc[0]:.4f}**. "
             "A neural network trains badly when inputs live on such different scales.")
    st.dataframe(df.head(20), width="stretch", hide_index=True)

# ---------------------------------------------------------------- training lab
with tab_lab:
    st.subheader("Training configuration")
    k1, k2, k3 = st.columns(3)
    hidden = k1.selectbox("Hidden neurons", HIDDEN_OPTIONS, index=HIDDEN_OPTIONS.index(16))
    dropout = k1.slider("Dropout", 0.0, 0.8, 0.30, 0.05)
    lr = k2.select_slider("Learning rate", options=LR_OPTIONS, value=1e-3, format_func=lambda v: f"{v:g}")
    wd = k2.slider("Weight decay", 0.0, 0.2, 0.05, 0.005)
    patience = k3.slider("Early stopping patience (0 = off)", 0, 50, 20)
    max_epochs = k3.slider("Maximum epochs", 10, 500, 200, 10)
    scaling = k3.checkbox("Feature scaling", value=True)

    if st.button("▶ Train model", type="primary"):
        n = len(st.session_state.history) + 1
        cfg = make_cfg(hidden=hidden, dropout=dropout, lr=float(lr), weight_decay=wd,
                       patience=(patience or None), max_epochs=max_epochs, scaling=scaling)
        bar, chart = st.progress(0.0, text="Training..."), st.empty()

        def live(epoch, history, total):
            if epoch == 1 or epoch % 3 == 0:
                bar.progress(min(epoch / total, 1.0), text=f"Epoch {epoch}/{total}")
                chart.line_chart(pd.DataFrame(history).rename(columns={"train_loss": "Train", "val_loss": "Validation"}),
                                 height=300)

        out = run_experiment(f"Dash-{n}", cfg, prep, progress_cb=live)
        bar.empty(); chart.empty()
        st.session_state.history.append(out["row"])
        st.session_state.last = out

    last = st.session_state.last
    if last:
        row, va = last["row"], last["val"]
        st.divider()
        st.subheader(f"Result: {row['Experiment ID']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Training loss", f"{row['Training Loss']:.3f}")
        m2.metric("Validation loss", f"{row['Validation Loss']:.3f}",
                  f"{row['Validation Loss'] - row['Training Loss']:+.3f} vs train", delta_color="inverse")
        m3.metric("Accuracy", f"{row['Accuracy']:.1%}")
        m4.metric("Recall", f"{row['Recall']:.1%}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Cancers detected", f"{va['tp']} / {va['tp'] + va['fn']}")
        m6.metric("Cancers missed", va["fn"])
        m7.metric("False alarms", va["fp"])
        m8.metric("Epochs run / best", f"{row['Epochs Run']} / {row['Best Epoch']}")

        note = {"Healthy": (st.success, "Training and validation loss agree, with no sign of memorising."),
                "Overfitting": (st.warning, "Validation loss got worse than its earlier best. Try more dropout, "
                                            "weight decay, early stopping or a smaller model."),
                "Underfitting": (st.error, "Training loss is still above 0.30. Try more epochs, a larger "
                                           "learning rate, or turn feature scaling on.")}
        show, text = note[row["Diagnosis"]]
        show(f"**{row['Diagnosis']}.** {text}")

        pa, pb = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(7, 3.8))
        loss_curve(ax, last["history"], "Training vs validation loss", row["Best Epoch"])
        pa.pyplot(fig); plt.close(fig)
        pb.pyplot(confusion_figure([[va["tn"], va["fp"]], [va["fn"], va["tp"]]], "Confusion matrix (validation)"))
        plt.close("all")

    if st.session_state.history:
        st.subheader("Run history")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, width="stretch", hide_index=True)
        st.download_button("Download history (CSV)", hist_df.to_csv(index=False).encode(),
                           file_name="experiment_history.csv", mime="text/csv")

# ---------------------------------------------------------------- experiments
with tab_exp:
    st.subheader("The six experiments")
    st.caption("Baseline, overfitting, regularization, early stopping, small model, no scaling. "
               "Each is trained on the training set and scored on the validation set.")
    if st.button("▶ Run all six experiments"):
        plan = experiment_plan(200)
        bar = st.progress(0.0, text="Starting...")
        rows, hists = [], {}
        for i, (name, cfg) in enumerate(plan):
            bar.progress(i / len(plan), text=f"Running {name}...")
            out = run_experiment(name, cfg, prep)
            rows.append(out["row"]); hists[name] = out["history"]
        bar.empty()
        st.session_state.exp = {"df": pd.DataFrame(rows), "hists": hists}

    exp = st.session_state.exp
    if exp:
        res = exp["df"]
        st.dataframe(res, width="stretch", hide_index=True)
        fig = loss_curve_grid(exp["hists"], dict(zip(res["Experiment ID"], res["Diagnosis"])))
        st.pyplot(fig); plt.close(fig)
        fig = metrics_bar_figure(res)
        st.pyplot(fig); plt.close(fig)
        st.download_button("Download experiments (CSV)", res.to_csv(index=False).encode(),
                           file_name="experiments.csv", mime="text/csv")
        st.markdown("**Questions for your report:** Which experiment has the biggest train/validation gap? Did dropout + weight decay "
                    "shrink it? Did early stopping help? Did the small model generalise better? What did removing scaling do? "
                    "Which experiment missed the fewest cancers, and is it the one with the best accuracy?")

# ---------------------------------------------------------------- predict
with tab_pred:
    st.subheader("Prediction demo (educational only)")
    try:
        bundle = load_bundle()
    except FileNotFoundError:
        st.info("Train the final model first. In a terminal run:  `python -m src.train`")
    else:
        tm = bundle["meta"]["test_metrics"]
        st.caption(f"Final model: **{bundle['meta']['experiment']}** · test recall {tm['recall']:.1%} · "
                   f"test accuracy {tm['accuracy']:.1%} (evaluated once on held-out patients)")
        thr = st.slider("Decision threshold for 'Malignant'", 0.05, 0.95, bundle["threshold"], 0.05,
                        help="Lower = catches more cancers but gives more false alarms")
        if SAMPLE_PATH.exists():
            sample = pd.read_csv(SAMPLE_PATH)
            scored = score_dataframe(sample, thr, bundle)
            pick = st.selectbox("Choose a held-out test patient", scored["id"].tolist())
            r = scored[scored["id"] == pick].iloc[0]
            a, b, c = st.columns(3)
            a.metric("Actual diagnosis", r["actual"])
            b.metric("Predicted diagnosis", r["prediction"])
            c.metric("Malignant probability", f"{r['p_malignant']:.1%}")
            (st.success if r["correct"] else st.error)("Prediction result: " + ("Correct" if r["correct"] else "WRONG"))
            st.progress(float(r["p_malignant"]), text=f"P(malignant) = {r['p_malignant']:.1%}  |  P(benign) = {r['p_benign']:.1%}")
            cm = confusion_matrix(scored["actual"] == "Malignant", scored["prediction"] == "Malignant", labels=[False, True])
            st.write(f"On all {len(scored)} held-out patients at threshold {thr:.2f}: "
                     f"**{cm[1, 1]} / {cm[1, 0] + cm[1, 1]}** cancers found, **{cm[1, 0]}** missed, **{cm[0, 1]}** false alarms.")
            with st.expander("All held-out patients"):
                st.dataframe(scored, width="stretch", hide_index=True)
        up = st.file_uploader("…or upload your own CSV (30 feature columns)", type="csv", key="pred_up")
        if up is not None:
            try:
                res = score_dataframe(pd.read_csv(up), thr, bundle)
                st.dataframe(res, width="stretch", hide_index=True)
                st.download_button("Download predictions (CSV)", res.to_csv(index=False).encode(),
                                   file_name="predictions.csv", mime="text/csv")
            except ValueError as e:
                st.error(f"Cannot score this file: {e}")
    st.caption("⚠️ Educational demo. This model must never be used to diagnose real patients.")
