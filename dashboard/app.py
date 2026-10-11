"""
Diabetes Risk Assessment in Young Adults - model comparison dashboard.
Cloud Computing Lab, Question 4.

Run with:  streamlit run app.py
Expected files in the same folder as this script:
  results.csv, cv_results.csv, tuning_results.csv, best_models.joblib,
  test_predictions.joblib, train.csv, test.csv   (timing.csv is optional)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(page_title="Diabetes Risk in Young Adults", page_icon="🩺", layout="wide")

BASE = Path(__file__).parent
TARGET = "Diabetes_binary"
MODELS_ORDER = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]
REQUIRED = ["results.csv", "cv_results.csv", "tuning_results.csv", "best_models.joblib",
            "test_predictions.joblib", "train.csv", "test.csv"]
COLORS = {"Logistic Regression": "#1f77b4", "KNN": "#ff7f0e", "Decision Tree": "#2ca02c",
          "Random Forest": "#d62728", "SVM": "#9467bd"}
PX_TEMPLATE = "plotly_white"

missing = [f for f in REQUIRED if not (BASE / f).exists()]
if missing:
    st.error("These files are missing from the app folder: " + ", ".join(missing))
    st.info("Copy them from your notebook folder next to app.py and reload the page.")
    st.stop()


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_tables():
    results = pd.read_csv(BASE / "results.csv")
    cv = pd.read_csv(BASE / "cv_results.csv")
    tuning = pd.read_csv(BASE / "tuning_results.csv")
    train = pd.read_csv(BASE / "train.csv")
    test = pd.read_csv(BASE / "test.csv")
    timing = pd.read_csv(BASE / "timing.csv") if (BASE / "timing.csv").exists() else None
    return results, cv, tuning, train, test, timing


@st.cache_resource
def load_models():
    models = joblib.load(BASE / "best_models.joblib")
    saved = joblib.load(BASE / "test_predictions.joblib")
    return models, saved["preds"], saved["probs"]


results, cv, tuning, train, test, timing = load_tables()
models, preds, scores = load_models()

FEATURES = [c for c in train.columns if c != TARGET]
full = pd.concat([train, test], ignore_index=True)
y_test = test[TARGET].astype(int).to_numpy()
order = [m for m in MODELS_ORDER if m in results["Model"].tolist()]
test_pos_rate = y_test.mean()


def native_threshold(name):
    """Default decision point: 0 for SVM decision scores, 0.5 for probabilities."""
    return 0.0 if name == "SVM" else 0.5


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def parse_pm(series):
    """Turn strings such as '0.780 ± 0.004' into (mean, std) float columns."""
    parts = series.astype(str).str.split("±", expand=True)
    return parts[0].str.strip().astype(float), parts[1].str.strip().astype(float)


def style(fig, height=None):
    fig.update_layout(template=PX_TEMPLATE, margin=dict(l=10, r=10, t=60, b=10))
    if height:
        fig.update_layout(height=height)
    return fig


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🩺 Diabetes risk dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Dataset", "Model comparison", "ROC curves & confusion matrices",
     "Feature importance", "Risk predictor (demo)"],
)
st.sidebar.markdown("---")

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
if page == "Overview":
    st.title("Diabetes Risk Assessment in Young Adults")
    st.write(
        "This dashboard compares five machine learning classifiers (Logistic Regression, KNN, "
        "Decision Tree, Random Forest, SVM) that predict prediabetes/diabetes from survey "
        "answers of respondents aged 18-34 in the CDC Behavioral Risk Factor Surveillance "
        "System (BRFSS) 2015 data."
    )
    n_pos = int(full[TARGET].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents (ages 18-34)", f"{len(full):,}")
    c2.metric("Positive cases", f"{n_pos:,}", f"{n_pos / len(full):.2%} of respondents", delta_color="off")
    c3.metric("Training / test rows", f"{len(train):,} / {len(test):,}")
    c4.metric("Features", len(FEATURES))

    best_auc = results.loc[results["ROC-AUC"].idxmax()]
    best_rec = results.loc[results["Recall"].idxmax()]
    best_f1 = results.loc[results["F1"].idxmax()]
    st.subheader("Headline results (held-out test set)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest ROC-AUC", f"{best_auc['ROC-AUC']:.3f}", best_auc["Model"], delta_color="off")
    c2.metric("Highest recall", f"{best_rec['Recall']:.3f}", best_rec["Model"], delta_color="off")
    c3.metric("Highest F1-score", f"{best_f1['F1']:.3f}", best_f1["Model"], delta_color="off")

    st.info(
        "**Why recall matters.** Only about 2% of respondents are positive, so accuracy is "
        "misleading: a model that always says 'no diabetes' would be about "
        f"{1 - test_pos_rate:.1%} accurate and find nobody. In a health-risk setting, missing a "
        "true case (false negative) is usually costlier than an extra check (false positive), so "
        "recall is watched alongside precision and ROC-AUC."
    )
    st.caption("Use the sidebar to explore the dataset, compare models, and try the risk predictor.")

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
elif page == "Dataset":
    st.title("Dataset overview")
    st.caption("Young-adult subset (BRFSS age categories 1-3, ages 18-34). Training and test rows combined.")

    # Class balance
    counts = full[TARGET].map({0: "No diabetes", 1: "Prediabetes / diabetes"}).value_counts().reset_index()
    counts.columns = ["Class", "Respondents"]
    counts["Share"] = (counts["Respondents"] / counts["Respondents"].sum() * 100).round(2)
    c1, c2 = st.columns([2, 3])
    with c1:
        fig = px.bar(counts, x="Class", y="Respondents", color="Class", text="Respondents",
                     title="Class distribution",
                     color_discrete_sequence=["#1f77b4", "#d62728"])
        fig.update_layout(showlegend=False, yaxis_title="Number of respondents", xaxis_title="")
        st.plotly_chart(style(fig, 380))
    with c2:
        st.dataframe(counts.rename(columns={"Share": "Share (%)"}), hide_index=True)
        split = pd.DataFrame({
            "Set": ["Training", "Test"],
            "Rows": [len(train), len(test)],
            "Positives": [int(train[TARGET].sum()), int(test[TARGET].sum())],
        })
        split["Positive rate (%)"] = (split["Positives"] / split["Rows"] * 100).round(2)
        st.write("Stratified 80/20 split")
        st.dataframe(split, hide_index=True)

    # Feature distribution by class
    st.subheader("Feature distributions")
    feat = st.selectbox("Choose a feature", FEATURES, index=FEATURES.index("BMI") if "BMI" in FEATURES else 0)
    data = full.copy()
    data["Class"] = data[TARGET].map({0: "No diabetes", 1: "Prediabetes / diabetes"})
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(data, x=feat, color="Class", barmode="overlay", opacity=0.65,
                           histnorm="percent", title=f"Distribution of {feat} by class",
                           color_discrete_sequence=["#1f77b4", "#d62728"])
        fig.update_layout(yaxis_title="Share of respondents in class (%)", xaxis_title=feat)
        st.plotly_chart(style(fig, 400))
    with c2:
        if feat == "BMI":
            grp = pd.cut(full[feat], bins=[0, 18.5, 25, 30, 35, 40, 200],
                         labels=["<18.5", "18.5-24.9", "25-29.9", "30-34.9", "35-39.9", "40+"])
        elif feat in ("MentHlth", "PhysHlth"):
            grp = pd.cut(full[feat], bins=[-1, 0, 5, 15, 29, 30],
                         labels=["0 days", "1-5", "6-15", "16-29", "30 days"])
        else:
            grp = full[feat].astype(int).astype(str)
        rate = full.groupby(grp, observed=True)[TARGET].agg(["mean", "size"]).reset_index()
        rate.columns = [feat, "Rate", "Respondents"]
        rate["Rate"] = rate["Rate"] * 100
        fig = px.bar(rate, x=feat, y="Rate", text=rate["Respondents"].map(lambda n: f"n={n:,}"),
                     title=f"Diabetes/prediabetes rate by {feat}")
        fig.add_hline(y=full[TARGET].mean() * 100, line_dash="dash", line_color="red",
                      annotation_text="Overall rate")
        fig.update_layout(yaxis_title="Positive rate (%)", xaxis_title=feat)
        st.plotly_chart(style(fig, 400))

    # Correlations
    st.subheader("Correlations")
    corr = full.corr()
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-0.5, zmax=0.5,
                        title="Correlation heatmap (colour scale clipped to -0.5 ... 0.5)",
                        labels=dict(color="Correlation"))
        st.plotly_chart(style(fig, 650))
    with c2:
        ct = corr[TARGET].drop(TARGET).sort_values()
        fig = px.bar(ct, orientation="h", title="Correlation of each feature with the target",
                     labels=dict(value="Correlation", index="Feature"))
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 650))

    with st.expander("Show a sample of the data"):
        st.dataframe(full.head(50))

# ----------------------------------------------------------------------------
# Model comparison
# ----------------------------------------------------------------------------
elif page == "Model comparison":
    st.title("Model comparison")

    st.subheader("Test-set metrics")
    metric_options = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    chosen = st.multiselect("Metrics to show", metric_options,
                            default=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    if chosen:
        long = results.melt(id_vars="Model", value_vars=chosen, var_name="Metric", value_name="Score")
        long["Model"] = pd.Categorical(long["Model"], categories=order, ordered=True)
        fig = px.bar(long.sort_values("Model"), x="Model", y="Score", color="Metric", barmode="group",
                     text_auto=".2f", title="Performance of the five classifiers on the held-out test set")
        fig.update_layout(yaxis_range=[0, 1.05], yaxis_title="Score (0-1)", xaxis_title="")
        st.plotly_chart(style(fig, 480))
    st.caption(
        f"Baseline for reference: a model that always predicts 'no diabetes' would reach accuracy "
        f"{1 - test_pos_rate:.3f} but recall 0. Accuracy of the tuned models is lower because class "
        "weighting trades accuracy for recall on the rare class."
    )
    st.dataframe(results.round(3), hide_index=True)

    st.subheader("Recall versus precision")
    fig = px.scatter(results, x="Recall", y="Precision", text="Model", color="Model",
                     color_discrete_map=COLORS, size="F1", size_max=28,
                     title="Precision-recall trade-off on the test set (marker size = F1-score)")
    fig.update_traces(textposition="top center")
    fig.update_layout(showlegend=False, xaxis_title="Recall (share of true cases found)",
                      yaxis_title="Precision (share of flagged cases that are true)")
    st.plotly_chart(style(fig, 450))

    st.subheader("Cross-validation on the training set (5-fold, mean ± standard deviation)")
    cv_metrics = {"roc_auc": "ROC-AUC", "recall": "Recall", "f1": "F1", "precision": "Precision",
                  "accuracy": "Accuracy"}
    cv_metrics = {k: v for k, v in cv_metrics.items() if k in cv.columns}
    key = st.selectbox("Metric", list(cv_metrics), format_func=lambda k: cv_metrics[k])
    mean, std = parse_pm(cv[key])
    cvp = pd.DataFrame({"Model": cv["Model"], "Mean": mean, "Std": std})
    fig = go.Figure(go.Bar(x=cvp["Model"], y=cvp["Mean"], error_y=dict(type="data", array=cvp["Std"]),
                           marker_color=[COLORS.get(m, "#888") for m in cvp["Model"]],
                           text=cvp["Mean"].round(3), textposition="outside"))
    fig.update_layout(title=f"Cross-validated {cv_metrics[key]} (error bars = ±1 standard deviation)",
                      yaxis_title=cv_metrics[key], yaxis_range=[0, 1.05])
    st.plotly_chart(style(fig, 430))
    st.dataframe(cv, hide_index=True)

    st.subheader("Hyperparameter tuning")
    st.caption("Grid search with ROC-AUC scoring (5-fold; 3-fold for the SVM).")
    st.dataframe(tuning, hide_index=True)

    if timing is not None and timing.shape[1] >= 3:
        st.subheader("Training and prediction time")
        tcols = timing.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(timing, x=tcols[0], y=tcols[1], title="Training time", text_auto=".2f",
                         labels={tcols[1]: tcols[1], tcols[0]: ""})
            st.plotly_chart(style(fig, 380))
        with c2:
            fig = px.bar(timing, x=tcols[0], y=tcols[2], title="Prediction time (test set)", text_auto=".3f",
                         labels={tcols[2]: tcols[2], tcols[0]: ""})
            st.plotly_chart(style(fig, 380))

# ----------------------------------------------------------------------------
# ROC and confusion matrices
# ----------------------------------------------------------------------------
elif page == "ROC curves & confusion matrices":
    st.title("ROC curves and confusion matrices")

    st.subheader("ROC curves")
    shown = st.multiselect("Models", order, default=order)
    fig = go.Figure()
    for name in shown:
        fpr, tpr, _ = roc_curve(y_test, np.asarray(scores[name]))
        auc = roc_auc_score(y_test, np.asarray(scores[name]))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC = {auc:.3f})",
                                 line=dict(color=COLORS.get(name), width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color="black", dash="dash")))
    fig.update_layout(title="ROC curves on the held-out test set",
                      xaxis_title="False positive rate", yaxis_title="True positive rate (recall)",
                      legend=dict(x=0.55, y=0.05))
    st.plotly_chart(style(fig, 560))

    st.subheader("Confusion matrix")
    name = st.selectbox("Choose a model", order, index=order.index("Random Forest") if "Random Forest" in order else 0)
    cm = confusion_matrix(y_test, np.asarray(preds[name]))
    tn, fp, fn, tp = cm.ravel()
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted: no diabetes", "Predicted: diabetes"],
                        y=["Actual: no diabetes", "Actual: diabetes"],
                        title=f"Confusion matrix - {name} (test set)")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, 430))
    with c2:
        recall = tp / (tp + fn) if (tp + fn) else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        st.metric("Recall (cases found)", f"{recall:.3f}", f"{tp} of {tp + fn} true cases", delta_color="off")
        st.metric("Precision (flags that are real)", f"{precision:.3f}", f"{tp} of {tp + fp} flagged", delta_color="off")
        st.metric("Specificity (non-cases cleared)", f"{specificity:.3f}", f"{tn:,} of {tn + fp:,}", delta_color="off")
        st.write(f"**Missed cases:** {fn}  |  **False alarms:** {fp:,}")

    with st.expander("What-if: change the decision threshold (exploration only)"):
        st.caption(
            "The study used each model's default decision rule. This slider only shows how the trade-off "
            "between missed cases and false alarms would move; the threshold was not tuned or validated."
        )
        sc = np.asarray(scores[name], dtype=float)
        lo, hi = float(np.round(sc.min(), 3)), float(np.round(sc.max(), 3))
        default = float(np.clip(native_threshold(name), lo, hi))
        step = max((hi - lo) / 200, 0.001)
        thr = st.slider("Decision threshold on the model score", lo, hi, default, step=step)
        pred_t = (sc >= thr).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_test, pred_t, labels=[0, 1]).ravel()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall", f"{tp2 / (tp2 + fn2):.3f}" if (tp2 + fn2) else "n/a")
        c2.metric("Precision", f"{tp2 / (tp2 + fp2):.3f}" if (tp2 + fp2) else "n/a")
        c3.metric("Missed cases", f"{fn2}")
        c4.metric("False alarms", f"{fp2:,}")

# ----------------------------------------------------------------------------
# Feature importance
# ----------------------------------------------------------------------------
elif page == "Feature importance":
    st.title("Feature importance")
    options = []
    if "Random Forest" in models:
        options.append("Random Forest (impurity importance)")
    if "Decision Tree" in models:
        options.append("Decision Tree (impurity importance)")
    if "Logistic Regression" in models:
        options.append("Logistic Regression (standardised coefficient)")
    choice = st.selectbox("Model", options)

    if choice.startswith("Random Forest"):
        vals = models["Random Forest"].feature_importances_
        xlabel = "Importance"
    elif choice.startswith("Decision Tree"):
        vals = models["Decision Tree"].feature_importances_
        xlabel = "Importance"
    else:
        vals = models["Logistic Regression"].named_steps["clf"].coef_[0]
        xlabel = "Coefficient (per 1 standard deviation of the feature)"

    imp = pd.DataFrame({"Feature": FEATURES, "Value": np.asarray(vals, dtype=float)}).sort_values("Value")
    if choice.startswith("Logistic"):
        imp["Direction"] = np.where(imp["Value"] >= 0, "Raises predicted risk", "Lowers predicted risk")
        fig = px.bar(imp, x="Value", y="Feature", orientation="h", color="Direction",
                     color_discrete_map={"Raises predicted risk": "#d62728", "Lowers predicted risk": "#1f77b4"},
                     title=choice)
    else:
        fig = px.bar(imp, x="Value", y="Feature", orientation="h", title=choice,
                     color_discrete_sequence=["#0F8B8D"])
    fig.update_layout(xaxis_title=xlabel, yaxis_title="")
    st.plotly_chart(style(fig, 650))
    st.caption(
        "Impurity-based importances tend to favour features with many distinct values (such as BMI), so "
        "read them as a rough ranking. Logistic Regression coefficients show direction as well as strength "
        "because the features were standardised inside the model pipeline."
    )

# ----------------------------------------------------------------------------
# Risk predictor
# ----------------------------------------------------------------------------
else:
    st.title("Risk predictor (demonstration)")
    st.warning(
        "Demonstration of the study's models only. This is not medical advice or a diagnosis. The models "
        "were trained on self-reported survey data, have low precision (most flagged people do not have "
        "diabetes), and their scores are not real-world probabilities."
    )

    yes_no = ["No", "Yes"]
    GEN = {1: "1 - Excellent", 2: "2 - Very good", 3: "3 - Good", 4: "4 - Fair", 5: "5 - Poor"}
    EDU = {1: "1 - Never attended / kindergarten only", 2: "2 - Grades 1-8", 3: "3 - Grades 9-11",
           4: "4 - Grade 12 or GED", 5: "5 - College 1-3 years", 6: "6 - College 4 years or more"}
    INC = {1: "1 - Under $10,000", 2: "2 - $10,000 to under $15,000", 3: "3 - $15,000 to under $20,000",
           4: "4 - $20,000 to under $25,000", 5: "5 - $25,000 to under $35,000",
           6: "6 - $35,000 to under $50,000", 7: "7 - $50,000 to under $75,000", 8: "8 - $75,000 or more"}
    AGE = {1: "18-24", 2: "25-29", 3: "30-34"}

    binary_q = {
        "HighBP": ("High blood pressure", 0), "HighChol": ("High cholesterol", 0),
        "CholCheck": ("Cholesterol check in past 5 years", 1), "Smoker": ("Smoked 100+ cigarettes in lifetime", 0),
        "Stroke": ("Ever had a stroke", 0), "HeartDiseaseorAttack": ("Coronary heart disease or heart attack", 0),
        "PhysActivity": ("Physical activity in past 30 days", 1), "Fruits": ("Eats fruit daily", 1),
        "Veggies": ("Eats vegetables daily", 1), "HvyAlcoholConsump": ("Heavy alcohol consumption", 0),
        "AnyHealthcare": ("Has health care coverage", 1), "NoDocbcCost": ("Could not see a doctor because of cost", 0),
        "DiffWalk": ("Serious difficulty walking or climbing stairs", 0),
    }

    vals = {}
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Body and health**")
        vals["BMI"] = st.number_input("BMI (kg/m²)", min_value=12.0, max_value=98.0, value=25.0, step=0.5)
        vals["GenHlth"] = st.selectbox("General health", list(GEN), index=1, format_func=GEN.get)
        vals["MentHlth"] = st.slider("Days of poor mental health (past 30)", 0, 30, 0)
        vals["PhysHlth"] = st.slider("Days of poor physical health (past 30)", 0, 30, 0)
        for f in ["HighBP", "HighChol", "CholCheck"]:
            label, d = binary_q[f]
            vals[f] = yes_no.index(st.selectbox(label, yes_no, index=d, key=f))
    with col2:
        st.markdown("**Conditions and lifestyle**")
        for f in ["Stroke", "HeartDiseaseorAttack", "DiffWalk", "Smoker", "HvyAlcoholConsump",
                  "PhysActivity", "Fruits", "Veggies"]:
            label, d = binary_q[f]
            vals[f] = yes_no.index(st.selectbox(label, yes_no, index=d, key=f))
    with col3:
        st.markdown("**Background**")
        vals["Age"] = st.selectbox("Age group", list(AGE), index=2, format_func=AGE.get)
        vals["Sex"] = ["Female", "Male"].index(st.selectbox("Sex", ["Female", "Male"]))
        vals["Education"] = st.selectbox("Education", list(EDU), index=5, format_func=EDU.get)
        vals["Income"] = st.selectbox("Income", list(INC), index=7, format_func=INC.get)
        for f in ["AnyHealthcare", "NoDocbcCost"]:
            label, d = binary_q[f]
            vals[f] = yes_no.index(st.selectbox(label, yes_no, index=d, key=f))

    row = pd.DataFrame([[vals[f] for f in FEATURES]], columns=FEATURES)
    out = []
    for name in order:
        model = models[name]
        flagged = int(model.predict(row)[0]) == 1
        score = float(get_score(model, row)[0])
        pct = float((np.asarray(scores[name], dtype=float) < score).mean() * 100)
        out.append({"Model": name, "Flagged as higher risk": "Yes" if flagged else "No",
                    "Model score": round(score, 3), "Percentile among test respondents": round(pct, 1)})
    res = pd.DataFrame(out)

    st.subheader("Result")
    n_flag = int((res["Flagged as higher risk"] == "Yes").sum())
    st.metric("Models flagging this profile as higher risk", f"{n_flag} of {len(res)}")
    fig = px.bar(res, x="Model", y="Percentile among test respondents", color="Model",
                 color_discrete_map=COLORS, text="Percentile among test respondents",
                 title="How this profile ranks among the test respondents (higher = more risk-like)")
    fig.update_layout(showlegend=False, yaxis_range=[0, 105], yaxis_title="Percentile (%)", xaxis_title="")
    st.plotly_chart(style(fig, 400))
    st.dataframe(res, hide_index=True)
    st.caption(
        "Model score: probability-like output (0-1) for Logistic Regression, KNN, Decision Tree and Random "
        "Forest; decision-function value for the SVM. Percentile = share of test respondents with a lower "
        "score from the same model."
    )