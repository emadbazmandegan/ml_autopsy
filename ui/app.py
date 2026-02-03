"""
Model Autopsy UI
Streamlit-based frontend for ML model diagnostics
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.metrics import compute_classification_metrics, compute_auc_metrics
from engine.slicing import generate_all_slices, rank_slices
from engine.clustering import run_error_clustering
from engine.attribution import run_attribution_analysis
from engine.blindspots import run_blindspot_analysis

# Page configuration
# Page configuration
st.set_page_config(
    page_title="Model Autopsy",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Model Autopsy</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Explain <em>why</em> machine learning models fail — not just how accurate they are.</div>',
    unsafe_allow_html=True
)

# Initialize session state
if "results" not in st.session_state:
    st.session_state["results"] = None

# Sidebar
with st.sidebar:
    st.header("Navigation")
    
    # Define pages
    pages = ["Upload & Configure", "Results Dashboard", "Compare Models", "Report"]
    
    # Ensure current_page is in state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = pages[0]
        
    # Check for programmatic switch (force widget update)
    if st.session_state.get("programmatic_switch", False):
        st.session_state["nav_radio_widget"] = st.session_state["current_page"]
        st.session_state["programmatic_switch"] = False
        
    # Render widget
    selected_page = st.radio(
        "Select Page",
        pages,
        key="nav_radio_widget"
    )
    
    # Sync user choice to state
    if selected_page != st.session_state["current_page"]:
        st.session_state["current_page"] = selected_page
        st.rerun()
        
    page = st.session_state["current_page"]
    
    st.divider()
    st.caption("Model Autopsy v0.1.0")

# Main content based on page selection
if page == "Upload & Configure":
    st.header("Upload Your Data", divider="gray")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Required Files")
        
        dataset_file = st.file_uploader(
            "Dataset (CSV)",
            type=["csv"],
            help="CSV file containing feature columns"
        )
        
        predictions_file = st.file_uploader(
            "Predictions (CSV)",
            type=["csv"],
            help="CSV with y_pred (hard labels) or y_score (probabilities)"
        )
    
    with col2:
        st.subheader("Optional Files")
        
        labels_file = st.file_uploader(
            "Labels (CSV) - if not in dataset",
            type=["csv"],
            help="CSV with y_true column (if not included in dataset)"
        )
        
        st.subheader("Configuration")
        
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Threshold for converting y_score to y_pred"
        )
    
    st.divider()
    
    # Demo button
    col_demo, col_run = st.columns(2)
    
    with col_demo:
        if st.button("Try Demo (UCI Adult Income)", use_container_width=True, icon=":material/play_circle:"):
            st.session_state["demo_mode"] = True
            st.success("UCI Adult dataset loaded! Click 'Run Autopsy' to analyze.", icon=":material/check_circle:")
            st.caption(
                "**Dataset:** [UCI Adult Income](https://archive.ics.uci.edu/dataset/2/adult) "
                "(1994 Census, 100 samples)  \n"
                "**Models:** LogisticRegression (Baseline) vs RandomForest (Comparison)"
            )
    
    with col_run:
        if st.button("Run Autopsy", type="primary", use_container_width=True, icon=":material/science:"):
            is_demo = st.session_state.get("demo_mode", False)
            
            if is_demo:
                with st.spinner("Running model autopsy on demo data..."):
                    # Load demo data
                    demo_dir = Path(__file__).parent.parent / "examples" / "demo"
                    dataset_df = pd.read_csv(demo_dir / "adult_dataset.csv")
                    preds_df = pd.read_csv(demo_dir / "adult_predictions.csv")
                    
                    # Auto-load comparison models for the Compare page
                    st.session_state["comp_file_a"] = str(demo_dir / "adult_predictions.csv")
                    st.session_state["comp_file_b"] = str(demo_dir / "adult_predictions_rf.csv")
                    
                    # Build canonical dataframe
                    canonical = dataset_df.copy()
                    canonical["y_true"] = preds_df["y_true"]
                    canonical["y_pred"] = preds_df["y_pred"]
                    canonical["y_score"] = preds_df["y_score"]
                    canonical["is_error"] = canonical["y_true"] != canonical["y_pred"]
                    canonical["error_type"] = np.where(
                        ~canonical["is_error"], 
                        np.where(canonical["y_true"] == 1, "TP", "TN"),
                        np.where(canonical["y_true"] == 1, "FN", "FP")
                    )
                    
                    # Run analysis
                    metrics = compute_classification_metrics(
                        canonical["y_true"], canonical["y_pred"]
                    )
                    
                    auc_metrics = compute_auc_metrics(
                        canonical["y_true"], canonical["y_score"]
                    )
                    metrics.update(auc_metrics)
                    
                    # Slicing
                    feature_cols = ["age", "workclass", "education", "occupation", "sex", "hours_per_week", "race"]
                    slices = generate_all_slices(canonical, feature_cols)
                    ranked_slices = rank_slices(slices, min_support=5, top_n=10)
                    
                    # Clustering
                    errors_df, profiles, X_2d = run_error_clustering(
                        canonical, feature_cols=feature_cols, n_clusters=3
                    )
                    
                    # Attribution
                    global_imp, error_imp, delta = run_attribution_analysis(
                        canonical, feature_cols=feature_cols, n_repeats=5
                    )
                    
                    # Blind spots
                    confident_wrong, divergence = run_blindspot_analysis(canonical)
                    
                    # Store results
                    st.session_state["results"] = {
                        "canonical": canonical,
                        "metrics": metrics,
                        "feature_cols": feature_cols,
                        "slices": ranked_slices,
                        "clusters": {"df": errors_df, "profiles": profiles, "X_2d": X_2d},
                        "attribution": {"global": global_imp, "delta": delta},
                        "blindspots": {"cw": confident_wrong, "divergence": divergence},
                    }
                    
                    st.success("Analysis complete! Switching to dashboard...", icon=":material/verified:")
                    
                    # Smooth transition: Wait briefly so user sees the success message
                    time.sleep(0.8)
                    
                    # Switch to Results Dashboard (Update internal state only)
                    # Widget will update automatically on rerun because index is derived from current_page
                    st.session_state["current_page"] = "Results Dashboard"
                    st.session_state["programmatic_switch"] = True
                    st.rerun()
                    
            elif dataset_file is None or predictions_file is None:
                st.error("Please upload both dataset and predictions files.")
            else:
                # Custom file analysis
                with st.spinner("Validating and processing your files..."):
                    try:
                        from engine.ingest import ingest_uploaded_files, ValidationError
                        
                        # Ingest uploaded files
                        canonical, config, feature_cols = ingest_uploaded_files(
                            dataset_file=dataset_file,
                            predictions_file=predictions_file,
                            labels_file=labels_file,
                            threshold=threshold,
                        )
                        
                        st.success(
                            f"Files validated: {config['n_rows']} samples, "
                            f"{config['n_features']} features, task type: {config['task_type']}",
                            icon=":material/check_circle:"
                        )
                        
                    except ValidationError as e:
                        st.error(f"Validation Error: {str(e)}")
                        st.stop()
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
                        st.stop()
                
                with st.spinner("Running model autopsy..."):
                    # Run analysis (same as demo mode)
                    metrics = compute_classification_metrics(
                        canonical["y_true"], canonical["y_pred"]
                    )
                    
                    # AUC metrics (only if y_score exists)
                    if "y_score" in canonical.columns:
                        auc_metrics = compute_auc_metrics(
                            canonical["y_true"], canonical["y_score"]
                        )
                        metrics.update(auc_metrics)
                    
                    # Slicing - use detected feature columns
                    slices = generate_all_slices(canonical, feature_cols)
                    min_support = max(5, len(canonical) // 20)  # Dynamic min support
                    ranked_slices = rank_slices(slices, min_support=min_support, top_n=10)
                    
                    # Clustering
                    n_clusters = min(3, canonical["is_error"].sum())  # Don't exceed error count
                    n_clusters = max(1, n_clusters)  # At least 1
                    errors_df, profiles, X_2d = run_error_clustering(
                        canonical, feature_cols=feature_cols, n_clusters=n_clusters
                    )
                    
                    # Attribution
                    global_imp, error_imp, delta = run_attribution_analysis(
                        canonical, feature_cols=feature_cols, n_repeats=5
                    )
                    
                    # Blind spots
                    confident_wrong, divergence = run_blindspot_analysis(canonical)
                    
                    # Store results
                    st.session_state["results"] = {
                        "canonical": canonical,
                        "metrics": metrics,
                        "feature_cols": feature_cols,
                        "slices": ranked_slices,
                        "clusters": {"df": errors_df, "profiles": profiles, "X_2d": X_2d},
                        "attribution": {"global": global_imp, "delta": delta},
                        "blindspots": {"cw": confident_wrong, "divergence": divergence},
                    }
                    
                    st.success("Analysis complete! Switching to dashboard...", icon=":material/verified:")
                    
                    # Smooth transition
                    time.sleep(0.8)
                    
                    # Switch to Results Dashboard
                    st.session_state["current_page"] = "Results Dashboard"
                    st.session_state["programmatic_switch"] = True
                    st.rerun()


elif page == "Results Dashboard":
    st.header("Results Dashboard", divider="gray")
    
    results = st.session_state.get("results")
    
    if results is None:
        st.warning("No results yet. Go to 'Upload & Configure' and run an autopsy first.", icon=":material/warning:")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", "Failure Slices", "Error Clusters", "Attribution", "Blind Spots"
        ])
        
        with tab1:
            st.subheader("Model Performance")
            
            canonical = results["canonical"]
            metrics = results["metrics"]
            
            # Check if y_score is available for threshold slider
            has_scores = "y_score" in canonical.columns
            
            if has_scores:
                # Threshold slider (only if we have probability scores)
                st.divider()
                col_slider, col_info = st.columns([3, 1])
                
                with col_slider:
                    new_threshold = st.slider(
                        "Adjust Classification Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        help="Adjust threshold and see how metrics change in real-time"
                    )
                
                # Import threshold module
                from engine.threshold import compute_metrics_at_threshold, compute_flip_samples, find_optimal_threshold
                
                # Compute metrics at current threshold
                live_metrics = compute_metrics_at_threshold(
                    canonical["y_true"].values,
                    canonical["y_score"].values,
                    new_threshold
                )
                
                # Show flip info
                with col_info:
                    flips = compute_flip_samples(canonical["y_score"].values, 0.5, new_threshold)
                    if flips["flip_count"] > 0:
                        st.metric("Samples Flipped", flips["flip_count"], 
                                  delta=f"{flips['flip_rate']:.1%} of data")
                    else:
                        st.metric("Samples Flipped", 0)
                
                st.divider()
                
                # Live metrics
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Accuracy", f"{live_metrics['accuracy']:.1%}")
                col2.metric("Precision", f"{live_metrics['precision']:.1%}")
                col3.metric("Recall", f"{live_metrics['recall']:.1%}")
                col4.metric("F1 Score", f"{live_metrics['f1']:.1%}")
                
                col5, col6, col7, col8 = st.columns(4)
                
                # Original AUC metrics (don't change with threshold)
                col5.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
                col6.metric("AUC-PR", f"{metrics.get('auc_pr', 0):.3f}")
                
                col7.metric("Error Rate", f"{live_metrics['error_rate']:.1%}")
                col8.metric("Total Samples", f"{len(canonical):,}")
                
                # Optimal threshold suggestion
                if st.checkbox("Show optimal threshold"):
                    opt_t, opt_f1 = find_optimal_threshold(
                        canonical["y_true"].values,
                        canonical["y_score"].values
                    )
                    st.info(f"**Optimal threshold for F1:** {opt_t:.2f} (F1 = {opt_f1:.3f})", icon=":material/lightbulb:")
            else:
                # No y_score - show static metrics only
                st.divider()
                st.info("Threshold adjustment not available (no probability scores provided).", icon=":material/info:")
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                col2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
                col3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
                col4.metric("F1 Score", f"{metrics.get('f1', 0):.1%}")
                
                col5, col6, col7, col8 = st.columns(4)
                
                # Handle case where no AUC (no scores)
                col5.metric("AUC-ROC", f"{metrics.get('auc_roc', 'N/A')}" if 'auc_roc' in metrics else "N/A")
                col6.metric("AUC-PR", f"{metrics.get('auc_pr', 'N/A')}" if 'auc_pr' in metrics else "N/A")
                col7.metric("Error Rate", f"{metrics.get('error_rate', 0):.1%}")
                col8.metric("Total Samples", f"{len(canonical):,}")

        
        with tab2:
            st.subheader("Top Failure Slices")
            
            slices = results["slices"]
            if slices:
                # Add selection column
                slices_df = pd.DataFrame(slices)
                
                # Show clickable table
                slice_idx = st.selectbox(
                    "Drill down into slice:", 
                    options=range(len(slices_df)),
                    format_func=lambda i: f"Slice {i+1}: {slices_df.iloc[i]['rule']} (Lift: {slices_df.iloc[i]['lift']:.2f}x)"
                )
                
                # Show slice details
                selected_slice = slices_df.iloc[slice_idx]
                st.info(f"**Rule:** {selected_slice['rule']}")
                
                col_a, col_b = st.columns(2)
                col_a.metric("Error Rate", f"{selected_slice['error_rate']:.1%}")
                col_b.metric("Support", f"{selected_slice['support']} samples")
                
            else:
                st.info("No significant slices found.")
        
        with tab3:
            st.subheader("Error Clusters")
            
            profiles = results["clusters"]["profiles"]
            errors_df = results["clusters"]["df"]
            
            if profiles:
                col_list, col_drill = st.columns([1, 2])
                
                with col_list:
                    selected_cluster_id = st.radio(
                        "Select Cluster",
                        options=[p['cluster_id'] for p in profiles],
                        format_func=lambda x: f"Cluster {x}"
                    )
                
                with col_drill:
                    # Filter samples in this cluster
                    cluster_samples = errors_df[errors_df["cluster"] == selected_cluster_id]
                    
                    if not cluster_samples.empty:
                        # Allow selecting a specific sample
                        selected_sample_id = st.selectbox(
                            "Select Error Sample to Inspect",
                            options=cluster_samples.index,
                            format_func=lambda x: f"Sample {x}"
                        )
                        
                        # Import drilldown & explain modules
                        from engine.drilldown import get_sample_details, compute_feature_deviation, find_nearest_neighbors
                        from engine.explain import generate_explanation
                        
                        # Drilldown Analysis
                        with st.spinner("Analyzing sample..."):
                            feature_cols = results.get("feature_cols", [])
                            details = get_sample_details(canonical, selected_sample_id)
                            devs = compute_feature_deviation(canonical, selected_sample_id, feature_cols)
                            neighbors = find_nearest_neighbors(canonical, selected_sample_id, feature_cols)
                            explanation = generate_explanation(details, devs, neighbors)
                        
                        # Result Card
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: left; padding: 1.5rem;">
                            <h3>Analysis for Sample {selected_sample_id}</h3>
                            <p style="font-size: 1.1em;">{explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Details Expander
                        with st.expander("See Feature Deviations", expanded=True):
                            if not devs.empty:
                                st.dataframe(
                                    devs[["feature", "value", "mean", "z_score"]].style.background_gradient(
                                        subset=["z_score"], cmap="RdBu_r"
                                    ),
                                    use_container_width=True
                                )
                        
                        with st.expander(f"See {len(neighbors)} Nearest Neighbors"):
                             for n in neighbors:
                                 st.write(f"Sample {n['index']} (Distance: {n['distance']:.2f})")
                                 st.json(n['data'])
                    else:
                        st.info("No samples in this cluster.")
            else:
                st.info("No clusters generated.")
        
        with tab4:
            st.subheader("Feature Attribution")
            
            global_imp = results["attribution"]["global"]
            if len(global_imp) > 0:
                st.bar_chart(global_imp.set_index("feature")["importance_mean"].head(10))
            else:
                st.info("No attribution data.")
        
        with tab5:
            st.subheader("Blind Spots")
            
            cw = results["blindspots"]["cw"]
            div = results["blindspots"]["divergence"]
            
            st.metric("Confident-Wrong Samples", len(cw))
            
            if len(div) > 0:
                st.write("**Top Divergent Features:**")
                st.dataframe(div.head(5), use_container_width=True)

elif page == "Compare Models":
    st.header("Model Comparison")
    
    st.info(
        "Compare two models on the same dataset to find performance differences. "
        "Upload your **Challenger (Model B)** file to see how it performs against the **Baseline (Model A)**."
    )
    
    # Conditional Demo Explanation
    if "comp_file_b" in st.session_state and "adult_predictions_rf.csv" in str(st.session_state["comp_file_b"]):
        st.info("**Demo Context:** Model B is a **Random Forest Classifier** trained to outperform the linear baseline.", icon=":material/info:")

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Model A (Baseline)")
        file_a = st.file_uploader("Upload Predictions A", type=["csv"], key="file_a")
        
    with col_b:
        st.subheader("Model B (Challenger)")
        file_b = st.file_uploader("Upload Predictions B", type=["csv"], key="file_b")
        
    # Logic to load files (upload or demo)
    df_a = None
    df_b = None
    
    # Check demo state
    if "comp_file_a" in st.session_state and not file_a:
        df_a = pd.read_csv(st.session_state["comp_file_a"])
        with col_a:
             st.success("Demo Model A Loaded", icon=":material/check:")
             
    if "comp_file_b" in st.session_state and not file_b:
        df_b = pd.read_csv(st.session_state["comp_file_b"])
        with col_b:
             st.success("Demo Model B Loaded", icon=":material/check:")
             
    # Check uploads (override demo)
    if file_a:
        df_a = pd.read_csv(file_a)
    if file_b:
        df_b = pd.read_csv(file_b)
        
    if df_a is not None and df_b is not None:
        try:
            # Common label assumption (users usually upload full prediction files)
            # Try to find y_true in Model A first
            if "y_true" in df_a.columns:
                y_true = df_a["y_true"]
            elif "y_true" in df_b.columns:
                y_true = df_b["y_true"]
            else:
                 st.error("Could not find 'y_true' column in either file.")
                 st.stop()
                 
            # Extract predictions
            pred_col_a = "y_pred" if "y_pred" in df_a.columns else df_a.columns[0]
            pred_col_b = "y_pred" if "y_pred" in df_b.columns else df_b.columns[0]
            
            y_pred_a = df_a[pred_col_a]
            y_pred_b = df_b[pred_col_b]
            
            # Generate Comparison Data
            from engine.compare import compute_comparison_metrics, find_disagreements, compare_slices
            
            # 1. Metrics
            comp_results = compute_comparison_metrics(y_true, y_pred_a, y_pred_b)
            
            # 2. Disagreements (for Executive Summary)
            df_analysis = pd.DataFrame({
                "y_true": y_true,
                "Model A": y_pred_a,
                "Model B": y_pred_b
            })
            disagreements = find_disagreements(df_analysis, "y_true", "Model A", "Model B")
            
            # 3. Slices (for Executive Summary)
            # Carry over features if available
            feature_cols = []
            df_slice_input = df_analysis.copy()
            
            if "comp_file_a" in st.session_state:
                 # Load demo features
                 demo_dataset_path = Path(__file__).parent.parent / "examples" / "demo" / "adult_dataset.csv"
                 if demo_dataset_path.exists():
                     demo_features = pd.read_csv(demo_dataset_path)
                     for c in demo_features.columns:
                         if c not in df_slice_input:
                             df_slice_input[c] = demo_features[c]
                             feature_cols.append(c)
            
            feature_cols = [c for c in feature_cols if c not in ["y_true", "y_pred_a", "y_pred_b", "income", "Model A", "Model B"]]
            
            top_slice_win = None
            slice_comp = pd.DataFrame()
            
            if feature_cols:
                slice_comp = compare_slices(df_slice_input, "y_true", "Model A", "Model B", feature_cols)
                if not slice_comp.empty:
                    top_slice_win = slice_comp.iloc[0] # Sorted by improvement descending
            
            # --- Executive Summary ---
            st.divider()
            st.subheader("Executive Summary")
            
            summ_col1, summ_col2, summ_col3 = st.columns(3)
            
            # 1. Net Impact
            fixed = disagreements["b_correct_count"]
            regressed = disagreements["a_correct_count"]
            net_fix = fixed - regressed
            
            with summ_col1:
                st.metric(
                    "Net Fix Impact",
                    f"{net_fix:+} Samples",
                    f"{fixed} Fixed, {regressed} Regressed",
                    delta_color="normal"
                )
            
            # 2. Verdict
            acc_delta = comp_results["delta"].get("accuracy", 0)
            f1_delta = comp_results["delta"].get("f1", 0)
            
            if acc_delta > 0.001:
                verdict = "Model B Wins"
                reason = f"Accuracy +{acc_delta:.1%}"
            elif acc_delta < -0.001:
                verdict = "Model A Wins"
                reason = f"Model B Accuracy {acc_delta:.1%}"
            else:
                verdict = "Tie"
                reason = "Performance is identical"
                
            with summ_col2:
                st.metric("Verdict", verdict, reason)
                
            # 3. Top Subgroup Win
            with summ_col3:
                if top_slice_win is not None and top_slice_win["improvement"] > 0.01:
                    rule = top_slice_win["rule"]
                    imp = top_slice_win["improvement"]
                    # Clean up rule string for display
                    if len(rule) > 25: rule = rule[:22] + "..."
                    st.metric("Top Subgroup Win", f"+{imp:.1%} err", rule)
                else:
                    st.metric("Top Subgroup Win", "None", "No significantly better slice")

            st.divider()
            
            # --- Detailed Metrics ---
            # 1. Metric Delta Board
            st.subheader("Performance Overview (Model B vs Model A)")
            
            metric_data = [] # For the detailed table
            
            metrics = ["accuracy", "precision", "recall", "f1"]
            cols = st.columns(len(metrics))
            
            deltas = comp_results["delta"]
            model_a = comp_results["model_a"]
            model_b = comp_results["model_b"]
            
            for i, m in enumerate(metrics):
                val_a = model_a.get(m, 0)
                val_b = model_b.get(m, 0)
                delta = deltas.get(m, 0)
                
                # Add to detailed table data
                metric_data.append({
                    "Metric": m.replace("_", " ").title(),
                    "Model A (Baseline)": val_a,
                    "Model B (Challenger)": val_b,
                    "Delta": delta
                })
                
                # Display KPI
                cols[i].metric(
                    label=m.replace("_", " ").title(),
                    value=f"{val_b:.1%}",
                    delta=f"{delta:+.1%}",
                    delta_color="normal",
                    help=f"Model A: {val_a:.1%}" # Tooltip for clarity
                )
            
            # Detailed Table
            st.caption("Detailed Comparison Board")
            metrics_df = pd.DataFrame(metric_data)
            
            # Styling
            st.dataframe(
                metrics_df.style.format({
                    "Model A (Baseline)": "{:.1%}",
                    "Model B (Challenger)": "{:.1%}",
                    "Delta": "{:+.1%}"
                }).background_gradient(subset=["Delta"], cmap="RdBu", vmin=-0.1, vmax=0.1),
                use_container_width=True,
                hide_index=True
            )

            # 2. Slice Comparison (New)
            st.divider()
            st.subheader("Comparison by Slice")
            
            if not slice_comp.empty:
                # Format for display
                disp_comp = slice_comp[["rule", "support", "error_rate_a", "error_rate_b", "improvement"]].copy()
                disp_comp.columns = ["Slice", "Support", "Model A Error", "Model B Error", "Improvement"]
                
                disp_comp["Model A Error"] = disp_comp["Model A Error"].apply(lambda x: f"{x:.1%}")
                disp_comp["Model B Error"] = disp_comp["Model B Error"].apply(lambda x: f"{x:.1%}")
                
                # Color code improvement
                def color_improvement(val):
                    if val > 0.05: return "background-color: #d4edda; color: #155724" # Green
                    if val < -0.05: return "background-color: #f8d7da; color: #721c24" # Red
                    return ""
                    
                st.dataframe(
                    disp_comp.style.map(color_improvement, subset=["Improvement"])
                                .format({"Improvement": "+{:.1%}"}),
                    use_container_width=True
                )
            elif feature_cols:
                st.info("No significant slices found for comparison.")
            else:
                st.warning("Feature columns not found. Cannot compare slices without features.")


            # 3. Disagreement Analysis
            st.divider()
            st.subheader("Disagreement Analysis")
            
            # Using already calculated disagreements
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Disagreements", disagreements["count"])
            c2.metric("Model A Correct", disagreements["a_correct_count"])
            c3.metric("Model B Correct", disagreements["b_correct_count"])
            
            if disagreements["count"] > 0:
                with st.expander("See Disagreement Samples"):
                    st.dataframe(disagreements["df"], use_container_width=True)

        except Exception as e:
            st.error(f"Error comparing models: {str(e)}")


elif page == "Report":
    st.header("Report Export", divider="gray")
    
    results = st.session_state.get("results")
    
    if results is None:
        st.info("Run an autopsy first to generate a report.")
    else:
        st.success("Report ready for download!")
        
        # Generate simple report
        from engine.report import generate_report
        
        report = generate_report(
            run_id="demo",
            metrics=results["metrics"],
            slices_df=pd.DataFrame(results["slices"]) if results["slices"] else None,
            profiles=results["clusters"]["profiles"],
            global_importance=results["attribution"]["global"],
            delta=results["attribution"]["delta"],
            total_samples=len(results["canonical"]),
            error_count=int(results["canonical"]["is_error"].sum()),
        )
        
        st.download_button(
            "Download Report (MD)",
            data=report,
            file_name="autopsy_report.md",
            mime="text/markdown"
        )

# Footer
st.divider()
st.caption("Model Autopsy — Open-source ML diagnostics for understanding model failures.")
