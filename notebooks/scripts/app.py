# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import sys
import os
from pathlib import Path
from visualize_results import SafetyReportCard
import pandas as pd
import numpy as np

# Define NVIDIA green color constant
NVIDIA_GREEN = '#76b900'

st.set_page_config(
    page_title="Model Safety Report Card",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_model_paths_from_args():
    """Extract model paths from command line arguments passed after --"""
    if '--' in sys.argv:
        model_args = sys.argv[sys.argv.index('--') + 1:]
        if len(model_args) >= 1:
            original_path = model_args[0]
            post_trained_path = model_args[1] if len(model_args) > 1 else None
            return original_path, post_trained_path
    return None, None

def main():
    # Check for command line arguments first
    original_model_path, post_trained_model_path = get_model_paths_from_args()

    # Simple, clean CSS styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        h1 {
            color: #2c3e50;
            font-size: 2.2rem;
            margin-bottom: 1.5rem;
        }
        h2 {
            color: #2c3e50;
            padding: 1rem 0 0.5rem 0;
            border-bottom: 2px solid #76b900;
            font-size: 1.5rem;
            margin: 1.5rem 0 1rem 0;
        }
        h3 {
            color: #34495e;
            padding: 0.5rem 0;
            font-size: 1.2rem;
            margin: 1rem 0 0.5rem 0;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border: 1px solid #e9ecef;
        }
        .plot-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.title("🛠️ Configuration")

        # Model paths section
        st.header("Model Paths")

        if original_model_path:
            # Use command line arguments
            results_dir = original_model_path
            comparison_dir = post_trained_model_path
            enable_comparison = post_trained_model_path is not None

            st.info("**Using command line arguments:**")
            st.text_area("Original Model Path", results_dir, disabled=True)
            if comparison_dir:
                st.text_area("Post-trained Model Path", comparison_dir, disabled=True)
            else:
                st.warning("No post-trained model specified")
        else:
            # Interactive input
            st.info("💡 **Tip:** Run with command line arguments:\n`streamlit run streamlitapp.py -- /path/to/original /path/to/post-trained`")

            results_dir = st.text_input(
                "Original Model Path",
                "/ephemeral/workspace/results/DeepSeek-R1-Distill-Llama-8B",
                help="Path to the original model results directory"
            )

            enable_comparison = st.checkbox("Enable Model Comparison", value=True)

            if enable_comparison:
                comparison_dir = st.text_input(
                    "Post-trained Model Path",
                    "/ephemeral/workspace/results/DeepSeek-R1-Distill-Llama-8B-Safety-Trained",
                    help="Path to the post-trained model results directory"
                )
            else:
                comparison_dir = None

        # View options section
        st.header("View Options")
        show_content_safety = st.checkbox("Show Content Safety", value=True)
        show_security = st.checkbox("Show Security Analysis", value=True)
        show_accuracy = st.checkbox("Show Accuracy Analysis", value=True)
        show_detailed_metrics = st.checkbox("Show Detailed Metrics", value=True)

        # Export section
        st.header("Export")
        if st.button("📊 Generate All Plots", use_container_width=True):
            if results_dir:
                try:
                    report_card = SafetyReportCard(results_dir)
                    comparison_model = SafetyReportCard(comparison_dir) if enable_comparison and comparison_dir else None
                    plots_dir = report_card.generate_all_plots(comparison_model)
                    st.success(f"✅ Plots saved to {plots_dir}")
                except Exception as e:
                    st.error(f"❌ Error generating plots: {str(e)}")
            else:
                st.error("❌ Please enter a results directory path.")

    # Validate paths
    if not results_dir:
        st.warning("⚠️ Please provide a results directory path.")
        return

    if not Path(results_dir).exists():
        st.error(f"❌ Original model path does not exist: {results_dir}")
        return

    if enable_comparison and comparison_dir and not Path(comparison_dir).exists():
        st.error(f"❌ Post-trained model path does not exist: {comparison_dir}")
        return

    # Initialize the report cards
    try:
        report_card = SafetyReportCard(results_dir)
        comparison_model = None
        if enable_comparison and comparison_dir:
            comparison_model = SafetyReportCard(comparison_dir)
    except Exception as e:
        st.error(f"❌ Error initializing report cards: {str(e)}")
        return

    # Main content area
    st.title("🛡️ Model Safety Report Card")

    # Model information section
    model_name = Path(results_dir).name
    if enable_comparison and comparison_model:
        comparison_name = Path(comparison_dir).name
        st.info(f"**📋 Comparing:** `{model_name}` **vs** `{comparison_name}`")
    else:
        st.info(f"**📋 Analyzing Model:** `{model_name}`")

    # Load all data
    try:
        content_scores = report_card.read_content_safety_scores()
        garak_results = report_card.read_garak_results()
        accuracy_scores = report_card.read_accuracy_scores()

        comparison_content_scores = None
        comparison_garak_results = None
        comparison_accuracy_scores = None
        if comparison_model:
            comparison_content_scores = comparison_model.read_content_safety_scores()
            comparison_garak_results = comparison_model.read_garak_results()
            comparison_accuracy_scores = comparison_model.read_accuracy_scores()

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return

    # Summary metrics section
    st.header("📊 Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_content_score = np.mean([v['score'] for v in content_scores.values()])
        if comparison_model:
            comp_content_score = np.mean([v['score'] for v in comparison_content_scores.values()])
            delta = comp_content_score - avg_content_score
            st.metric("Average Content Safety", f"{avg_content_score:.1f}%", f"{delta:+.1f}%")
        else:
            st.metric("Average Content Safety", f"{avg_content_score:.1f}%")

    with col2:
        aegis_score = content_scores['aegis_v2']['score']
        if comparison_model:
            comp_aegis = comparison_content_scores['aegis_v2']['score']
            delta = comp_aegis - aegis_score
            st.metric("Aegis v2 Score", f"{aegis_score:.1f}%", f"{delta:+.1f}%")
        else:
            st.metric("Aegis v2 Score", f"{aegis_score:.1f}%")

    with col3:
        wildguard_score = content_scores['wildguard']['score']
        if comparison_model:
            comp_wildguard = comparison_content_scores['wildguard']['score']
            delta = comp_wildguard - wildguard_score
            st.metric("WildGuard Score", f"{wildguard_score:.1f}%", f"{delta:+.1f}%")
        else:
            st.metric("WildGuard Score", f"{wildguard_score:.1f}%")

    with col4:
        resilience = garak_results['resilience']
        if comparison_model:
            comp_resilience = comparison_garak_results['resilience']
            delta = comp_resilience - resilience
            st.metric("Security Resilience", f"{resilience:.1f}%", f"{delta:+.1f}%")
        else:
            st.metric("Security Resilience", f"{resilience:.1f}%")

    # Content Safety Analysis
    if show_content_safety:
        st.header("🛡️ Content Safety Analysis")

        # Combined content safety plot
        st.subheader("Content Safety Evaluation")
        try:
            fig = report_card.plot_content_safety_combined(content_scores, comparison_content_scores)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error generating combined content safety plot: {str(e)}")

        # Average content safety plot
        st.subheader("Average Content Safety Score")
        try:
            fig = report_card.plot_average_content_safety(content_scores, comparison_content_scores)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error generating average content safety plot: {str(e)}")

        # Detailed content safety metrics
        if show_detailed_metrics:
            st.subheader("📈 Detailed Content Safety Metrics")

            metrics_data = []
            for k, v in content_scores.items():
                row = {
                    "Dataset": k.replace('_', ' ').title(),
                    "Score": f"{v['score']:.1f}%"
                }

                if comparison_model:
                    comp_score = comparison_content_scores[k]['score']
                    row["Post-trained Score"] = f"{comp_score:.1f}%"
                    row["Improvement"] = f"{comp_score - v['score']:+.1f}%"

                metrics_data.append(row)

            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Security Analysis
    if show_security:
        st.header("🔒 Security Analysis")

        # Security resilience plot
        st.subheader("Security Resilience Score")
        try:
            fig = report_card.plot_resilience_scores(garak_results, comparison_garak_results)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error generating security resilience plot: {str(e)}")

        # Security metrics summary
        st.subheader("Security Performance Summary")

        col1, col2, col3 = st.columns(3)

        total_probes = len(garak_results['probes'])
        excellent_probes = len([p for p in garak_results['probes'].values() if p['label'] == 'excellent'])
        competitive_probes = len([p for p in garak_results['probes'].values() if p['label'] == 'competitive'])

        with col1:
            if comparison_model:
                comp_total = len(comparison_garak_results['probes'])
                delta = comp_total - total_probes
                st.metric("Total Security Probes", f"{total_probes}", f"{delta:+d}")
            else:
                st.metric("Total Security Probes", f"{total_probes}")

        with col2:
            excellent_pct = (excellent_probes / total_probes * 100) if total_probes > 0 else 0
            if comparison_model:
                comp_excellent = len([p for p in comparison_garak_results['probes'].values() if p['label'] == 'excellent'])
                comp_total = len(comparison_garak_results['probes'])
                comp_excellent_pct = (comp_excellent / comp_total * 100) if comp_total > 0 else 0
                delta = comp_excellent_pct - excellent_pct 
                st.metric("Excellent Performance", f"{excellent_pct:.1f}%", f"{delta:+.1f}%")
            else:
                st.metric("Excellent Performance", f"{excellent_pct:.1f}%")

        with col3:
            competitive_pct = (competitive_probes / total_probes * 100) if total_probes > 0 else 0
            if comparison_model:
                comp_competitive = len([p for p in comparison_garak_results['probes'].values() if p['label'] == 'competitive'])
                comp_competitive_pct = (comp_competitive / comp_total * 100) if comp_total > 0 else 0
                delta = comp_competitive_pct - competitive_pct
                st.metric("Competitive Performance", f"{competitive_pct:.1f}%", f"{delta:+.1f}%")
            else:
                st.metric("Competitive Performance", f"{competitive_pct:.1f}%")

        # Detailed security metrics
        if show_detailed_metrics:
            st.subheader("🔒 Detailed Security Probe Results")

            garak_details = []
            for probe_name, probe_data in garak_results['probes'].items():
                detail = {
                    "Probe": probe_name,
                    "Pass Rate": f"{probe_data['pass_rate']:.1f}%",
                    "Z-Score": f"{probe_data['z_score']:.2f}",
                    "Performance": probe_data['label'].title()
                }

                if comparison_model and probe_name in comparison_garak_results['probes']:
                    comp_data = comparison_garak_results['probes'][probe_name]
                    detail["Post-trained Pass Rate"] = f"{comp_data['pass_rate']:.1f}%"
                    detail["Post-trained Performance"] = comp_data['label'].title()
                    detail["Improvement"] = f"{comp_data['pass_rate'] - probe_data['pass_rate']:+.1f}%"

                garak_details.append(detail)

            garak_df = pd.DataFrame(garak_details).sort_values('Pass Rate', ascending=False)
            st.dataframe(garak_df, use_container_width=True, hide_index=True)

    # Accuracy Analysis
    if show_accuracy and accuracy_scores:
        st.header("🎯 Accuracy Analysis")

        # Accuracy plot
        st.subheader("Accuracy Evaluation Scores")
        try:
            fig = report_card.plot_accuracy_scores(accuracy_scores, comparison_accuracy_scores)
            if fig:
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("ℹ️ No accuracy scores available to plot.")
        except Exception as e:
            st.error(f"❌ Error generating accuracy plot: {str(e)}")

        # Accuracy metrics summary
        if accuracy_scores:
            st.subheader("Accuracy Performance Summary")

            avg_accuracy = np.mean(list(accuracy_scores.values()))
            max_accuracy = max(accuracy_scores.values()) if accuracy_scores else 0
            min_accuracy = min(accuracy_scores.values()) if accuracy_scores else 0

            col1, col2, col3 = st.columns(3)

            with col1:
                if comparison_model and comparison_accuracy_scores:
                    comp_avg = np.mean(list(comparison_accuracy_scores.values()))
                    delta = comp_avg - avg_accuracy
                    st.metric("Average Accuracy", f"{avg_accuracy:.1f}%", f"{delta:+.1f}%")
                else:
                    st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")

            with col2:
                if comparison_model and comparison_accuracy_scores:
                    comp_max = max(comparison_accuracy_scores.values())
                    delta = comp_max - max_accuracy
                    st.metric("Best Performance", f"{max_accuracy:.1f}%", f"{delta:+.1f}%")
                else:
                    st.metric("Best Performance", f"{max_accuracy:.1f}%")

            with col3:
                if comparison_model and comparison_accuracy_scores:
                    comp_min = min(comparison_accuracy_scores.values())
                    delta = comp_min - min_accuracy
                    st.metric("Lowest Performance", f"{min_accuracy:.1f}%", f"{delta:+.1f}%")
                else:
                    st.metric("Lowest Performance", f"{min_accuracy:.1f}%")

            # Detailed accuracy metrics
            if show_detailed_metrics:
                st.subheader("🎯 Detailed Accuracy Results")

                def format_dataset_name(name):
                    # Convert technical names to more readable format
                    name = name.replace('_', ' ')
                    name = name.replace('prompt strict', 'Prompt-Level Strict')
                    name = name.replace('inst strict', 'Instruction-Level Strict')
                    return name.title()

                acc_details = []
                for dataset, score in accuracy_scores.items():
                    detail = {
                        "Dataset": format_dataset_name(dataset),
                        "Score": f"{score:.1f}%"
                    }

                    if comparison_model and dataset in comparison_accuracy_scores:
                        comp_score = comparison_accuracy_scores[dataset]
                        detail["Post-trained Score"] = f"{comp_score:.1f}%"
                        detail["Improvement"] = f"{comp_score - score:+.1f}%"

                    acc_details.append(detail)

                acc_df = pd.DataFrame(acc_details).sort_values('Score', ascending=False)
                st.dataframe(acc_df, use_container_width=True, hide_index=True)

    elif show_accuracy and not accuracy_scores:
        st.header("🎯 Accuracy Analysis")
        st.info("ℹ️ No accuracy evaluation data found. Make sure accuracy-evals directory contains the evaluation results.")

if __name__ == "__main__":
    main()