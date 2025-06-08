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

    # Custom CSS for a compact look
    st.markdown("""
        <style>
        .main {
            padding: 1rem 0.5rem 0rem 0.5rem;
        }
        .stPlotContainer {
            max-width: 500px;
            margin: auto;
        }
        .dataframe {
            max-width: 100%;
            margin-bottom: 0.25rem;
            font-size: 0.9em;
        }
        .stMetric {
            max-width: 180px;
            padding: 0.25rem 0;
        }
        .stMetric > div {
            padding: 0.25rem;
        }
        .block-container {
            padding: 1rem 0.5rem 0.5rem 0.5rem;
            max-width: 100%;
        }
        h1 {
            padding: 1rem 0 0.5rem 0;
            color: #2c3e50;
            font-size: 1.8rem;
            margin: 1rem 0 0.5rem 0;
        }
        h2 {
            color: #2c3e50;
            padding: 0.5rem 0 0.25rem 0;
            border-bottom: 1px solid #76b900;
            font-size: 1.3rem;
            margin: 0.5rem 0;
        }
        h3 {
            color: #2c3e50;
            padding: 0.25rem 0;
            font-size: 1.1rem;
            margin: 0.25rem 0;
        }
        h4 {
            color: #34495e;
            padding: 0.2rem 0;
            font-size: 1rem;
        }
        /* Sidebar styling */
        .css-1d391kg {
            padding: 0.5rem;
        }
        /* Compact metric containers */
        .metric-container {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 4px;
            margin: 0.25rem 0;
        }
        .chart-container {
            background-color: white;
            padding: 0.5rem;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .plot-section {
            margin: 0.5rem 0;
            padding: 0.5rem;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            background-color: #fafafa;
        }
        /* Compact expander */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
            padding: 0.25rem;
        }
        .streamlit-expanderContent {
            padding: 0.25rem;
        }
        /* Compact columns */
        .css-1r6slb0 {
            padding: 0.25rem;
        }
        /* Reduce space in sidebar */
        .css-1lcbmhc {
            padding-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # If command line arguments provided, use them; otherwise show input fields
    if original_model_path:
        # Use command line arguments
        results_dir = original_model_path
        comparison_dir = post_trained_model_path
        enable_comparison = post_trained_model_path is not None

        # Show command line args in sidebar for reference
        with st.sidebar:
            st.title("🛠️ Model Paths")
            st.markdown("**From Command Line Arguments:**")
            st.text_input("Original Model Path", results_dir, disabled=True)
            if comparison_dir:
                st.text_input("Post-trained Model Path", comparison_dir, disabled=True)
            else:
                st.info("No post-trained model specified")

            # View options
            st.markdown("**View Options**")
            show_individual_plots = st.checkbox("Individual Content Plots", True)
            show_detailed_metrics = st.checkbox("Detailed Metrics", True)
            show_distribution = st.checkbox("Performance Distribution", True)

            # Plot selection
            st.markdown("**Select Plots**")
            show_content_safety = st.checkbox("Content Safety", True)
            show_security = st.checkbox("Security", True)
            show_accuracy = st.checkbox("Accuracy", True)

            # Export options
            st.markdown("**Export**")
            if st.button("📊 Generate All Plots", use_container_width=True):
                try:
                    report_card = SafetyReportCard(results_dir)
                    comparison_model = SafetyReportCard(comparison_dir) if comparison_dir else None
                    plots_dir = report_card.generate_all_plots(comparison_model)
                    st.success(f"Plots saved to {plots_dir}")
                except Exception as e:
                    st.error(f"Error generating plots: {str(e)}")
    else:
        # Use interactive input fields
        with st.sidebar:
            st.title("🛠️ Controls")

            # Model selection
            st.markdown("**Model Selection**")
            st.info("💡 **Tip:** Run with command line arguments for faster setup:")
            st.code("streamlit run app.py -- /path/to/original /path/to/post-trained")

            results_dir = st.text_input(
                "Original Model Path",
                "workspace/results/DeepSeek-R1-Distill-Llama-8B"
            )

            # Comparison toggle
            enable_comparison = st.checkbox("Enable Model Comparison", True)

            if enable_comparison:
                comparison_dir = st.text_input(
                    "Post-trained Model Path",
                    "workspace/results/DeepSeek-R1-Distill-Llama-8B-Safety-Trained"
                )
                if comparison_dir:
                    st.caption(f"Comparing with: {Path(comparison_dir).name}")
            else:
                comparison_dir = None

            # View options
            st.markdown("**View Options**")
            show_individual_plots = st.checkbox("Individual Content Plots", True)
            show_detailed_metrics = st.checkbox("Detailed Metrics", True)
            show_distribution = st.checkbox("Performance Distribution", True)

            # Plot selection
            st.markdown("**Select Plots**")
            show_content_safety = st.checkbox("Content Safety", True)
            show_security = st.checkbox("Security", True)
            show_accuracy = st.checkbox("Accuracy", True)

            # Export options
            st.markdown("**Export**")
            if st.button("📊 Generate All Plots", use_container_width=True):
                if results_dir:
                    try:
                        report_card = SafetyReportCard(results_dir)
                        comparison_model = SafetyReportCard(comparison_dir) if enable_comparison and comparison_dir else None
                        plots_dir = report_card.generate_all_plots(comparison_model)
                        st.success(f"Plots saved to {plots_dir}")
                    except Exception as e:
                        st.error(f"Error generating plots: {str(e)}")
                else:
                    st.error("Enter results directory path.")

    # Validate paths
    if not results_dir:
        st.warning("Please provide a results directory path.")
        return

    if not Path(results_dir).exists():
        st.error(f"Original model path does not exist: {results_dir}")
        return

    if enable_comparison and comparison_dir and not Path(comparison_dir).exists():
        st.error(f"Post-trained model path does not exist: {comparison_dir}")
        return

    # Initialize the report cards
    try:
        report_card = SafetyReportCard(results_dir)
        comparison_model = None
        if enable_comparison and comparison_dir:
            comparison_model = SafetyReportCard(comparison_dir)
    except Exception as e:
        st.error(f"Error initializing report cards: {str(e)}")
        return

    # Main content area
    st.title("🛡️ Safety Report Card")

    # Model info section
    model_name = Path(results_dir).name
    if enable_comparison and comparison_model:
        comparison_name = Path(comparison_dir).name
        st.markdown(f"**📋 Original Model:** `{model_name}` **vs** **Post-trained Safe Model:** `{comparison_name}`")
    else:
        st.markdown(f"**📋 Model:** `{model_name}`")

    # Get all results
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
        st.error(f"Error loading data: {str(e)}")
        return

    # Summary metrics
    st.markdown("## 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_content_score = np.mean([v['score'] for v in content_scores.values()])
        if comparison_model:
            comp_content_score = np.mean([v['score'] for v in comparison_content_scores.values()])
            delta = comp_content_score - avg_content_score
            st.metric("Avg Content Safety", f"{comp_content_score:.1f}%", f"{delta:+.1f}%")
        else:
            st.metric("Avg Content Safety", f"{avg_content_score:.1f}%")

    with col2:
        aegis_score = content_scores['aegis_v2']['score']
        if comparison_model:
            comp_aegis = comparison_content_scores['aegis_v2']['score']
            delta = comp_aegis - aegis_score
            st.metric("Aegis v2", f"{comp_aegis:.1f}%", f"{delta:+.1f}%")
        else:
            st.metric("Aegis v2", f"{aegis_score:.1f}%")

    with col3:
        wildguard_score = content_scores['wildguard']['score']
        if comparison_model:
            comp_wildguard = comparison_content_scores['wildguard']['score']
            delta = comp_wildguard - wildguard_score
            st.metric("WildGuard", f"{comp_wildguard:.1f}%", f"{delta:+.1f}%")
        else:
            st.metric("WildGuard", f"{wildguard_score:.1f}%")

    with col4:
        resilience = garak_results['resilience']
        if comparison_model:
            comp_resilience = comparison_garak_results['resilience']
            delta = comp_resilience - resilience
            st.metric("Security Resilience", f"{comp_resilience:.1f}%", f"{delta:+.1f}%")
        else:
            st.metric("Security Resilience", f"{resilience:.1f}%")

    # Content Safety Plots
    if show_content_safety:
        st.markdown("## 🛡️ Content Safety")

        # Combined content safety plot
        st.markdown("**Content Safety Evaluation**")
        try:
            fig = report_card.plot_content_safety_combined(content_scores, comparison_content_scores)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating combined content safety plot: {str(e)}")

        if show_individual_plots:
            # Individual plots row
            col1, col2, col3 = st.columns(3)

            # with col1:
            #     st.markdown("**Aegis v2**")
            #     try:
            #         fig = report_card.plot_aegis_scores(content_scores, comparison_content_scores)
            #         st.pyplot(fig)
            #     except Exception as e:
            #         st.error(f"Error generating Aegis plot: {str(e)}")

            # with col2:
            #     st.markdown("**WildGuard**")
            #     try:
            #         fig = report_card.plot_wildguard_scores(content_scores, comparison_content_scores)
            #         st.pyplot(fig)
            #     except Exception as e:
            #         st.error(f"Error generating WildGuard plot: {str(e)}")

            # with col3:
            st.markdown("**Average Content Safety**")
            try:
                fig = report_card.plot_average_content_safety(content_scores, comparison_content_scores)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating average content safety plot: {str(e)}")

        # Detailed content safety metrics
        if show_detailed_metrics:
            with st.expander("📈 Content Safety Details"):
                metrics_df = pd.DataFrame([
                    {"Dataset": k.replace('_', ' ').title(), "Score": f"{v['score']:.1f}%"}
                    for k, v in content_scores.items()
                ])
                if comparison_model:
                    comp_metrics = pd.DataFrame([
                        {"Dataset": k.replace('_', ' ').title(), "Post-trained-Safe Score": f"{v['score']:.1f}%"}
                        for k, v in comparison_content_scores.items()
                    ])
                    metrics_df = metrics_df.merge(comp_metrics, on="Dataset")
                    # Add improvement column
                    metrics_df['Improvement'] = [
                        f"{float(row['Post-trained-Safe Score'].rstrip('%')) - float(row['Score'].rstrip('%')):.1f}%"
                        for _, row in metrics_df.iterrows()
                    ]
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # Security Analysis
    if show_security:
        st.markdown("## 🔒 Security")

        # Security resilience plot
        st.markdown("**Security Resilience**")
        try:
            fig = report_card.plot_resilience_scores(garak_results, comparison_garak_results)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating security resilience plot: {str(e)}")

        # Security metrics
        col1, col2, col3 = st.columns(3)

        total_probes = len(garak_results['probes'])
        excellent_probes = len([p for p in garak_results['probes'].values() if p['label'] == 'excellent'])
        competitive_probes = len([p for p in garak_results['probes'].values() if p['label'] == 'competitive'])

        with col1:
            if comparison_model:
                comp_total = len(comparison_garak_results['probes'])
                delta = total_probes - comp_total
                st.metric("Total Probes", f"{total_probes}", f"{delta:+d}")
            else:
                st.metric("Total Probes", f"{total_probes}")

        with col2:
            excellent_pct = (excellent_probes / total_probes * 100) if total_probes > 0 else 0
            if comparison_model:
                comp_excellent = len([p for p in comparison_garak_results['probes'].values() if p['label'] == 'excellent'])
                comp_total = len(comparison_garak_results['probes'])
                comp_excellent_pct = (comp_excellent / comp_total * 100) if comp_total > 0 else 0
                delta = comp_excellent_pct - excellent_pct
                st.metric("Excellent Probes", f"{excellent_pct:.1f}%", f"{delta:+.1f}%")
            else:
                st.metric("Excellent Probes", f"{excellent_pct:.1f}%")

        with col3:
            competitive_pct = (competitive_probes / total_probes * 100) if total_probes > 0 else 0
            if comparison_model:
                comp_competitive = len([p for p in comparison_garak_results['probes'].values() if p['label'] == 'competitive'])
                comp_competitive_pct = (comp_competitive / comp_total * 100) if comp_total > 0 else 0
                delta = comp_competitive_pct - competitive_pct
                st.metric("Competitive Probes", f"{competitive_pct:.1f}%", f"{delta:+.1f}%")
            else:
                st.metric("Competitive Probes", f"{competitive_pct:.1f}%")

        # Detailed security metrics
        if show_detailed_metrics:
            with st.expander("🔒 Security Probe Details"):
                garak_details = []
                for probe_name, probe_data in garak_results['probes'].items():
                    detail = {
                        "Probe": probe_name,
                        "Pass Rate": f"{probe_data['pass_rate']:.1f}%",
                        "Z-Score": f"{probe_data['z_score']:.2f}",
                        "Status": probe_data['label'].title()
                    }
                    if comparison_model and probe_name in comparison_garak_results['probes']:
                        comp_data = comparison_garak_results['probes'][probe_name]
                        detail["Post-trained-Safe Pass Rate"] = f"{comp_data['pass_rate']:.1f}%"
                        detail["Post-trained-Safe Status"] = comp_data['label'].title()
                        detail["Improvement"] = f"{comp_data['pass_rate'] - probe_data['pass_rate']:.1f}%"
                    garak_details.append(detail)
                garak_df = pd.DataFrame(garak_details).sort_values('Pass Rate', ascending=False)
                st.dataframe(garak_df, hide_index=True, use_container_width=True)

    # Accuracy Analysis
    if show_accuracy and accuracy_scores:
        st.markdown("## 🎯 Accuracy Analysis")

        # Accuracy plot
        st.markdown("### Accuracy Evaluation Scores")
        try:
            fig = report_card.plot_accuracy_scores(accuracy_scores, comparison_accuracy_scores)
            if fig:
                st.pyplot(fig)
            else:
                st.info("No accuracy scores available to plot.")
        except Exception as e:
            st.error(f"Error generating accuracy plot: {str(e)}")

        # Accuracy metrics
        if accuracy_scores:
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
                with st.expander("🎯 Accuracy Details"):
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
                            detail["Post-trained-Safe Score"] = f"{comp_score:.1f}%"
                            detail["Improvement"] = f"{comp_score - score:.1f}%"
                        acc_details.append(detail)
                    acc_df = pd.DataFrame(acc_details).sort_values('Score', ascending=False)
                    st.dataframe(acc_df, hide_index=True, use_container_width=True)

    elif show_accuracy and not accuracy_scores:
        st.markdown("## 🎯 Accuracy Analysis")
        st.info("No accuracy evaluation data found. Make sure accuracy-evals directory contains the evaluation results.")

    # Performance distribution
    if show_distribution and show_security:
        st.markdown("## 📈 Security Performance Distribution")

        # Create performance distribution
        performance_counts = {}
        for probe in garak_results['probes'].values():
            label = probe['label'].title()
            performance_counts[label] = performance_counts.get(label, 0) + 1

        if comparison_model:
            comp_counts = {}
            for probe in comparison_garak_results['probes'].values():
                label = probe['label'].title()
                comp_counts[label] = comp_counts.get(label, 0) + 1

        # Ensure all categories are present with correct order
        categories = ['Poor', 'Below Average', 'Average', 'Competitive', 'Excellent']
        for category in categories:
            if category not in performance_counts:
                performance_counts[category] = 0
            if comparison_model and category not in comp_counts:
                comp_counts[category] = 0

        # Create distribution DataFrame
        if comparison_model:
            distribution_df = pd.DataFrame({
                'Rating': categories,
                'Open Model': [performance_counts[cat] for cat in categories],
                'Post-trained-Safe Model': [comp_counts[cat] for cat in categories]
            })
            # Melt the DataFrame for the grouped bar chart
            distribution_df = distribution_df.melt(
                id_vars=['Rating'],
                value_vars=['Open Model', 'Post-trained-Safe Model'],
                var_name='Model',
                value_name='Count'
            )
        else:
            distribution_df = pd.DataFrame({
                'Rating': categories,
                'Count': [performance_counts[cat] for cat in categories]
            })

        # Create chart config with side-by-side histogram bars
        if comparison_model:
            chart_config = {
                'mark': {'type': 'bar', 'cornerRadius': 3},
                'encoding': {
                    'x': {
                        'field': 'Rating',
                        'type': 'ordinal',
                        'title': 'Performance Rating',
                        'sort': categories,  # Explicit sort order
                        'axis': {'labelAngle': -45}
                    },
                    'y': {
                        'field': 'Count',
                        'type': 'quantitative',
                        'title': 'Number of Probes'
                    },
                    'color': {
                        'field': 'Model',
                        'type': 'nominal',
                        'scale': {
                            'domain': ['Open Model', 'Post-trained-Safe Model'],
                            'range': ['#95a5a6', NVIDIA_GREEN]
                        }
                    },
                    'xOffset': {'field': 'Model'},  # Side-by-side positioning
                    'tooltip': [
                        {'field': 'Rating', 'type': 'nominal'},
                        {'field': 'Count', 'type': 'quantitative'},
                        {'field': 'Model', 'type': 'nominal'}
                    ]
                },
                'height': 300,
                'width': 'container'
            }
        else:
            chart_config = {
                'mark': {'type': 'bar', 'cornerRadius': 3},
                'encoding': {
                    'x': {
                        'field': 'Rating',
                        'type': 'ordinal',
                        'title': 'Performance Rating',
                        'sort': categories,  # Explicit sort order
                        'axis': {'labelAngle': -45}
                    },
                    'y': {
                        'field': 'Count',
                        'type': 'quantitative',
                        'title': 'Number of Probes'
                    },
                    'color': {'value': NVIDIA_GREEN},
                    'tooltip': [
                        {'field': 'Rating', 'type': 'nominal'},
                        {'field': 'Count', 'type': 'quantitative'}
                    ]
                },
                'height': 300,
                'width': 'container'
            }

        st.vega_lite_chart(distribution_df, chart_config, use_container_width=True)

        # Performance distribution table
        st.markdown("**Performance Distribution Summary**")
        if comparison_model:
            table_data = []
            for category in categories:
                table_data.append({
                    "Performance Rating": category,
                    "Open Model": performance_counts[category],
                    "Post-trained-Safe Model": comp_counts[category],
                    "Change": comp_counts[category] - performance_counts[category]
                })
            table_df = pd.DataFrame(table_data)
        else:
            table_data = []
            for category in categories:
                table_data.append({
                    "Performance Rating": category,
                    "Number of Probes": performance_counts[category]
                })
            table_df = pd.DataFrame(table_data)

        st.dataframe(table_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()