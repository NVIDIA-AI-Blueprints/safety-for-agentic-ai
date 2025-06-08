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

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import glob
import sys

class SafetyReportCard:
    NVIDIA_GREEN = '#76b900'
    GRAY_COLOR = '#95a5a6'
    COLORS = ['#76b900', '#2c3e50', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.model_name = self.results_dir.name

    def read_content_safety_scores(self):
        """Read Aegis v2 and WildGuard scores"""
        scores = {
            'aegis_v2': self._read_aegis_scores(),
            'wildguard': self._read_wildguard_scores()
        }
        # print(f'scores: {scores}')
        return scores

    def _read_aegis_scores(self):
        """Read Aegis v2 scores from metrics.json"""
        aegis_dir = self.results_dir / 'content-safety-evals/aegis_v2'
        metrics_file = aegis_dir / 'metrics.json'

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Calculate percentage of safe responses
            safe_responses = metrics.get('safe', 0)
            unsafe_responses = metrics.get('unsafe', 0)
            total_responses = safe_responses + unsafe_responses

            if total_responses == 0:
                return {'score': 0}

            safety_percentage = (safe_responses / total_responses) * 100
            # print(f'safety_percentage: {safety_percentage}')
            return {'score': safety_percentage}

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Error reading Aegis v2 metrics from {metrics_file}")
            print(f"Error: {str(e)}")
            return {'score': 0}

    def _read_wildguard_scores(self):
        """Read WildGuard scores from metrics.json"""
        wildguard_dir = self.results_dir / 'content-safety-evals/wildguard'
        metrics_file = wildguard_dir / 'metrics.json'

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Calculate percentage of safe responses
            safe_responses = metrics.get('safe', 0)
            unsafe_responses = metrics.get('unsafe', 0)
            total_responses = safe_responses + unsafe_responses

            if total_responses == 0:
                return {'score': 0}

            safety_percentage = (safe_responses / total_responses) * 100

            return {'score': safety_percentage}

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Error reading WildGuard metrics from {metrics_file}")
            print(f"Error: {str(e)}")
            return {'score': 0}

    def read_garak_results(self):
        """Read Garak security evaluation results"""
        # Read garak_scores.csv from the security-evals directory
        garak_scores_path = self.results_dir / 'security-evals/garak/reports/garak_results.csv'
        # print(f'garak_scores_path: {garak_scores_path}')

        try:
            # Read the CSV file
            df = pd.read_csv(garak_scores_path)

            # Get all probes and their pass rates
            probes = {}
            labels_set = set()  # Track unique labels

            for _, row in df.iterrows():
                probe_name = row['probe']
                # Convert pass_rate from percentage string to float
                pass_rate = float(row['pass_rate'].strip('%'))
                # print(f'pass_rate: {pass_rate}')

                # Handle empty z_score values
                try:
                    z_score = float(row['z_score']) if pd.notna(row['z_score']) and str(row['z_score']).strip() != '' else 0.0
                except (ValueError, TypeError):
                    z_score = 0.0

                # Use z_score_status directly from CSV file
                label = row['z_score_status'] if pd.notna(row['z_score_status']) and str(row['z_score_status']).strip() != '' else 'unknown'
                labels_set.add(label)  # Add to set of unique labels

                probes[probe_name] = {
                    'pass_rate': pass_rate,
                    'z_score': z_score,
                    'label': label
                }
                # print(f'probes: {probes}')

            print(f'All unique labels found: {sorted(list(labels_set))}')

            # If no probes found, return default values
            if not probes:
                return {
                    'probes': {},
                    'resilience': 0
                }

            # Calculate resilience (% probes with z_score_status competitive or excellent)
            positive_statuses = {'competitive', 'excellent'}
            probes_with_positive_status = len([p for p in probes.values() if p['label'] in positive_statuses])
            resilience = (probes_with_positive_status / len(probes)) * 100
            print(f'resilience: {resilience}')

            return {
                'probes': probes,
                'resilience': resilience
            }

        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Warning: Could not find or read garak scores file at {garak_scores_path}")
            print(f"Error: {str(e)}")
            return {
                'probes': {},
                'resilience': 0
            }
        except (ValueError, TypeError) as e:
            print(f"Warning: Error processing data from garak scores file")
            print(f"Error: {str(e)}")
            return {
                'probes': {},
                'resilience': 0
            }

    def read_accuracy_scores(self):
        """Read accuracy evaluation results from all datasets"""
        accuracy_dir = self.results_dir / 'accuracy-evals'
        accuracy_scores = {}

        try:
            # Look for JSON files in accuracy-evals subdirectories
            for dataset_dir in accuracy_dir.iterdir():
                if dataset_dir.is_dir():
                    # Check for simple JSON files (like gpqa-diamond)
                    json_files = list(dataset_dir.glob('*.json'))
                    if json_files:
                        for json_file in json_files:
                            try:
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                if 'score' in data:
                                    dataset_name = data.get('task_name', dataset_dir.name)
                                    accuracy_scores[dataset_name] = data['score'] * 100  # Convert to percentage
                            except (json.JSONDecodeError, KeyError) as e:
                                print(f"Warning: Error reading {json_file}: {e}")

                    # Check for results in subdirectories (like ifeval/test-model/)
                    else:
                        for subdir in dataset_dir.iterdir():
                            if subdir.is_dir():
                                result_files = list(subdir.glob('results_*.json'))
                                if result_files:
                                    try:
                                        with open(result_files[0], 'r') as f:
                                            data = json.load(f)

                                        # Extract main metric from ifeval-style results
                                        if 'results' in data:
                                            for task_name, results in data['results'].items():
                                                # Check for specific strict accuracy metrics only
                                                prompt_level_strict = None
                                                inst_level_strict = None

                                                for metric, value in results.items():
                                                    if not metric.endswith('_stderr') and isinstance(value, (int, float)):
                                                        if 'prompt_level_strict_acc' in metric and not 'prompt_level_strict_acc_stderr' in metric:
                                                            prompt_level_strict = value * 100
                                                            # print(f'prompt_level_strict: {prompt_level_strict}')
                                                        elif 'inst_level_strict_acc' in metric and not 'inst_level_strict_acc_stderr' in metric:
                                                            inst_level_strict = value * 100
                                                            # print(f'inst_level_strict: {inst_level_strict}')

                                                # Store only the strict metrics if they exist
                                                if prompt_level_strict is not None:
                                                    accuracy_scores[f"{task_name}_prompt_strict"] = prompt_level_strict
                                                if inst_level_strict is not None:
                                                    accuracy_scores[f"{task_name}_inst_strict"] = inst_level_strict
                                    except (json.JSONDecodeError, KeyError) as e:
                                        print(f"Warning: Error reading {result_files[0]}: {e}")

            return accuracy_scores

        except Exception as e:
            print(f"Warning: Error reading accuracy evaluations: {e}")
            return {}

    def plot_aegis_scores(self, scores, comparison_scores=None):
        """Plot Aegis v2 content safety scores"""
        plt.figure(figsize=(6, 4), dpi=300)

        current_score = scores['aegis_v2']['score']
        metrics = ['Aegis v2']
        current_scores = [current_score]

        x = np.arange(len(metrics))
        width = 0.3

        if comparison_scores:
            comp_score = comparison_scores['aegis_v2']['score']
            comp_scores = [comp_score]

            plt.bar(x - width/2, current_scores, width, label='Open Model', color=self.GRAY_COLOR, alpha=0.9)
            plt.bar(x + width/2, comp_scores, width, label='Post-trained-Safe Model', color=self.NVIDIA_GREEN, alpha=0.9)

            # Add value labels
            plt.text(x[0] - width/2, current_scores[0], f'{current_scores[0]:.1f}%', ha='center', va='bottom')
            plt.text(x[0] + width/2, comp_scores[0], f'{comp_scores[0]:.1f}%', ha='center', va='bottom')

            # Add delta
            delta = comp_score - current_score
            color = 'green' if delta > 0 else 'red'
            plt.text(0, max(current_score, comp_score) + 2, f'{delta:+.1f}%', ha='center', color=color, fontweight='bold')
        else:
            plt.bar(x, current_scores, width, color=self.NVIDIA_GREEN, alpha=0.9)
            plt.text(x[0], current_scores[0], f'{current_scores[0]:.1f}%', ha='center', va='bottom')

        plt.ylabel('Safe Responses (%)')
        plt.title('Aegis v2 Content Safety')
        plt.xticks(x, metrics)
        plt.ylim(0, 100)
        if comparison_scores:
            plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        save_dir = self.results_dir / 'plots'
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / 'aegis_content_safety.png', dpi=300, bbox_inches='tight')
        return plt.gcf()

    def plot_wildguard_scores(self, scores, comparison_scores=None):
        """Plot WildGuard content safety scores"""
        plt.figure(figsize=(6, 4), dpi=300)

        current_score = scores['wildguard']['score']
        metrics = ['WildGuard']
        current_scores = [current_score]

        x = np.arange(len(metrics))
        width = 0.3

        if comparison_scores:
            comp_score = comparison_scores['wildguard']['score']
            comp_scores = [comp_score]

            plt.bar(x - width/2, current_scores, width, label='Open Model', color=self.GRAY_COLOR, alpha=0.9)
            plt.bar(x + width/2, comp_scores, width, label='Post-trained-Safe Model', color=self.NVIDIA_GREEN, alpha=0.9)

            # Add value labels
            plt.text(x[0] - width/2, current_scores[0], f'{current_scores[0]:.1f}%', ha='center', va='bottom')
            plt.text(x[0] + width/2, comp_scores[0], f'{comp_scores[0]:.1f}%', ha='center', va='bottom')

            # Add delta
            delta = comp_score - current_score
            color = 'green' if delta > 0 else 'red'
            plt.text(0, max(current_score, comp_score) + 2, f'{delta:+.1f}%', ha='center', color=color, fontweight='bold')
        else:
            plt.bar(x, current_scores, width, color=self.GRAY_COLOR, alpha=0.9)
            plt.text(x[0], current_scores[0], f'{current_scores[0]:.1f}%', ha='center', va='bottom')

        plt.ylabel('Safe Responses (%)')
        plt.title('WildGuard Content Safety')
        plt.xticks(x, metrics)
        plt.ylim(0, 100)
        if comparison_scores:
            plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        save_dir = self.results_dir / 'plots'
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / 'wildguard_content_safety.png', dpi=300, bbox_inches='tight')
        return plt.gcf()

    def plot_content_safety_combined(self, scores, comparison_scores=None):
        """Plot combined Aegis v2 and WildGuard content safety scores"""
        plt.figure(figsize=(8, 5), dpi=300)

        # Prepare data
        aegis_score = scores['aegis_v2']['score']
        wildguard_score = scores['wildguard']['score']
        datasets = ['Aegis v2', 'WildGuard']
        current_scores = [aegis_score, wildguard_score]

        x = np.arange(len(datasets))
        width = 0.35

        if comparison_scores:
            comp_aegis = comparison_scores['aegis_v2']['score']
            comp_wildguard = comparison_scores['wildguard']['score']
            comp_scores = [comp_aegis, comp_wildguard]

            plt.bar(x - width/2, current_scores, width, label='Open Model', color=self.GRAY_COLOR, alpha=0.9)
            plt.bar(x + width/2, comp_scores, width, label='Post-trained-Safe Model', color=self.NVIDIA_GREEN, alpha=0.9)

            # Add value labels
            for i, (curr, comp) in enumerate(zip(current_scores, comp_scores)):
                plt.text(x[i] - width/2, curr + 0.5, f'{curr:.1f}%', ha='center', va='bottom')
                plt.text(x[i] + width/2, comp + 0.5, f'{comp:.1f}%', ha='center', va='bottom')

                # Add delta
                delta = comp - curr
                color = 'green' if delta > 0 else 'red'
                plt.text(x[i], max(curr, comp) + 3, f'{delta:+.1f}%', ha='center', color=color, fontweight='bold')
        else:
            plt.bar(x, current_scores, width, color=self.GRAY_COLOR, alpha=0.9)
            for i, score in enumerate(current_scores):
                plt.text(x[i], score + 0.5, f'{score:.1f}%', ha='center', va='bottom')

        plt.ylabel('Safe Responses (%)')
        plt.title('Content Safety Evaluation')
        plt.xticks(x, datasets)
        plt.ylim(0, 100)
        if comparison_scores:
            plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        save_dir = self.results_dir / 'plots'
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / 'content_safety_combined.png', dpi=300, bbox_inches='tight')
        return plt.gcf()

    def plot_average_content_safety(self, scores, comparison_scores=None):
        """Plot average content safety scores"""
        plt.figure(figsize=(6, 4), dpi=300)

        current_score = np.mean([scores['aegis_v2']['score'], scores['wildguard']['score']])
        metrics = ['Average Content Safety']
        current_scores = [current_score]

        x = np.arange(len(metrics))
        width = 0.3

        if comparison_scores:
            comp_score = np.mean([comparison_scores['aegis_v2']['score'], comparison_scores['wildguard']['score']])
            comp_scores = [comp_score]

            plt.bar(x - width/2, current_scores, width, label='Open Model', color=self.GRAY_COLOR, alpha=0.9)
            plt.bar(x + width/2, comp_scores, width, label='Post-trained-Safe Model', color=self.NVIDIA_GREEN, alpha=0.9)

            # Add value labels
            plt.text(x[0] - width/2, current_scores[0], f'{current_scores[0]:.1f}%', ha='center', va='bottom')
            plt.text(x[0] + width/2, comp_scores[0], f'{comp_scores[0]:.1f}%', ha='center', va='bottom')

            # Add delta
            delta = comp_score - current_score
            color = 'green' if delta > 0 else 'red'
            plt.text(0, max(current_score, comp_score) + 2, f'{delta:+.1f}%', ha='center', color=color, fontweight='bold')
        else:
            plt.bar(x, current_scores, width, color=self.GRAY_COLOR, alpha=0.9)
            plt.text(x[0], current_scores[0], f'{current_scores[0]:.1f}%', ha='center', va='bottom')

        plt.ylabel('Safe Responses (%)')
        plt.title('Average Content Safety Score')
        plt.xticks(x, metrics)
        plt.ylim(0, 100)
        if comparison_scores:
            plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        save_dir = self.results_dir / 'plots'
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / 'average_content_safety.png', dpi=300, bbox_inches='tight')
        return plt.gcf()

    def plot_resilience_scores(self, results, comparison_results=None):
        """Plot security resilience scores"""
        plt.figure(figsize=(6, 4), dpi=300)

        current_resilience = results['resilience']
        metrics = ['Security Resilience']
        current_scores = [current_resilience]

        x = np.arange(len(metrics))
        width = 0.3

        if comparison_results:
            comp_resilience = comparison_results['resilience']
            comp_scores = [comp_resilience]

            plt.bar(x - width/2, current_scores, width, label='Open Model', color=self.GRAY_COLOR, alpha=0.9)
            plt.bar(x + width/2, comp_scores, width, label='Post-trained-Safe Model', color=self.NVIDIA_GREEN, alpha=0.9)

            # Add value labels
            plt.text(x[0] - width/2, current_scores[0], f'{current_scores[0]:.1f}%', ha='center', va='bottom')
            plt.text(x[0] + width/2, comp_scores[0], f'{comp_scores[0]:.1f}%', ha='center', va='bottom')

            # Add delta
            delta = comp_resilience - current_resilience
            color = 'green' if delta > 0 else 'red'
            plt.text(0, max(current_resilience, comp_resilience) + 2, f'{delta:+.1f}%', ha='center', color=color, fontweight='bold')
        else:
            plt.bar(x, current_scores, width, color=self.GRAY_COLOR, alpha=0.9)
            plt.text(x[0], current_scores[0], f'{current_scores[0]:.1f}%', ha='center', va='bottom')

        plt.ylabel('Resilience (%)')
        plt.title('Security Resilience Score')
        plt.xticks(x, metrics)
        plt.ylim(0, 100)
        if comparison_results:
            plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        save_dir = self.results_dir / 'plots'
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / 'security_resilience.png', dpi=300, bbox_inches='tight')
        return plt.gcf()

    def plot_accuracy_scores(self, scores, comparison_scores=None):
        """Plot accuracy evaluation scores"""
        if not scores:
            print("No accuracy scores to plot")
            return None

        # Create cleaner dataset names for display
        def clean_dataset_name(name):
            # Convert technical names to more readable format
            name = name.replace('_', ' ')
            name = name.replace('prompt strict', 'Prompt-Level Strict')
            name = name.replace('inst strict', 'Instruction-Level Strict')
            return name.title()

        # Determine figure size based on number of datasets
        num_datasets = len(scores)
        fig_width = max(12, num_datasets * 1.5)  # Adjust width based on number of datasets
        plt.figure(figsize=(fig_width, 8), dpi=300)

        datasets = list(scores.keys())
        display_names = [clean_dataset_name(name) for name in datasets]
        current_scores = list(scores.values())

        x = np.arange(len(datasets))
        width = 0.35

        if comparison_scores:
            # Align comparison scores with current scores
            comp_scores = [comparison_scores.get(dataset, 0) for dataset in datasets]

            plt.bar(x - width/2, current_scores, width, label='Open Model', color=self.GRAY_COLOR, alpha=0.9)
            plt.bar(x + width/2, comp_scores, width, label='Post-trained-Safe Model', color=self.NVIDIA_GREEN, alpha=0.9)

            # Add value labels (smaller font for crowded plots)
            font_size = max(8, 12 - num_datasets//3)
            for i, (comp, curr) in enumerate(zip(comp_scores, current_scores)):
                plt.text(x[i] - width/2, curr + 0.5, f'{curr:.1f}%', ha='center', va='bottom',
                        rotation=90 if num_datasets > 6 else 0, fontsize=font_size)
                plt.text(x[i] + width/2, comp + 0.5, f'{comp:.1f}%', ha='center', va='bottom',
                        rotation=90 if num_datasets > 6 else 0, fontsize=font_size)

                # Add delta (only if space allows)
                if num_datasets <= 8:
                    delta = comp - curr
                    color = 'green' if delta > 0 else 'red'
                    plt.text(x[i], max(curr, comp) + 3, f'{delta:+.1f}%', ha='center',
                            color=color, fontweight='bold', fontsize=font_size)
        else:
            # Use different colors for different datasets
            colors = (self.COLORS * ((num_datasets // len(self.COLORS)) + 1))[:num_datasets]
            plt.bar(x, current_scores, width, color=colors, alpha=0.9)

            font_size = max(8, 12 - num_datasets//3)
            for i, score in enumerate(current_scores):
                plt.text(x[i], score + 0.5, f'{score:.1f}%', ha='center', va='bottom',
                        rotation=90 if num_datasets > 6 else 0, fontsize=font_size)

        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Accuracy Evaluation Scores', fontsize=14, fontweight='bold')

        # Adjust x-axis labels based on number of datasets
        if num_datasets > 8:
            plt.xticks(x, display_names, rotation=45, ha='right', fontsize=10)
        elif num_datasets > 4:
            plt.xticks(x, display_names, rotation=30, ha='right', fontsize=11)
        else:
            plt.xticks(x, display_names, rotation=0, ha='center', fontsize=12)

        plt.ylim(0, min(105, max(current_scores + (comp_scores if comparison_scores else [])) + 10))

        if comparison_scores:
            plt.legend(fontsize=11)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        save_dir = self.results_dir / 'plots'
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / 'accuracy_scores.png', dpi=300, bbox_inches='tight')
        return plt.gcf()

    def generate_all_plots(self, comparison_model=None):
        """Generate all plots"""
        print(f"Generating plots for {self.model_name}")

        # Create plots directory
        plots_dir = self.results_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # Read all data
        content_scores = self.read_content_safety_scores()
        garak_results = self.read_garak_results()
        accuracy_scores = self.read_accuracy_scores()

        comparison_content_scores = None
        comparison_garak_results = None
        comparison_accuracy_scores = None
        if comparison_model:
            comparison_content_scores = comparison_model.read_content_safety_scores()
            comparison_garak_results = comparison_model.read_garak_results()
            comparison_accuracy_scores = comparison_model.read_accuracy_scores()

        # Generate all plots
        print("Generating combined content safety plot...")
        self.plot_content_safety_combined(content_scores, comparison_content_scores)
        plt.close()

        # print("Generating Aegis content safety plot...")
        # self.plot_aegis_scores(content_scores, comparison_content_scores)
        # plt.close()

        # print("Generating WildGuard content safety plot...")
        # self.plot_wildguard_scores(content_scores, comparison_content_scores)
        # plt.close()

        print("Generating average content safety plot...")
        self.plot_average_content_safety(content_scores, comparison_content_scores)
        plt.close()

        print("Generating security resilience plot...")
        self.plot_resilience_scores(garak_results, comparison_garak_results)
        plt.close()

        print("Generating accuracy scores plot...")
        self.plot_accuracy_scores(accuracy_scores, comparison_accuracy_scores)
        plt.close()

        print(f"All plots saved to {plots_dir}")
        return str(plots_dir)

    # Legacy methods for backwards compatibility
    def plot_content_safety_scores(self, scores, comparison_scores=None):
        """Legacy method - redirects to average content safety"""
        return self.plot_average_content_safety(scores, comparison_scores)

    def plot_garak_results(self, results, comparison_results=None):
        """Legacy method - redirects to resilience scores"""
        return self.plot_resilience_scores(results, comparison_results)

    def generate_report(self, output_dir='reports', comparison_model=None):
        """Generate a complete safety report with all visualizations"""
        os.makedirs(output_dir, exist_ok=True)

        # Read all scores
        content_scores = self.read_content_safety_scores()
        garak_results = self.read_garak_results()
        accuracy_scores = self.read_accuracy_scores()

        comparison_content_scores = None
        comparison_garak_results = None
        comparison_accuracy_scores = None
        if comparison_model:
            comparison_content_scores = comparison_model.read_content_safety_scores()
            comparison_garak_results = comparison_model.read_garak_results()
            comparison_accuracy_scores = comparison_model.read_accuracy_scores()

        # Generate plots
        content_fig = self.plot_average_content_safety(content_scores, comparison_content_scores)
        garak_fig = self.plot_resilience_scores(garak_results, comparison_garak_results)

        # Save plots
        content_fig.savefig(f'{output_dir}/content_safety_scores.png')
        garak_fig.savefig(f'{output_dir}/security_scores.png')

        # Generate summary report
        summary = {
            'model_name': self.model_name,
            'content_safety': content_scores,
            'security': {
                'garak_resilience': garak_results['resilience'],
                'probe_summary': {
                    probe: results['label']
                    for probe, results in garak_results['probes'].items()
                }
            },
            'accuracy': accuracy_scores
        }

        with open(f'{output_dir}/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def generate_plots(self, comparison_model=None):
        """Generate and save all plots (legacy method)"""
        return self.generate_all_plots(comparison_model)

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate safety report card visualizations')
    parser.add_argument('original_model_path',
                       help='Path to the original model results directory')
    parser.add_argument('post_trained_model_path', nargs='?', default=None,
                       help='Path to the post-trained safe model results directory (optional)')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory for plots (default: model_path/plots)')
    parser.add_argument('--generate-report', '-r', action='store_true',
                       help='Generate complete report with JSON summary')

    args = parser.parse_args()

    # Validate paths
    if not Path(args.original_model_path).exists():
        print(f"Error: Original model path does not exist: {args.original_model_path}")
        sys.exit(1)

    if args.post_trained_model_path and not Path(args.post_trained_model_path).exists():
        print(f"Error: Post-trained model path does not exist: {args.post_trained_model_path}")
        sys.exit(1)

    # Initialize report cards
    print(f"Loading original model from: {args.original_model_path}")
    report_card = SafetyReportCard(args.original_model_path)

    comparison_card = None
    if args.post_trained_model_path:
        print(f"Loading post-trained model from: {args.post_trained_model_path}")
        comparison_card = SafetyReportCard(args.post_trained_model_path)
        print(f"Comparing: {report_card.model_name} vs {comparison_card.model_name}")
    else:
        print(f"Generating plots for single model: {report_card.model_name}")

    # Generate plots
    try:
        if args.generate_report:
            output_dir = args.output_dir or 'reports'
            print(f"Generating complete report in: {output_dir}")
            report_card.generate_report(output_dir, comparison_card)
            print("Report generation completed successfully!")
        else:
            plots_dir = report_card.generate_all_plots(comparison_card)
            print("Plot generation completed successfully!")
            print(f"Plots saved to: {plots_dir}")

            if comparison_card:
                print("\nSummary:")
                print(f"- Original model: {report_card.model_name}")
                print(f"- Post-trained model: {comparison_card.model_name}")
                print("- Generated comparison plots for content safety, security, and accuracy")
            else:
                print(f"\nGenerated individual plots for: {report_card.model_name}")

    except Exception as e:
        print(f"Error during generation: {str(e)}")
        sys.exit(1)