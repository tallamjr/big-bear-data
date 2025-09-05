"""Visualization tools for benchmark results"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from typing import Optional


class BenchmarkVisualizer:
    def __init__(self, results_file: str = "benchmark_results.parquet"):
        self.results_file = Path(results_file)

    def load_results(self) -> Optional[pl.DataFrame]:
        """Load benchmark results"""
        if not self.results_file.exists():
            print(f"Results file {self.results_file} not found")
            return None
        return pl.read_parquet(self.results_file)

    def plot_execution_times(
        self, save_path: Optional[str] = None, interactive: bool = False
    ):
        """Create bar plot comparing execution times across libraries and queries"""
        df = self.load_results()
        if df is None:
            return

        # Filter out failed runs (negative times)
        df = df.filter(pl.col("execution_time_ms") > 0)

        if interactive:
            return self._plot_execution_times_interactive(df, save_path)
        else:
            return self._plot_execution_times_static(df, save_path)

    def _plot_execution_times_static(
        self, df: pl.DataFrame, save_path: Optional[str] = None
    ):
        """Static matplotlib/seaborn plot"""
        # Convert to pandas for seaborn
        pdf = df.to_pandas()

        plt.figure(figsize=(15, 8))

        # Create grouped bar plot
        sns.barplot(
            data=pdf,
            x="query_name",
            y="execution_time_ms",
            hue="library",
            palette="Set2",
        )

        plt.title("Benchmark Results: Execution Time by Query and Library", fontsize=16)
        plt.xlabel("Query", fontsize=12)
        plt.ylabel("Execution Time (ms)", fontsize=12)
        plt.yscale("log")  # Log scale for better readability
        plt.xticks(rotation=45)
        plt.legend(title="Library", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def _plot_execution_times_interactive(
        self, df: pl.DataFrame, save_path: Optional[str] = None
    ):
        """Interactive plotly plot"""
        # Convert to pandas for plotly
        pdf = df.to_pandas()

        fig = px.bar(
            pdf,
            x="query_name",
            y="execution_time_ms",
            color="library",
            title="Benchmark Results: Execution Time by Query and Library",
            labels={
                "execution_time_ms": "Execution Time (ms)",
                "query_name": "Query",
                "library": "Library",
            },
            log_y=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        fig.update_layout(
            xaxis_tickangle=-45, height=600, showlegend=True, font=dict(size=12)
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")

        fig.show()
        return fig

    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage comparison"""
        df = self.load_results()
        if df is None:
            return

        # Filter out failed runs
        df = df.filter(pl.col("execution_time_ms") > 0)
        pdf = df.to_pandas()

        plt.figure(figsize=(15, 8))

        sns.barplot(
            data=pdf, x="query_name", y="peak_memory_mb", hue="library", palette="Set3"
        )

        plt.title("Memory Usage by Query and Library", fontsize=16)
        plt.xlabel("Query", fontsize=12)
        plt.ylabel("Peak Memory Usage (MB)", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title="Library", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Memory plot saved to {save_path}")

        plt.show()

    def plot_performance_matrix(self, save_path: Optional[str] = None):
        """Create performance matrix heatmap"""
        df = self.load_results()
        if df is None:
            return

        # Create pivot table
        pivot_df = (
            df.filter(pl.col("execution_time_ms") > 0)
            .to_pandas()
            .pivot_table(
                values="execution_time_ms",
                index="query_name",
                columns="library",
                aggfunc="mean",
            )
        )

        plt.figure(figsize=(12, 8))

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Execution Time (ms)"},
        )

        plt.title("Performance Matrix: Average Execution Time", fontsize=16)
        plt.xlabel("Library", fontsize=12)
        plt.ylabel("Query", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Heatmap saved to {save_path}")

        plt.show()

    def plot_speedup_comparison(
        self, baseline: str = "pandas", save_path: Optional[str] = None
    ):
        """Plot speedup relative to baseline library"""
        df = self.load_results()
        if df is None:
            return

        # Calculate speedup relative to baseline
        pdf = df.filter(pl.col("execution_time_ms") > 0).to_pandas()

        # Get baseline times
        baseline_times = pdf[pdf["library"] == baseline].set_index("query_name")[
            "execution_time_ms"
        ]

        speedup_data = []
        for _, row in pdf.iterrows():
            query = row["query_name"]
            if query in baseline_times.index and row["library"] != baseline:
                speedup = baseline_times[query] / row["execution_time_ms"]
                speedup_data.append(
                    {"query_name": query, "library": row["library"], "speedup": speedup}
                )

        if not speedup_data:
            print(f"No speedup data available for baseline: {baseline}")
            return

        speedup_df = pl.DataFrame(speedup_data).to_pandas()

        plt.figure(figsize=(15, 8))

        sns.barplot(
            data=speedup_df,
            x="query_name",
            y="speedup",
            hue="library",
            palette="viridis",
        )

        plt.axhline(
            y=1, color="red", linestyle="--", alpha=0.7, label=f"{baseline} baseline"
        )
        plt.title(f"Speedup Comparison (relative to {baseline})", fontsize=16)
        plt.xlabel("Query", fontsize=12)
        plt.ylabel("Speedup Factor", fontsize=12)
        plt.yscale("log")
        plt.xticks(rotation=45)
        plt.legend(title="Library", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Speedup plot saved to {save_path}")

        plt.show()

    def generate_summary_report(self):
        """Generate text summary of benchmark results"""
        df = self.load_results()
        if df is None:
            return

        print("=== Benchmark Summary Report ===\n")

        # Overall statistics
        df_clean = df.filter(pl.col("execution_time_ms") > 0)

        print(f"Total benchmark runs: {len(df_clean)}")
        print(
            f"Libraries tested: {df_clean.select('library').unique().to_pandas()['library'].tolist()}"
        )
        print(
            f"Queries tested: {df_clean.select('query_name').unique().to_pandas()['query_name'].tolist()}"
        )
        print(
            f"Dataset: {df_clean.select('dataset_size').unique().to_pandas()['dataset_size'].iloc[0]}"
        )

        # Performance summary by library
        print("\n--- Performance by Library ---")
        library_stats = (
            df_clean.group_by("library")
            .agg(
                [
                    pl.mean("execution_time_ms").alias("avg_time"),
                    pl.median("execution_time_ms").alias("median_time"),
                    pl.mean("peak_memory_mb").alias("avg_memory"),
                ]
            )
            .sort("avg_time")
            .to_pandas()
        )

        for _, row in library_stats.iterrows():
            print(
                f"{row['library']:15s}: {row['avg_time']:8.1f}ms avg, {row['median_time']:8.1f}ms median, {row['avg_memory']:6.1f}MB memory"
            )

        # Best performing library per query
        print("\n--- Fastest Library per Query ---")
        fastest_per_query = (
            df_clean.group_by(["query_name", "library"])
            .agg([pl.mean("execution_time_ms").alias("avg_time")])
            .sort(["query_name", "avg_time"])
            .group_by("query_name")
            .first()
            .to_pandas()
        )

        for _, row in fastest_per_query.iterrows():
            print(
                f"{row['query_name']:20s}: {row['library']:15s} ({row['avg_time']:6.1f}ms)"
            )

    def create_dashboard(self, output_dir: str = "benchmark_plots"):
        """Generate complete set of visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Generating benchmark dashboard in {output_path}...")

        # Generate all plots
        self.plot_execution_times(
            output_path / "execution_times.png", interactive=False
        )
        self.plot_execution_times(
            output_path / "execution_times_interactive.html", interactive=True
        )
        self.plot_memory_usage(output_path / "memory_usage.png")
        self.plot_performance_matrix(output_path / "performance_matrix.png")
        self.plot_speedup_comparison(save_path=output_path / "speedup_comparison.png")

        print(f"\nDashboard complete. Files saved in {output_path}")
        self.generate_summary_report()
