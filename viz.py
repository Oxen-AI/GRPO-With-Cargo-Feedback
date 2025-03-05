import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    import matplotlib.pyplot as plt
    from typing import Optional
    from oxen import RemoteRepo
    import os
    from pathlib import Path
    return Optional, Path, RemoteRepo, json, mo, os, pd, plt


@app.cell
def _():
    experiment = 'GRPO_94_2025-03-05_15-57-16_Qwen2.5-Coder-1.5B-Instruct'
    results_dir = 'results'
    return experiment, results_dir


@app.cell
def _(experiment, plot_file):
    plot_file(experiment, 'cargo_test_rewards.jsonl', f'{experiment}\n\nCargo Test Reward (max score=2.0)')
    return


@app.cell
def _(experiment, plot_file):
    plot_file(experiment, 'cargo_build_rewards.jsonl', f'{experiment}\n\nCargo Build Reward (max score=1.0)')
    return


@app.cell
def _(experiment, plot_file):
    plot_file(experiment, 'cargo_clippy_rewards.jsonl', f'{experiment}\n\nCargo Clippy Reward (max score=1.0)')
    return


@app.cell
def _(experiment, plot_file):
    plot_file(experiment, 'non_empty_rewards.jsonl', f'{experiment}\n\nNon-Empty Reward (max score=1.0)')
    return


@app.cell
def _(experiment, plot_file):
    plot_file(experiment, 'test_block_count_rewards.jsonl', f'{experiment}\n\nTest Block Count Reward (max score=1.0)')
    return


@app.cell
def _(Path, RemoteRepo, mo, os, plot_rolling_average, results_dir):
    def plot_file(experiment: str, filename: str, title: str):
        repo = RemoteRepo("ox/Rust")
        path = Path(f"outputs/{experiment}/{filename}")
        print(f"Downloading {path.name}")

        # make the results dir if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
    
        output_file = Path(results_dir) / path.name
        repo.download(path, revision=experiment, dst=output_file)


        image_name = filename.replace('.jsonl', '') + "_rolling_average.png"
        output_image_file = Path(results_dir) / image_name
        plot_rolling_average(output_file, title=title, window_size=100, save_path=output_image_file)
        return mo.image(src=output_image_file, rounded=True)
    return (plot_file,)


@app.cell
def _(Optional, json, pd, plt):
    def plot_rolling_average(filepath: str, title='Rolling Average Score Over Time', window_size: int = 100, save_path: Optional[str] = None) -> None:
        """
        Reads a JSONL file containing timestamp and score columns, calculates and plots
        the rolling average of scores over specified window size.

        Args:
            filepath (str): Path to the JSONL file
            window_size (int): Size of the rolling window for averaging (default: 100)
            save_path (Optional[str]): If provided, saves the plot to this path instead of displaying

        Returns:
            None
        """
        # Read JSONL file line by line into a list
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        # Convert to DataFrame
        # df = pd.DataFrame(data)[window_size:]
        df = pd.DataFrame(data)

        # Calculate rolling average
        rolling_avg = df['score'].rolling(window=window_size, min_periods=1).mean()

        # Create the plot using row indices
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(rolling_avg)), rolling_avg, 'b-',
                label=f'Rolling Average (window={window_size})')

        # Customize the plot
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Average Score')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Either save or display the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    return (plot_rolling_average,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
