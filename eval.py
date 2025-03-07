import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from uuid import uuid4
    return mo, plt, uuid4


@app.cell
def _(mo):
    mo.md(
        r"""
        # Rust Eval 🦀

        This is an example of writing custom eval code that pulls a data frame from an oxen repo, then running code on it. In this case we have a data frame with columns `rust_code`, `rust_test_list` and run `cargo build`, `cargo clippy`, and `cargo test` then save the results.

        The predictions file is generated from the `inference.py` script
        """
    )
    return


@app.cell
def _(RustTool, mo):
    tools = [
        RustTool("build"),
        RustTool("clippy"),
        RustTool("test")
    ]
    should_add_tests = True
    num_rows = -1 # -1 == all

    file_path_text = mo.ui.text(value="/home/oxen/Code/Oxen/OxenNotebooks/results/GRPO_85_2025-03-05_03-47-22_Qwen2.5-Coder-3B-Instruct/predictions_code_and_tests.parquet", full_width=True)
    output_path_text = mo.ui.text(value="/home/oxen/Code/Oxen/OxenNotebooks/results/GRPO_85_2025-03-05_03-47-22_Qwen2.5-Coder-3B-Instruct/results_code_and_tests.parquet", full_width=True)

    run_form = mo.md(
        """
        Enter the local path to your data frame:
        {file_path_text}
        Output file path
        {output_path_text}
        """
    ).batch(
        file_path_text=file_path_text,
        output_path_text=output_path_text,
    ).form(
        submit_button_label="Run Eval",
        bordered=False,
        show_clear_button=True,
        clear_button_label="Reset"
    )

    run_form
    return (
        file_path_text,
        num_rows,
        output_path_text,
        run_form,
        should_add_tests,
        tools,
    )


@app.cell
def _(
    evaluate_solutions,
    file_path_text,
    mo,
    num_rows,
    output_path_text,
    pd,
    run_form,
    tools,
):
    # If the button is not pressed, stop execution
    mo.stop(
        run_form.value is None
    )

    # Read in df from oxen
    df = pd.read_parquet(file_path_text.value)

    results = evaluate_solutions(df, tools, output_path_text.value, max_rows=num_rows)
    return df, results


@app.cell
def _(plot_results, results):
    plot_results(results)
    return


@app.cell
def _(output_path_text, results):
    results.to_parquet(output_path_text.value)
    results
    return


@app.cell
def _(mo, plt):
    def plot_results(results):
        def _plot(df, column_name, title):
            build_passed_counts = results[column_name].value_counts()
            plt.figure(figsize=(4, 3))
            num_correct = build_passed_counts[True] if True in build_passed_counts else 0.0
            num_incorrect = build_passed_counts[False] if False in build_passed_counts else 0.0
            total = num_correct + num_incorrect
            percentage = (num_correct / total) * 100
            plt.title(f"{title}: {num_correct}/{total} = {percentage:.2f}%")

            # Create ordered index and corresponding values
            ordered_index = ['True', 'False']
            ordered_values = [build_passed_counts.get(True, 0), build_passed_counts.get(False, 0)]
    
            # Create color map
            # Retro color palette
            # https://www.color-hex.com/color-palette/165
            colors = ['#6fcb9f', '#fb2e01']
    
            # Plot with fixed order and colors
            plt.bar(ordered_index, ordered_values, color=colors)
            return plt.gca()

        return mo.vstack(
            [
                mo.md("# Results"),
                mo.hstack(
                    [
                        mo.as_html(_plot(results, 'build_passed', 'Build Passed')),
                        mo.as_html(_plot(results, 'clippy_passed', 'Clippy Passed')),
                        mo.as_html(_plot(results, 'test_passed', 'Test Passed'))
                    ],
                )
            ]
        )
    return (plot_results,)


@app.cell
def _():
    import pandas as pd
    import os
    import subprocess
    import shutil
    from pathlib import Path
    import argparse

    class RustTool:
        def __init__(self, name):
            self.name = name

        def run(self, results, project_dir):
            try:
                result = subprocess.run(
                    ["cargo", self.name, "--quiet"],
                    cwd=project_dir,
                    capture_output=True,
                    timeout=10
                )
                results[f'{self.name}_passed'] = result.returncode == 0
                results[f'{self.name}_stderr'] = result.stderr
            except:
                results[f'{self.name}_passed'] = False
                results[f'{self.name}_stderr'] = f"cargo {self.name} timeout"
            return results
    return Path, RustTool, argparse, os, pd, shutil, subprocess


@app.cell
def _():
    def extract_rust_code(rust_code: str) -> str:
        if "```rust" in rust_code:
            code = rust_code.split("```rust")[-1]
            code = code.split("```")[0]
            return code.strip()
        else:
            return rust_code
    return (extract_rust_code,)


@app.cell
def _():
    def template_rs_file():
        return """
    // {code}

    fn main() {
        println!("Hello, world!");
    }
    """
    return (template_rs_file,)


@app.cell
def _():
    def cargo_toml_file():
        return """
    [package]
    name = "rust-program"
    version = "0.1.0"
    edition = "2021"

    [dependencies]
    """
    return (cargo_toml_file,)


@app.cell
def _(
    Path,
    cargo_toml_file,
    extract_rust_code,
    shutil,
    template_rs_file,
    uuid4,
):
    def setup_and_test_rust_project(row, tools):
        """
        Sets up a Rust project from template and runs tests for a single row of data
        """
        # Create temporary project directory
        project_dir = Path("outputs") / Path("tests") / Path(f"temp_rust_project_{uuid4()}")
        project_dir_src = project_dir / Path("src")

        # mkdirs if they don't exist
        project_dir_src.mkdir(parents=True, exist_ok=True)

        # Read template
        template = template_rs_file()

        # Replace placeholders
        rust_code = extract_rust_code(row['response'])
        template = template.replace("// {code}", rust_code)

        print(template)

        # Write the cargo project files
        main_rs_path = project_dir_src / Path("main.rs")
        with open(main_rs_path, "w") as f:
            f.write(template)

        cargo_file_path = project_dir / Path("Cargo.toml")
        with open(cargo_file_path, "w") as f:
            f.write(cargo_toml_file())

        results = {
            'template': template
        }

        for tool in tools:
            results = tool.run(results, project_dir)

        # Clean up
        shutil.rmtree(project_dir)

        return results
    return (setup_and_test_rust_project,)


@app.cell
def _():
    def row_passed(row):
        if row['clippy_passed'] and row['tests_passed']:
            return True
        else:
            print(f"Row {row['idx']} failed:")
            print(f"Clippy failed: {row['clippy_passed']}")
            print(f"Tests failed: {row['tests_passed']}")
            return False
    return (row_passed,)


@app.cell
def _(mo, pd, setup_and_test_rust_project):
    def evaluate_solutions(df, tools, output_file, max_rows=-1):
        """
        Evaluates all solutions in the dataframe
        Returns dataframe with added clippy_passed and tests_passed columns
        """
        results = []

        total_passed = 0
        total_failed = 0
        num_rows = len(df) if max_rows < 0 else max_rows
        with mo.status.progress_bar(total=num_rows) as bar:
            for idx, row in df.iterrows():
            # for idx, row in mo.status.progress_bar(df.iterrows(), total=num_rows):
                if max_rows > 0 and idx >= max_rows:
                    break

                test_results = setup_and_test_rust_project(row, tools)
                test_results['idx'] = idx
                # merge the row with the test results
                row = row.to_dict()
                row.update(test_results)
                results.append(row)

                num_tools = len(tools)
                num_passed = 0
                for tool in tools:
                    passed = test_results[f'{tool.name}_passed']
                    # print results
                    if passed:
                        num_passed += 1
                all_passed = num_passed == num_tools
                print(f"Row {idx}: {num_passed}/{num_tools} passed")
                if all_passed:
                    total_passed += 1
                else:
                    total_failed += 1
                print(f"Total passed: {total_passed}, Total failed: {total_failed}")
                # print percentage
                accuracy = total_passed/(idx+1)*100
                percent_passed_str = f"Percentage passed {total_passed}/{idx+1} = {accuracy:.1f}%"
                print(percent_passed_str)
                bar.update(title=percent_passed_str)

                if idx % 100 == 0:
                    results_df = pd.DataFrame(results).set_index('idx')
                    results_df.to_parquet(output_file)

            # break

        # Convert results to dataframe and merge with original
        results_df = pd.DataFrame(results).set_index('idx')
        return results_df
    return (evaluate_solutions,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
