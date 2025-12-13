import subprocess
import sys


def run_script(script_name):
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    print("Starting portfolio analysis pipeline...\n")

    print("Step 1/3: Running calculations and optimization...")
    if not run_script('calculate_and_optimise.py'):
        print("Calculation and optimization failed. Stopping pipeline.")
        return

    print("\nStep 2/3: Analyzing reachable portfolios...")
    if not run_script('reachable_portfolios.py'):
        print("Reachable portfolios analysis failed. Stopping pipeline.")
        return

    print("\nStep 3/3: Creating visualizations and generating report...")
    if not run_script('visualise_and_summarise.py'):
        print("Visualization and report generation failed. Stopping pipeline.")
        return

    print("\n" + "=" * 50)
    print("Pipeline complete! All outputs saved in 'out' folder:")
    print("  - out/out_calculate/       (volatility & correlation data)")
    print("  - out/out_optimise/        (optimization results)")
    print("  - out/out_reachable/       (buy-only portfolio recommendations)")
    print("  - out/out_visualise/       (charts and graphs)")
    print("  - out/portfolio_report.html (interactive report)")
    print("=" * 50)


if __name__ == "__main__":
    main()