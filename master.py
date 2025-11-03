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

    print("Step 1/4: Running calculations...")
    if not run_script('calculate.py'):
        print("Calculation failed. Stopping pipeline.")
        return

    print("\nStep 2/4: Running optimisation...")
    if not run_script('optimise.py'):
        print("Optimisation failed. Stopping pipeline.")
        return

    print("\nStep 3/4: Creating visualisations...")
    if not run_script('visualise.py'):
        print("Visualisation failed. Stopping pipeline.")
        return

    print("\nStep 4/4: Generating report...")
    if not run_script('summarise.py'):
        print("Report generation failed. Stopping pipeline.")
        return

    print("\n" + "=" * 50)
    print("Pipeline complete! All outputs saved in 'out' folder:")
    print("  - out/out_calculate/")
    print("  - out/out_optimise/")
    print("  - out/out_visualise/")
    print("  - out/portfolio_report.html")
    print("=" * 50)


if __name__ == "__main__":
    main()