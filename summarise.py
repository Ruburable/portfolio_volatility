import os
import base64
import json
from datetime import datetime


def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def main():
    if not os.path.exists('out/out_visualise') or not os.path.exists('out/out_optimise'):
        print("Error: Required output folders not found. Run calculate, optimise, and visualise first.")
        return

    os.makedirs('out', exist_ok=True)

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio_name = data['portfolio']['name']

    volatility_img = image_to_base64('out/out_visualise/volatility.jpg')
    correlation_img = image_to_base64('out/out_visualise/correlation.jpg')
    frontier_img = image_to_base64('out/out_optimise/efficient_frontier.jpg')

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{portfolio_name} - Portfolio Analysis</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0a0a0a;
                color: #ffffff;
                line-height: 1.6;
            }}

            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 40px 20px;
            }}

            header {{
                text-align: center;
                margin-bottom: 60px;
                padding: 40px 0;
                border-bottom: 2px solid #52B788;
            }}

            h1 {{
                font-size: 3em;
                font-weight: 300;
                margin-bottom: 10px;
                color: #52B788;
            }}

            .subtitle {{
                font-size: 1.2em;
                color: #aaaaaa;
                font-weight: 300;
            }}

            .section {{
                margin-bottom: 80px;
            }}

            .section-title {{
                font-size: 2em;
                font-weight: 300;
                margin-bottom: 30px;
                color: #52B788;
                border-left: 4px solid #52B788;
                padding-left: 20px;
            }}

            .chart-container {{
                background-color: #1a1a1a;
                border-radius: 10px;
                padding: 30px;
                margin-bottom: 40px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }}

            .chart-container img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
            }}

            .two-column {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }}

            footer {{
                text-align: center;
                padding: 40px 0;
                border-top: 1px solid #333;
                color: #888;
                font-size: 0.9em;
            }}

            @media (max-width: 768px) {{
                .two-column {{
                    grid-template-columns: 1fr;
                }}

                h1 {{
                    font-size: 2em;
                }}

                .section-title {{
                    font-size: 1.5em;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>{portfolio_name}</h1>
                <p class="subtitle">Portfolio Analysis Report</p>
                <p class="subtitle">Generated on {datetime.now().strftime('%B %d, %Y')}</p>
            </header>

            <div class="section">
                <h2 class="section-title">Portfolio Volatility Analysis</h2>
                <div class="chart-container">
                    <img src="data:image/jpeg;base64,{volatility_img}" alt="Volatility Chart">
                </div>
            </div>

            <div class="section">
                <h2 class="section-title">Asset Correlation Matrix</h2>
                <div class="chart-container">
                    <img src="data:image/jpeg;base64,{correlation_img}" alt="Correlation Matrix">
                </div>
            </div>

            <div class="section">
                <h2 class="section-title">Efficient Frontier & Portfolio Optimization</h2>
                <div class="chart-container">
                    <img src="data:image/jpeg;base64,{frontier_img}" alt="Efficient Frontier">
                </div>
            </div>

            <footer>
                <p>Portfolio Analysis Dashboard | Modern Portfolio Theory | Monte Carlo Simulation</p>
            </footer>
        </div>
    </body>
    </html>
    """

    output_path = 'out/portfolio_report.html'
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()