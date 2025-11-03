import os
import base64
import json
from datetime import datetime


def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def main():
    if not os.path.exists('out/out_visualise'):
        print("Error: 'out/out_visualise' folder not found. Run all scripts first.")
        return

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio_name = data['portfolio']['name']

    volatility_img = image_to_base64('out/out_visualise/volatility.jpg')
    correlation_img = image_to_base64('out/out_visualise/correlation.jpg')
    frontier_img = image_to_base64('out/out_visualise/efficient_frontier.jpg')

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{portfolio_name} - Portfolio Analysis Deck</title>
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
                overflow: hidden;
                height: 100vh;
                display: flex;
                flex-direction: column;
            }}

            header {{
                text-align: center;
                padding: 30px 0 20px 0;
                border-bottom: 2px solid #52B788;
                background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            }}

            header h1 {{
                font-size: 2.5em;
                font-weight: 300;
                color: #52B788;
                margin-bottom: 5px;
            }}

            header .subtitle {{
                font-size: 1em;
                color: #aaaaaa;
                font-weight: 300;
            }}

            .dashboard {{
                flex: 1;
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 20px;
                padding: 20px;
                overflow: hidden;
            }}

            .chart-card {{
                background-color: #1a1a1a;
                border-radius: 15px;
                padding: 20px;
                display: flex;
                flex-direction: column;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                cursor: pointer;
                transition: transform 0.2s ease;
            }}

            .chart-card:hover {{
                transform: scale(1.02);
            }}

            .chart-card.top-left {{
                grid-column: 1;
                grid-row: 1 / 3;
            }}

            .chart-card.middle {{
                grid-column: 2;
                grid-row: 1 / 3;
            }}

            .chart-card.top-right {{
                grid-column: 3;
                grid-row: 1 / 3;
            }}

            .chart-title {{
                font-size: 1.3em;
                font-weight: 300;
                color: #52B788;
                margin-bottom: 15px;
                text-align: center;
            }}

            .chart-container {{
                flex: 1;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
            }}

            .chart-container img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                border-radius: 10px;
            }}

            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.95);
                justify-content: center;
                align-items: center;
            }}

            .modal.active {{
                display: flex;
            }}

            .modal-content {{
                max-width: 95%;
                max-height: 95%;
                object-fit: contain;
                border-radius: 10px;
            }}

            .close-modal {{
                position: absolute;
                top: 30px;
                right: 30px;
                font-size: 3em;
                color: #52B788;
                cursor: pointer;
                font-weight: 300;
                transition: transform 0.2s ease;
            }}

            .close-modal:hover {{
                transform: scale(1.2);
            }}

            @media (max-width: 1200px) {{
                .dashboard {{
                    grid-template-columns: 1fr;
                    grid-template-rows: repeat(3, 1fr);
                }}

                .chart-card.top-left,
                .chart-card.middle,
                .chart-card.top-right {{
                    grid-column: 1;
                    grid-row: auto;
                }}
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>{portfolio_name}</h1>
            <p class="subtitle">Portfolio Analysis Dashboard | Generated on {datetime.now().strftime('%B %d, %Y')}</p>
        </header>

        <div class="dashboard">
            <div class="chart-card top-left" onclick="openModal('volatility')">
                <h3 class="chart-title">Portfolio Volatility Analysis</h3>
                <div class="chart-container">
                    <img id="volatility-img" src="data:image/jpeg;base64,{volatility_img}" alt="Volatility Chart">
                </div>
            </div>

            <div class="chart-card middle" onclick="openModal('correlation')">
                <h3 class="chart-title">Asset Correlation Matrix</h3>
                <div class="chart-container">
                    <img id="correlation-img" src="data:image/jpeg;base64,{correlation_img}" alt="Correlation Matrix">
                </div>
            </div>

            <div class="chart-card top-right" onclick="openModal('frontier')">
                <h3 class="chart-title">Efficient Frontier & Portfolio Optimization</h3>
                <div class="chart-container">
                    <img id="frontier-img" src="data:image/jpeg;base64,{frontier_img}" alt="Efficient Frontier">
                </div>
            </div>
        </div>

        <div class="modal" id="modal" onclick="closeModal()">
            <span class="close-modal">&times;</span>
            <img class="modal-content" id="modal-img">
        </div>

        <script>
            function openModal(imageId) {{
                const modal = document.getElementById('modal');
                const modalImg = document.getElementById('modal-img');
                const img = document.getElementById(imageId + '-img');

                modal.classList.add('active');
                modalImg.src = img.src;
            }}

            function closeModal() {{
                const modal = document.getElementById('modal');
                modal.classList.remove('active');
            }}

            document.addEventListener('keydown', (e) => {{
                if (e.key === 'Escape') closeModal();
            }});
        </script>
    </body>
    </html>
    """

    output_path = 'out/portfolio_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()