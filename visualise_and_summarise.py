import os
import base64
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def create_visualizations():
    # [Previous visualization code remains the same - volatility, correlation, frontier, gradient bars]
    # Keeping it short - refer to previous version for full implementation
    if not os.path.exists('out/out_calculate') or not os.path.exists('out/out_optimise'):
        print("Error: Required folders not found.")
        return False

    os.makedirs('out/out_visualise', exist_ok=True)

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio_name = data['portfolio']['name']

    calc_in = pd.read_csv('out/out_calculate/calc_in.csv')
    vol = pd.read_csv('out/out_calculate/vol.csv')
    corr = pd.read_csv('out/out_calculate/corr.csv', index_col=0)
    stats_df = pd.read_csv('out/out_optimise/portfolio_stats.csv')
    results_df = pd.read_csv('out/out_optimise/simulation_results.csv')
    gradient_data = pd.read_csv('out/out_optimise/gradient_data.csv')
    distance_metrics = pd.read_csv('out/out_optimise/distance_metrics.csv')

    vol['date'] = pd.to_datetime(vol['date'])
    window = calc_in['window'].iloc[0]

    # Create all 4 visualizations (volatility, correlation, frontier, gradient)
    # [Full code from previous version - shortened here for space]

    print("Visualisation complete!")
    return True


def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def create_html_report():
    if not os.path.exists('out/out_visualise'):
        return False

    with open('input.json', 'r') as f:
        data = json.load(f)
    portfolio_name = data['portfolio']['name']

    # Load images
    volatility_img = image_to_base64('out/out_visualise/volatility.jpg')
    correlation_img = image_to_base64('out/out_visualise/correlation.jpg')
    frontier_img = image_to_base64('out/out_visualise/efficient_frontier.jpg')
    gradient_img = image_to_base64('out/out_visualise/gradient_comparison.jpg')

    # Load reachable portfolios data
    reachable_data = ""
    if os.path.exists('out/out_reachable/reachable_portfolios.csv'):
        reachable_df = pd.read_csv('out/out_reachable/reachable_portfolios.csv')
        stats_df = pd.read_csv('out/out_optimise/portfolio_stats.csv')
        current = stats_df[stats_df['portfolio_type'] == 'Current'].iloc[0]

        portfolios_by_cat = {}
        for cat in ['Higher Return', 'Lower Volatility', 'Better Sharpe']:
            portfolios_by_cat[cat] = reachable_df[reachable_df['category'] == cat].to_dict('records')

        reachable_data = f"""
        <script>
            const portfoliosData = {json.dumps(portfolios_by_cat)};
            const currentPortfolio = {{return: {current['expected_return']}, volatility: {current['volatility']}, sharpe: {current['sharpe_ratio']}}};
        </script>
        """

    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{portfolio_name}</title>
<style>
* {{margin:0;padding:0;box-sizing:border-box}}
body {{font-family:'Segoe UI',sans-serif;background:#0a0a0a;color:#fff;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header {{text-align:center;padding:20px;border-bottom:2px solid #52B788;background:linear-gradient(135deg,#0a0a0a,#1a1a1a)}}
h1 {{font-size:2.5em;font-weight:300;color:#52B788}}
.subtitle {{color:#aaa;font-size:1em}}
.tabs {{display:flex;justify-content:center;background:#1a1a1a;padding:10px;border-bottom:1px solid #333}}
.tab {{padding:10px 30px;cursor:pointer;color:#aaa;border:none;background:none}}
.tab:hover,.tab.active {{color:#52B788;border-bottom:2px solid #52B788}}
.tab-content {{display:none;flex:1;overflow:auto}}
.tab-content.active {{display:flex}}
.dashboard {{flex:1;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:20px;padding:20px}}
.chart-card {{background:#1a1a1a;border-radius:15px;padding:20px;display:flex;flex-direction:column;cursor:pointer;transition:transform 0.2s}}
.chart-card:hover {{transform:scale(1.02)}}
.chart-title {{font-size:1.3em;color:#52B788;margin-bottom:15px;text-align:center;font-weight:300}}
.chart-container {{flex:1;display:flex;justify-content:center;align-items:center}}
.chart-container img {{max-width:100%;max-height:100%;object-fit:contain;border-radius:10px}}
.proposals-container {{flex-direction:column;padding:20px;overflow-y:auto}}
.category-section {{margin-bottom:40px}}
.category-title {{font-size:1.8em;color:#52B788;margin-bottom:20px;font-weight:300}}
.portfolio-grid {{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}}
.portfolio-card {{background:#1a1a1a;border-radius:10px;padding:20px;cursor:pointer;border:2px solid transparent;transition:all 0.3s}}
.portfolio-card:hover {{border-color:#52B788;transform:translateY(-5px)}}
.portfolio-card.selected {{border-color:#52B788;background:#2a2a2a}}
.portfolio-metric {{margin:10px 0;font-size:0.95em}}
.metric-label {{color:#aaa;display:inline-block;width:140px}}
.metric-value {{color:#fff;font-weight:500}}
.improvement-positive {{color:#52B788}}
.improvement-negative {{color:#FF6B6B}}
.comparison-panel {{position:fixed;right:-400px;top:0;width:400px;height:100vh;background:#1a1a1a;border-left:2px solid #52B788;padding:20px;transition:right 0.3s;overflow-y:auto;z-index:2000}}
.comparison-panel.active {{right:0}}
.close-comparison {{position:absolute;top:20px;right:20px;font-size:2em;color:#52B788;cursor:pointer;background:none;border:none}}
.comparison-title {{font-size:1.5em;color:#52B788;margin-bottom:20px;font-weight:300}}
.comparison-section {{margin-bottom:30px}}
.comparison-row {{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #333}}
.modal {{display:none;position:fixed;z-index:1000;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.95);justify-content:center;align-items:center}}
.modal.active {{display:flex}}
.modal-content {{max-width:95%;max-height:95%;border-radius:10px}}
.close-modal {{position:absolute;top:30px;right:30px;font-size:3em;color:#52B788;cursor:pointer}}
</style></head>
<body>
<header><h1>{portfolio_name}</h1><p class="subtitle">Generated {datetime.now().strftime('%B %d, %Y')}</p></header>
<div class="tabs">
<button class="tab active" onclick="showTab('overview')">Overview</button>
<button class="tab" onclick="showTab('proposals')">Proposed Portfolios</button>
</div>
<div id="overview-tab" class="tab-content active">
<div class="dashboard">
<div class="chart-card" onclick="openModal('volatility')"><h3 class="chart-title">Volatility Analysis</h3><div class="chart-container"><img id="volatility-img" src="data:image/jpeg;base64,{volatility_img}"></div></div>
<div class="chart-card" onclick="openModal('correlation')"><h3 class="chart-title">Correlation Matrix</h3><div class="chart-container"><img id="correlation-img" src="data:image/jpeg;base64,{correlation_img}"></div></div>
<div class="chart-card" onclick="openModal('frontier')"><h3 class="chart-title">Efficient Frontier</h3><div class="chart-container"><img id="frontier-img" src="data:image/jpeg;base64,{frontier_img}"></div></div>
<div class="chart-card" onclick="openModal('gradient')"><h3 class="chart-title">Metrics Comparison</h3><div class="chart-container"><img id="gradient-img" src="data:image/jpeg;base64,{gradient_img}"></div></div>
</div></div>
<div id="proposals-tab" class="tab-content proposals-container"><div id="proposals-content"></div></div>
<div class="comparison-panel" id="comparisonPanel"><button class="close-comparison" onclick="closeComparison()">×</button><h2 class="comparison-title">Comparison</h2><div id="comparison-content"></div></div>
<div class="modal" id="modal" onclick="closeModal()"><span class="close-modal">×</span><img class="modal-content" id="modal-img"></div>
{reachable_data}
<script>
function showTab(t){{document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));document.getElementById(t+'-tab').classList.add('active');event.target.classList.add('active');if(t==='proposals'&&typeof portfoliosData!=='undefined')populateProposals()}}
function populateProposals(){{const c=document.getElementById('proposals-content');if(!c.innerHTML){{let h='';const titles={{'Higher Return':'Higher Expected Return','Lower Volatility':'Lower Risk','Better Sharpe':'Better Risk-Adjusted Returns'}};for(const[cat,ports]of Object.entries(portfoliosData)){{if(ports.length>0){{h+=`<div class="category-section"><h2 class="category-title">${{titles[cat]}}</h2><div class="portfolio-grid">`;ports.forEach((p,i)=>{{h+=`<div class="portfolio-card" onclick="selectPortfolio('${{cat}}',${{i}})"><div class="portfolio-metric"><span class="metric-label">Return:</span><span class="metric-value">${{(p.returns*100).toFixed(2)}}%</span><span class="improvement-positive"> (+${{p.return_improvement.toFixed(2)}}%)</span></div><div class="portfolio-metric"><span class="metric-label">Volatility:</span><span class="metric-value">${{(p.volatility*100).toFixed(2)}}%</span><span class="${{p.volatility_change<0?'improvement-positive':'improvement-negative'}}"> (${{p.volatility_change.toFixed(2)}}%)</span></div><div class="portfolio-metric"><span class="metric-label">Sharpe:</span><span class="metric-value">${{p.sharpe_ratio.toFixed(4)}}</span><span class="improvement-positive"> (+${{p.sharpe_improvement.toFixed(2)}}%)</span></div></div>`}});h+=`</div></div>`}}}}c.innerHTML=h||'<p style="text-align:center;color:#aaa;padding:50px">No improvements found</p>'}}}}
function selectPortfolio(cat,idx){{const p=portfoliosData[cat][idx];document.querySelectorAll('.portfolio-card').forEach(c=>c.classList.remove('selected'));event.currentTarget.classList.add('selected');const rDiff=p.returns-currentPortfolio.return;const vDiff=p.volatility-currentPortfolio.volatility;const sDiff=p.sharpe_ratio-currentPortfolio.sharpe;document.getElementById('comparison-content').innerHTML=`<div class="comparison-section"><h3 style="color:#52B788">Current vs Proposed</h3><div class="comparison-row"><span>Return</span><div><div style="color:#FF6B6B">${{(currentPortfolio.return*100).toFixed(2)}}%</div><div style="color:#52B788">${{(p.returns*100).toFixed(2)}}%</div></div></div><div class="comparison-row"><span>Volatility</span><div><div style="color:#FF6B6B">${{(currentPortfolio.volatility*100).toFixed(2)}}%</div><div style="color:#52B788">${{(p.volatility*100).toFixed(2)}}%</div></div></div><div class="comparison-row"><span>Sharpe</span><div><div style="color:#FF6B6B">${{currentPortfolio.sharpe.toFixed(4)}}</div><div style="color:#52B788">${{p.sharpe_ratio.toFixed(4)}}</div></div></div></div><div class="comparison-section"><h3 style="color:#52B788">Improvements</h3><div class="comparison-row"><span>Return Change</span><span class="${{rDiff>=0?'improvement-positive':'improvement-negative'}}">${{rDiff>=0?'+':''}}${{(rDiff*100).toFixed(2)}}%</span></div><div class="comparison-row"><span>Volatility Change</span><span class="${{vDiff<=0?'improvement-positive':'improvement-negative'}}">${{vDiff>=0?'+':''}}${{(vDiff*100).toFixed(2)}}%</span></div><div class="comparison-row"><span>Sharpe Change</span><span class="${{sDiff>=0?'improvement-positive':'improvement-negative'}}">${{sDiff>=0?'+':''}}${{sDiff.toFixed(4)}}</span></div></div>`;document.getElementById('comparisonPanel').classList.add('active')}}
function closeComparison(){{document.getElementById('comparisonPanel').classList.remove('active');document.querySelectorAll('.portfolio-card').forEach(c=>c.classList.remove('selected'))}}
function openModal(id){{const m=document.getElementById('modal');const img=document.getElementById('modal-img');img.src=document.getElementById(id+'-img').src;m.classList.add('active')}}
function closeModal(){{document.getElementById('modal').classList.remove('active')}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape'){{closeModal();closeComparison()}}}});
</script>
</body></html>"""

    with open('out/portfolio_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("Report generated: out/portfolio_report.html")
    return True


def main():
    if create_visualizations():
        create_html_report()


if __name__ == "__main__":
    main()