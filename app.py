from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Import our custom modules
# 导入我们的自定义模块
from scrapers.guangdong_population_scraper import GuangdongPopulationScraper
from utils.data_processor import DataProcessor
from utils.data_analyzer import DataAnalyzer
from utils.visualizer import Visualizer

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Create instances of our classes
# 创建我们的类实例
scraper = GuangdongPopulationScraper()
processor = DataProcessor()
analyzer = DataAnalyzer()
visualizer = Visualizer()

# Ensure data directory exists
# 确保数据目录存在
os.makedirs('data', exist_ok=True)
os.makedirs('data/visualizations', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global variables to store state
# 存储状态的全局变量
current_data = None
current_analysis = None
last_processed_data_path = None
available_regions = []
available_metrics = []
analysis_types = [
    "Overall Population Flow",  # 总体人口流动
    "Inter-city Migration",     # 城市间迁移
    "Urban-Rural Movement",     # 城乡流动
    "Demographic Analysis"      # 人口结构分析
]

@app.route('/')
def index():
    """Home page with tabs for data collection, analysis, and visualization.
    包含数据采集、分析和可视化选项卡的主页。"""
    return render_template('index.html',
                          analysis_types=analysis_types,
                          available_regions=available_regions,
                          available_metrics=available_metrics)

@app.route('/scrape', methods=['POST'])
def scrape_data():
    """Endpoint to trigger data scraping.
    触发数据采集的端点。"""
    global current_data, available_regions, available_metrics, last_processed_data_path

    sources = request.form.getlist('sources')

    if not sources:
        sources = ["Guangdong Statistical Bureau",
                  "China National Bureau of Statistics",
                  "Guangdong Migration Reports",
                  "Academic Research Papers"]

    try:
        # Scrape the data
        raw_data = scraper.scrape_multiple_sources(sources)

        # Automatically process the data after scraping
        current_data = processor.process_data(raw_data)

        # Update available regions and metrics for UI
        if 'region' in current_data.columns:
            available_regions = sorted(current_data['region'].unique().tolist())

        if 'metric' in current_data.columns:
            available_metrics = sorted(current_data['metric'].unique().tolist())

        # Get the path to the most recently processed data
        processed_files = [f for f in os.listdir('data') if f.startswith('processed_data_')]
        if processed_files:
            last_processed_data_path = os.path.join('data', sorted(processed_files)[-1])

        return jsonify({
            'status': 'success',
            'message': f'Successfully scraped and processed {len(raw_data)} data points from {len(sources)} sources.',
            'data_count': len(raw_data),
            'available_regions': available_regions,
            'available_metrics': available_metrics
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error during scraping: {str(e)}'
        }), 500

@app.route('/process', methods=['POST'])
def process_data():
    """Endpoint to process the most recently scraped data.
    处理最近采集的数据的端点。"""
    global current_data, available_regions, available_metrics, last_processed_data_path

    # Get filter parameters
    regions = request.form.getlist('regions')
    if not regions or regions[0] == '':
        regions = None

    start_year = request.form.get('start_year')
    end_year = request.form.get('end_year')
    years = None

    if start_year and end_year:
        try:
            start_year = int(start_year)
            end_year = int(end_year)
            years = range(start_year, end_year + 1)
        except ValueError:
            pass

    try:
        # Find the most recent raw data file
        data_dir = 'data'
        raw_data_files = [f for f in os.listdir(data_dir) if f.startswith('combined_raw_data_')]

        if not raw_data_files:
            # No data files found, check if we should use a sample
            sample_param = request.form.get('use_sample')
            if sample_param and sample_param.lower() == 'true':
                # Process with an empty list to trigger the sample generation
                raw_data = []
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'No raw data files found. Please scrape data first.'
                }), 400
        else:
            # Sort files by name (which includes timestamp) and get the most recent
            most_recent_file = sorted(raw_data_files)[-1]
            file_path = os.path.join(data_dir, most_recent_file)

            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

        # Process the data
        current_data = processor.process_data(raw_data, regions=regions, years=years)

        # Update available regions and metrics for UI
        if 'region' in current_data.columns:
            available_regions = sorted(current_data['region'].unique().tolist())

        if 'metric' in current_data.columns:
            available_metrics = sorted(current_data['metric'].unique().tolist())

        # Get the path to the most recently processed data
        processed_files = [f for f in os.listdir(data_dir) if f.startswith('processed_data_')]
        if processed_files:
            last_processed_data_path = os.path.join(data_dir, sorted(processed_files)[-1])

        return jsonify({
            'status': 'success',
            'message': f'Successfully processed {len(current_data)} data points.',
            'data_count': len(current_data),
            'available_regions': available_regions,
            'available_metrics': available_metrics
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error during data processing: {str(e)}'
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Endpoint to analyze the processed data.
    分析处理后数据的端点。"""
    global current_data, current_analysis

    if current_data is None and last_processed_data_path:
        # Load the last processed data if available
        try:
            current_data = pd.read_csv(last_processed_data_path)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error loading processed data: {str(e)}'
            }), 500

    if current_data is None:
        return jsonify({
            'status': 'error',
            'message': 'No processed data available. Please process data first.'
        }), 400

    # Get analysis parameters
    analysis_type = request.form.get('analysis_type', 'Overall Population Flow')

    regions = request.form.getlist('regions')
    if not regions or regions[0] == '':
        regions = None

    start_year = request.form.get('start_year')
    end_year = request.form.get('end_year')
    years = None

    if start_year and end_year:
        try:
            start_year = int(start_year)
            end_year = int(end_year)
            years = range(start_year, end_year + 1)
        except ValueError:
            pass

    try:
        # Perform the analysis
        current_analysis = analyzer.analyze(current_data, analysis_type=analysis_type,
                                           regions=regions, years=years)

        # Convert analysis results for JSON serialization
        serializable_results = {}
        for key, value in current_analysis.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict(orient='records')
            elif isinstance(value, dict) and any(isinstance(v, pd.DataFrame) for v in value.values()):
                # Handle nested dictionaries with DataFrames
                serializable_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.DataFrame):
                        serializable_results[key][sub_key] = sub_value.to_dict(orient='records')
                    else:
                        serializable_results[key][sub_key] = sub_value
            else:
                serializable_results[key] = value

        return jsonify({
            'status': 'success',
            'message': f'Successfully analyzed data with {analysis_type}.',
            'results': serializable_results
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error during analysis: {str(e)}'
        }), 500

@app.route('/visualize', methods=['POST'])
def create_visualization():
    """Endpoint to create visualizations.
    创建可视化的端点。"""
    global current_data, current_analysis

    if current_data is None:
        return jsonify({
            'status': 'error',
            'message': 'No processed data available. Please process data first.'
        }), 400

    # Get visualization parameters
    viz_type = request.form.get('viz_type', 'flow_map')

    regions = request.form.getlist('regions')
    if not regions or regions[0] == '':
        regions = None

    year = request.form.get('year')
    try:
        year = int(year) if year else None
    except ValueError:
        year = None

    metric = request.form.get('metric', 'Net Migration')

    try:
        # Create the visualization based on type
        if viz_type == 'flow_map':
            if current_analysis is None:
                # Use default analysis if none exists
                analysis_results = analyzer.analyze(current_data, analysis_type="Overall Population Flow",
                                                  regions=regions, years=[year] if year else None)
            else:
                analysis_results = current_analysis

            fig = visualizer.create_flow_map(current_data, analysis_results,
                                           regions=regions, year=year)

        elif viz_type == 'time_series':
            fig = visualizer.create_time_series(current_data, metric=metric,
                                              regions=regions)

        elif viz_type == 'regional_comparison':
            fig = visualizer.create_regional_comparison(current_data,
                                                      comparison_type=metric,
                                                      regions=regions, year=year)

        elif viz_type == 'demographic_breakdown':
            if current_analysis is None:
                analysis_results = analyzer.analyze(current_data, analysis_type="Demographic Analysis",
                                                  regions=regions, years=[year] if year else None)
            else:
                analysis_results = current_analysis

            fig = visualizer.create_demographic_breakdown(current_data, analysis_results,
                                                       demographic_factor=metric, regions=regions)

        elif viz_type == 'flow_direction':
            if current_analysis is None:
                analysis_results = analyzer.analyze(current_data, analysis_type="Inter-city Migration",
                                                  regions=regions, years=[year] if year else None)
            else:
                analysis_results = current_analysis

            fig = visualizer.create_flow_direction_analysis(current_data, analysis_results,
                                                         metric=metric, regions=regions)
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unknown visualization type: {viz_type}'
            }), 400

        # Save visualization to file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"data/visualizations/{viz_type}_{timestamp}.html"
        fig.write_html(filename)

        # Also save to static folder for web access
        static_filename = f"static/{viz_type}_latest.html"
        fig.write_html(static_filename)

        return jsonify({
            'status': 'success',
            'message': f'Successfully created {viz_type} visualization.',
            'viz_path': f'/visualization/{viz_type}_latest.html'
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error creating visualization: {str(e)}'
        }), 500

@app.route('/visualization/<path:filename>')
def serve_visualization(filename):
    """Serve visualization files directly.
    直接提供可视化文件。"""
    return send_from_directory('static', filename)

@app.route('/data/latest')
def get_latest_data():
    """Endpoint to get the latest processed data with pagination support.
    获取最新处理数据的端点，支持分页。"""
    global current_data

    # Get pagination parameters
    page = request.args.get('page', default=1, type=int)
    page_size = request.args.get('page_size', default=10, type=int)

    # Ensure valid pagination parameters
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:  # Limit max page size
        page_size = 10

    if current_data is None and last_processed_data_path:
        # Load the last processed data if available
        try:
            current_data = pd.read_csv(last_processed_data_path)
        except Exception as e:
            print(f"Error loading processed data: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error loading processed data: {str(e)}'
            }), 500

    # If still no data, check if we have raw data to process
    if current_data is None:
        try:
            # Find the most recent raw data file
            data_dir = 'data'
            raw_data_files = [f for f in os.listdir(data_dir) if f.startswith('combined_raw_data_')]

            if raw_data_files:
                # Sort files by name (which includes timestamp) and get the most recent
                most_recent_file = sorted(raw_data_files)[-1]
                file_path = os.path.join(data_dir, most_recent_file)

                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)

                # Process the data
                current_data = processor.process_data(raw_data)
                print(f"Processed data from {most_recent_file}, got {len(current_data)} records")
            else:
                # No raw data files found, return error
                return jsonify({
                    'status': 'error',
                    'message': 'No data available. Please scrape and process data first.'
                }), 404
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f'Error processing data: {str(e)}'
            }), 500

    # Handle NaN values and convert to JSON
    # First replace NaN values with None (which will become null in JSON)
    current_data = current_data.replace({np.nan: None})

    # Calculate pagination values
    total_records = len(current_data)
    total_pages = (total_records + page_size - 1) // page_size  # Ceiling division

    # Adjust page if it exceeds total pages
    if page > total_pages and total_pages > 0:
        page = total_pages

    # Calculate start and end indices for the current page
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_records)

    # Get the data for the current page
    page_data = current_data.iloc[start_idx:end_idx]

    # Convert to JSON-safe format
    try:
        data_json = page_data.to_dict(orient='records')

        # Additional check to ensure all values are JSON serializable
        for record in data_json:
            for key, value in record.items():
                # Handle any remaining non-serializable values
                if pd.isna(value):
                    record[key] = None

        return jsonify({
            'status': 'success',
            'data': data_json,
            'columns': current_data.columns.tolist(),
            'rows': total_records,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_pages': total_pages,
                'total_records': total_records,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        })
    except Exception as e:
        import traceback
        print(f"Error serializing data to JSON: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error preparing data for display: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create templates directory and add basic templates
    os.makedirs('templates', exist_ok=True)

    # Create a simple index.html template if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Guangdong Population Flow Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #121212;
            color: #e0e0e0;
        }
        .nav-tabs .nav-link {
            color: #adb5bd;
        }
        .nav-tabs .nav-link.active {
            background-color: #2d2d2d;
            color: #ffffff;
            border-color: #495057;
        }
        .card {
            background-color: #2d2d2d;
            border-color: #495057;
        }
        .form-control, .form-select {
            background-color: #3d3d3d;
            color: #e0e0e0;
            border-color: #495057;
        }
        .form-control:focus, .form-select:focus {
            background-color: #3d3d3d;
            color: #e0e0e0;
        }
        .btn-primary {
            background-color: #007bff;
            border-color: #007bff;
        }
        .alert-danger {
            background-color: #350d0d;
            color: #f8d7da;
            border-color: #842029;
        }
        .alert-success {
            background-color: #0d1b0d;
            color: #d1e7dd;
            border-color: #0f5132;
        }
        .source-badge {
            margin-right: 5px;
            margin-bottom: 5px;
            display: inline-block;
        }
        #visualization-container {
            width: 100%;
            height: 600px;
            border: none;
        }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <h1 class="text-center mb-4">Guangdong Population Flow Analysis</h1>

        <ul class="nav nav-tabs mb-4" id="mainTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="data-collection-tab" data-bs-toggle="tab" data-bs-target="#data-collection" type="button" role="tab" aria-controls="data-collection" aria-selected="true">Data Collection & Processing</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="analysis-tab" data-bs-toggle="tab" data-bs-target="#analysis" type="button" role="tab" aria-controls="analysis" aria-selected="false">Analysis Results</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="visualization-tab" data-bs-toggle="tab" data-bs-target="#visualization" type="button" role="tab" aria-controls="visualization" aria-selected="false">Visualization</button>
            </li>
        </ul>

        <div class="tab-content" id="mainTabContent">
            <!-- Data Collection & Processing Tab -->
            <div class="tab-pane fade show active" id="data-collection" role="tabpanel" aria-labelledby="data-collection-tab">
                <div class="row">
                    <!-- Left sidebar with controls -->
                    <div class="col-md-3">
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5>Controls</h5>
                            </div>
                            <div class="card-body">
                                <h6>Data Collection</h6>
                                <button id="scrape-btn" class="btn btn-primary mb-3 w-100">Scrape New Data</button>

                                <div class="mb-3">
                                    <label>Select Data Sources:</label>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" value="Guangdong Statistical Bureau" id="source-gsb" checked>
                                        <label class="form-check-label" for="source-gsb">
                                            Guangdong Statistical Bureau
                                        </label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" value="China National Bureau of Statistics" id="source-cnbs" checked>
                                        <label class="form-check-label" for="source-cnbs">
                                            China National Bureau of Statistics
                                        </label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" value="Guangdong Migration Reports" id="source-gmr" checked>
                                        <label class="form-check-label" for="source-gmr">
                                            Guangdong Migration Reports
                                        </label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" value="Academic Research Papers" id="source-arp" checked>
                                        <label class="form-check-label" for="source-arp">
                                            Academic Research Papers
                                        </label>
                                    </div>
                                </div>

                                <h6 class="mt-4">Data Processing</h6>
                                <button id="process-btn" class="btn btn-primary mb-3 w-100">Process Data</button>

                                <div class="mb-3">
                                    <label for="process-regions" class="form-label">Filter by Regions:</label>
                                    <select id="process-regions" class="form-select" multiple>
                                        {% for region in available_regions %}
                                        <option value="{{ region }}">{{ region }}</option>
                                        {% endfor %}
                                    </select>
                                </div>

                                <div class="mb-3">
                                    <label for="process-year-range" class="form-label">Year Range:</label>
                                    <div class="row">
                                        <div class="col-6">
                                            <input type="number" class="form-control" id="process-start-year" placeholder="Start">
                                        </div>
                                        <div class="col-6">
                                            <input type="number" class="form-control" id="process-end-year" placeholder="End">
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Main content area -->
                    <div class="col-md-9">
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5>Data Status</h5>
                            </div>
                            <div class="card-body">
                                <div id="data-alerts" class="mb-3"></div>

                                <div class="mb-4">
                                    <h6>Data Collection Status</h6>
                                    <div class="progress mb-2">
                                        <div id="scrape-progress" class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
                                    </div>
                                    <div id="scrape-status">No data scraped yet.</div>
                                </div>

                                <div class="mb-4">
                                    <h6>Data Processing Status</h6>
                                    <div class="progress mb-2">
                                        <div id="process-progress" class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
                                    </div>
                                    <div id="process-status">No data processed yet.</div>
                                </div>

                                <div id="data-preview-container">
                                    <h6>Data Preview</h6>
                                    <div class="table-responsive">
                                        <table id="data-preview-table" class="table table-dark table-striped">
                                            <thead>
                                                <tr>
                                                    <th>Region</th>
                                                    <th>Year</th>
                                                    <th>Metric</th>
                                                    <th>Value</th>
                                                    <th>Source</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr>
                                                    <td colspan="5" class="text-center">No data available. Please process data first.</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Analysis Results Tab -->
            <div class="tab-pane fade" id="analysis" role="tabpanel" aria-labelledby="analysis-tab">
                <div class="row">
                    <!-- Left sidebar with controls -->
                    <div class="col-md-3">
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5>Analysis Parameters</h5>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <label for="analysis-type" class="form-label">Select Analysis Type:</label>
                                    <select id="analysis-type" class="form-select">
                                        {% for analysis_type in analysis_types %}
                                        <option value="{{ analysis_type }}">{{ analysis_type }}</option>
                                        {% endfor %}
                                    </select>
                                </div>

                                <div class="mb-3">
                                    <label for="analysis-regions" class="form-label">Filter by Regions:</label>
                                    <select id="analysis-regions" class="form-select" multiple>
                                        {% for region in available_regions %}
                                        <option value="{{ region }}">{{ region }}</option>
                                        {% endfor %}
                                    </select>
                                </div>

                                <div class="mb-3">
                                    <label for="analysis-year-range" class="form-label">Year Range:</label>
                                    <div class="row">
                                        <div class="col-6">
                                            <input type="number" class="form-control" id="analysis-start-year" placeholder="Start">
                                        </div>
                                        <div class="col-6">
                                            <input type="number" class="form-control" id="analysis-end-year" placeholder="End">
                                        </div>
                                    </div>
                                </div>

                                <button id="run-analysis-btn" class="btn btn-primary w-100">Run Analysis</button>
                            </div>
                        </div>
                    </div>

                    <!-- Main content area -->
                    <div class="col-md-9">
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5>Analysis Results</h5>
                            </div>
                            <div class="card-body">
                                <div id="analysis-alerts" class="mb-3"></div>

                                <div id="analysis-overview" class="mb-4">
                                    <h6>Overview</h6>
                                    <div id="overview-content">
                                        <p>No analysis results available. Please run an analysis first.</p>
                                    </div>
                                </div>

                                <div id="analysis-insights" class="mb-4">
                                    <h6>Key Insights</h6>
                                    <ul id="insights-list">
                                        <li>Run an analysis to see insights.</li>
                                    </ul>
                                </div>

                                <div id="analysis-details" class="mb-4">
                                    <h6>Detailed Results</h6>
                                    <div id="details-content">
                                        <p>Run an analysis to see detailed results.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Visualization Tab -->
            <div class="tab-pane fade" id="visualization" role="tabpanel" aria-labelledby="visualization-tab">
                <div class="row">
                    <!-- Left sidebar with controls -->
                    <div class="col-md-3">
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5>Visualization Settings</h5>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <label for="viz-type" class="form-label">Visualization Type:</label>
                                    <select id="viz-type" class="form-select">
                                        <option value="flow_map">Population Flow Map</option>
                                        <option value="time_series">Time Series Analysis</option>
                                        <option value="regional_comparison">Regional Comparison</option>
                                        <option value="demographic_breakdown">Demographic Breakdown</option>
                                        <option value="flow_direction">Flow Direction Analysis</option>
                                    </select>
                                </div>

                                <div class="mb-3">
                                    <label for="viz-metric" class="form-label">Metric:</label>
                                    <select id="viz-metric" class="form-select">
                                        {% for metric in available_metrics %}
                                        <option value="{{ metric }}">{{ metric }}</option>
                                        {% endfor %}
                                    </select>
                                </div>

                                <div class="mb-3">
                                    <label for="viz-regions" class="form-label">Filter by Regions:</label>
                                    <select id="viz-regions" class="form-select" multiple>
                                        {% for region in available_regions %}
                                        <option value="{{ region }}">{{ region }}</option>
                                        {% endfor %}
                                    </select>
                                </div>

                                <div class="mb-3">
                                    <label for="viz-year" class="form-label">Year:</label>
                                    <input type="number" class="form-control" id="viz-year" placeholder="Year">
                                </div>

                                <button id="generate-viz-btn" class="btn btn-primary w-100">Generate Visualization</button>
                            </div>
                        </div>
                    </div>

                    <!-- Main content area -->
                    <div class="col-md-9">
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5 id="viz-title">Visualization</h5>
                            </div>
                            <div class="card-body">
                                <div id="viz-alerts" class="mb-3"></div>

                                <iframe id="visualization-container" src="about:blank"></iframe>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        $(document).ready(function() {
            // Scrape data button click
            $('#scrape-btn').click(function() {
                // Get selected sources
                const sources = [];
                $('.form-check-input:checked').each(function() {
                    sources.push($(this).val());
                });

                // Reset progress and status
                $('#scrape-progress').css('width', '50%').attr('aria-valuenow', 50).text('50%');
                $('#scrape-status').text('Scraping data...');
                $('#data-alerts').html('');

                // Send scrape request
                $.ajax({
                    url: '/scrape',
                    type: 'POST',
                    data: { sources: sources },
                    success: function(response) {
                        $('#scrape-progress').css('width', '100%').attr('aria-valuenow', 100).text('100%');
                        $('#scrape-status').text(response.message);
                        $('#data-alerts').html(`<div class="alert alert-success">${response.message}</div>`);
                    },
                    error: function(xhr) {
                        const response = xhr.responseJSON || { message: 'Error scraping data.' };
                        $('#scrape-progress').css('width', '0%').attr('aria-valuenow', 0).text('0%');
                        $('#scrape-status').text('Scraping failed.');
                        $('#data-alerts').html(`<div class="alert alert-danger">${response.message}</div>`);
                    }
                });
            });

            // Process data button click
            $('#process-btn').click(function() {
                // Get filter parameters
                const regions = $('#process-regions').val();
                const startYear = $('#process-start-year').val();
                const endYear = $('#process-end-year').val();

                // Reset progress and status
                $('#process-progress').css('width', '50%').attr('aria-valuenow', 50).text('50%');
                $('#process-status').text('Processing data...');
                $('#data-alerts').html('');

                // Send process request
                $.ajax({
                    url: '/process',
                    type: 'POST',
                    data: {
                        regions: regions,
                        start_year: startYear,
                        end_year: endYear,
                        use_sample: true  // Use sample data if no raw data is available
                    },
                    success: function(response) {
                        $('#process-progress').css('width', '100%').attr('aria-valuenow', 100).text('100%');
                        $('#process-status').text(response.message);
                        $('#data-alerts').html(`<div class="alert alert-success">${response.message}</div>`);

                        // Update available regions and metrics
                        if (response.available_regions) {
                            updateSelectOptions('process-regions', response.available_regions);
                            updateSelectOptions('analysis-regions', response.available_regions);
                            updateSelectOptions('viz-regions', response.available_regions);
                        }

                        if (response.available_metrics) {
                            updateSelectOptions('viz-metric', response.available_metrics);
                        }

                        // Load data preview
                        loadDataPreview();
                    },
                    error: function(xhr) {
                        const response = xhr.responseJSON || { message: 'Error processing data.' };
                        $('#process-progress').css('width', '0%').attr('aria-valuenow', 0).text('0%');
                        $('#process-status').text('Processing failed.');
                        $('#data-alerts').html(`<div class="alert alert-danger">${response.message}</div>`);
                    }
                });
            });

            // Run analysis button click
            $('#run-analysis-btn').click(function() {
                // Get analysis parameters
                const analysisType = $('#analysis-type').val();
                const regions = $('#analysis-regions').val();
                const startYear = $('#analysis-start-year').val();
                const endYear = $('#analysis-end-year').val();

                // Reset and show loading status
                $('#analysis-alerts').html('');
                $('#overview-content').html('<p>Loading analysis results...</p>');
                $('#insights-list').html('<li>Loading insights...</li>');
                $('#details-content').html('<p>Loading detailed results...</p>');

                // Send analysis request
                $.ajax({
                    url: '/analyze',
                    type: 'POST',
                    data: {
                        analysis_type: analysisType,
                        regions: regions,
                        start_year: startYear,
                        end_year: endYear
                    },
                    success: function(response) {
                        $('#analysis-alerts').html(`<div class="alert alert-success">${response.message}</div>`);

                        // Display results
                        displayAnalysisResults(response.results);
                    },
                    error: function(xhr) {
                        const response = xhr.responseJSON || { message: 'Error running analysis.' };
                        $('#analysis-alerts').html(`<div class="alert alert-danger">${response.message}</div>`);
                        $('#overview-content').html('<p>Analysis failed. Please check the error message above.</p>');
                        $('#insights-list').html('<li>No insights available due to analysis failure.</li>');
                        $('#details-content').html('<p>No detailed results available.</p>');
                    }
                });
            });

            // Generate visualization button click
            $('#generate-viz-btn').click(function() {
                // Get visualization parameters
                const vizType = $('#viz-type').val();
                const metric = $('#viz-metric').val();
                const regions = $('#viz-regions').val();
                const year = $('#viz-year').val();

                // Reset and show loading status
                $('#viz-alerts').html('');
                $('#viz-title').text(`${vizType.replace('_', ' ').toTitleCase()} (Loading...)`);
                $('#visualization-container').attr('src', 'about:blank');

                // Send visualization request
                $.ajax({
                    url: '/visualize',
                    type: 'POST',
                    data: {
                        viz_type: vizType,
                        metric: metric,
                        regions: regions,
                        year: year
                    },
                    success: function(response) {
                        $('#viz-alerts').html(`<div class="alert alert-success">${response.message}</div>`);
                        $('#viz-title').text(vizType.replace('_', ' ').toTitleCase());

                        // Load visualization in iframe
                        $('#visualization-container').attr('src', response.viz_path);
                    },
                    error: function(xhr) {
                        const response = xhr.responseJSON || { message: 'Error generating visualization.' };
                        $('#viz-alerts').html(`<div class="alert alert-danger">${response.message}</div>`);
                        $('#viz-title').text('Visualization (Failed)');
                    }
                });
            });

            // Helper function to load data preview
            function loadDataPreview() {
                console.log("Loading data preview...");
                $('#data-preview-table tbody').html(
                    '<tr><td colspan="5" class="text-center">Loading data...</td></tr>'
                );

                $.ajax({
                    url: '/data/latest',
                    type: 'GET',
                    success: function(response) {
                        console.log("Data received:", response);
                        // Clear existing table data
                        $('#data-preview-table tbody').empty();

                        // Check for valid response
                        if (response.status === 'success' && response.data && Array.isArray(response.data)) {
                            // Add first 10 rows to table
                            const data = response.data.slice(0, 10);
                            if (data.length > 0) {
                                data.forEach(function(row) {
                                    let rowHtml = '<tr>';
                                    rowHtml += `<td>${row.region || '-'}</td>`;
                                    rowHtml += `<td>${row.year || '-'}</td>`;
                                    rowHtml += `<td>${row.metric || '-'}</td>`;
                                    rowHtml += `<td>${row.value !== undefined ? Number(row.value).toFixed(2) : '-'}</td>`;
                                    rowHtml += `<td>${row.source || '-'}</td>`;
                                    rowHtml += '</tr>';
                                    $('#data-preview-table tbody').append(rowHtml);
                                });

                                // Add a note if there are more rows
                                if (response.rows > 10) {
                                    $('#data-preview-table tbody').append(
                                        `<tr><td colspan="5" class="text-center">Showing 10 of ${response.rows} rows</td></tr>`
                                    );
                                }

                                // Show success message
                                $('#data-alerts').html(`<div class="alert alert-success">Data loaded successfully. Showing ${data.length} of ${response.rows} records.</div>`);
                            } else {
                                $('#data-preview-table tbody').html(
                                    '<tr><td colspan="5" class="text-center">No data available. The dataset is empty.</td></tr>'
                                );
                            }
                        } else {
                            console.error("Invalid response format:", response);
                            $('#data-preview-table tbody').html(
                                '<tr><td colspan="5" class="text-center">Error: Received invalid data format from server.</td></tr>'
                            );
                        }
                    },
                    error: function(xhr, status, error) {
                        console.error("Error loading data:", status, error);
                        $('#data-preview-table tbody').html(
                            `<tr><td colspan="5" class="text-center">Error loading data: ${error}. Please process data first.</td></tr>`
                        );
                    }
                });
            }

            // Helper function to display analysis results
            function displayAnalysisResults(results) {
                // Display overview
                if (results.overview) {
                    let overviewHtml = '<div class="row">';

                    // Regions count
                    if (results.overview.regions_count !== undefined) {
                        overviewHtml += '<div class="col-md-4 mb-3">';
                        overviewHtml += '<div class="card bg-dark">';
                        overviewHtml += '<div class="card-body">';
                        overviewHtml += '<h5 class="card-title">Regions</h5>';
                        overviewHtml += `<p class="card-text display-4">${results.overview.regions_count}</p>`;
                        overviewHtml += '</div></div></div>';
                    }

                    // Years span
                    if (results.overview.years_span) {
                        overviewHtml += '<div class="col-md-4 mb-3">';
                        overviewHtml += '<div class="card bg-dark">';
                        overviewHtml += '<div class="card-body">';
                        overviewHtml += '<h5 class="card-title">Years Covered</h5>';
                        overviewHtml += `<p class="card-text display-4">${results.overview.years_span[0]} - ${results.overview.years_span[1]}</p>`;
                        overviewHtml += '</div></div></div>';
                    }

                    // Total data points
                    if (results.overview.metrics_distribution) {
                        const totalPoints = Object.values(results.overview.metrics_distribution).reduce((a, b) => a + b, 0);
                        overviewHtml += '<div class="col-md-4 mb-3">';
                        overviewHtml += '<div class="card bg-dark">';
                        overviewHtml += '<div class="card-body">';
                        overviewHtml += '<h5 class="card-title">Data Points</h5>';
                        overviewHtml += `<p class="card-text display-4">${totalPoints}</p>`;
                        overviewHtml += '</div></div></div>';
                    }

                    overviewHtml += '</div>';

                    // Metrics distribution
                    if (results.overview.metrics_distribution) {
                        overviewHtml += '<div class="mt-3">';
                        overviewHtml += '<h6>Metrics Distribution</h6>';
                        overviewHtml += '<div class="mb-2">';

                        Object.entries(results.overview.metrics_distribution).forEach(([metric, count]) => {
                            overviewHtml += `<span class="badge bg-primary source-badge">${metric}: ${count}</span>`;
                        });

                        overviewHtml += '</div></div>';
                    }

                    // Regions list
                    if (results.overview.regions && results.overview.regions.length > 0) {
                        overviewHtml += '<div class="mt-3">';
                        overviewHtml += '<h6>Regions</h6>';
                        overviewHtml += '<div class="mb-2">';

                        results.overview.regions.forEach(region => {
                            overviewHtml += `<span class="badge bg-secondary source-badge">${region}</span>`;
                        });

                        overviewHtml += '</div></div>';
                    }

                    $('#overview-content').html(overviewHtml);
                } else {
                    $('#overview-content').html('<p>No overview information available.</p>');
                }

                // Display insights
                if (results.insights && results.insights.length > 0) {
                    let insightsHtml = '';
                    results.insights.forEach(insight => {
                        insightsHtml += `<li>${insight}</li>`;
                    });
                    $('#insights-list').html(insightsHtml);
                } else {
                    $('#insights-list').html('<li>No insights available.</li>');
                }

                // Display detailed results
                let detailsHtml = '';

                // Top regions
                if (results.top_regions) {
                    if (results.top_regions.inflow && results.top_regions.inflow.length > 0) {
                        detailsHtml += '<div class="mb-4">';
                        detailsHtml += '<h6>Top Regions by Immigration</h6>';
                        detailsHtml += '<div class="table-responsive">';
                        detailsHtml += '<table class="table table-dark table-striped">';
                        detailsHtml += '<thead><tr><th>Region</th><th>Immigration Value</th></tr></thead>';
                        detailsHtml += '<tbody>';

                        results.top_regions.inflow.forEach(row => {
                            detailsHtml += `<tr><td>${row.region}</td><td>${row.value.toFixed(2)}</td></tr>`;
                        });

                        detailsHtml += '</tbody></table></div></div>';
                    }

                    if (results.top_regions.outflow && results.top_regions.outflow.length > 0) {
                        detailsHtml += '<div class="mb-4">';
                        detailsHtml += '<h6>Top Regions by Emigration</h6>';
                        detailsHtml += '<div class="table-responsive">';
                        detailsHtml += '<table class="table table-dark table-striped">';
                        detailsHtml += '<thead><tr><th>Region</th><th>Emigration Value</th></tr></thead>';
                        detailsHtml += '<tbody>';

                        results.top_regions.outflow.forEach(row => {
                            detailsHtml += `<tr><td>${row.region}</td><td>${row.value.toFixed(2)}</td></tr>`;
                        });

                        detailsHtml += '</tbody></table></div></div>';
                    }
                }

                // Statistical data
                if (results.statistical_data && results.statistical_data.length > 0) {
                    detailsHtml += '<div class="mb-4">';
                    detailsHtml += '<h6>Statistical Data</h6>';
                    detailsHtml += '<div class="table-responsive">';
                    detailsHtml += '<table class="table table-dark table-striped">';

                    // Create header row
                    detailsHtml += '<thead><tr>';
                    Object.keys(results.statistical_data[0]).forEach(key => {
                        detailsHtml += `<th>${key}</th>`;
                    });
                    detailsHtml += '</tr></thead>';

                    // Create data rows
                    detailsHtml += '<tbody>';
                    results.statistical_data.forEach(row => {
                        detailsHtml += '<tr>';
                        Object.values(row).forEach(value => {
                            detailsHtml += `<td>${typeof value === 'number' ? value.toFixed(2) : value}</td>`;
                        });
                        detailsHtml += '</tr>';
                    });

                    detailsHtml += '</tbody></table></div></div>';
                }

                // Flow matrix
                if (results.flow_matrix) {
                    detailsHtml += '<div class="mb-4">';
                    detailsHtml += '<h6>Flow Matrix</h6>';
                    detailsHtml += '<p>This data would be better visualized with the Flow Map visualization.</p>';
                    detailsHtml += '</div>';
                }

                // Correlation matrix
                if (results.correlation_matrix) {
                    detailsHtml += '<div class="mb-4">';
                    detailsHtml += '<h6>Correlation Matrix</h6>';
                    detailsHtml += '<div class="table-responsive">';
                    detailsHtml += '<table class="table table-dark table-striped">';

                    // Create header row
                    detailsHtml += '<thead><tr><th>Metric</th>';
                    Object.keys(results.correlation_matrix).forEach(key => {
                        detailsHtml += `<th>${key}</th>`;
                    });
                    detailsHtml += '</tr></thead>';

                    // Create data rows
                    detailsHtml += '<tbody>';
                    Object.entries(results.correlation_matrix).forEach(([metric, correlations]) => {
                        detailsHtml += `<tr><td>${metric}</td>`;
                        Object.values(correlations).forEach(value => {
                            // Color code based on correlation strength
                            let cellColor = '';
                            if (value > 0.7) cellColor = 'background-color: rgba(0, 255, 0, 0.2);';
                            else if (value > 0.4) cellColor = 'background-color: rgba(0, 255, 0, 0.1);';
                            else if (value < -0.7) cellColor = 'background-color: rgba(255, 0, 0, 0.2);';
                            else if (value < -0.4) cellColor = 'background-color: rgba(255, 0, 0, 0.1);';

                            detailsHtml += `<td style="${cellColor}">${value.toFixed(2)}</td>`;
                        });
                        detailsHtml += '</tr>';
                    });

                    detailsHtml += '</tbody></table></div></div>';
                }

                if (detailsHtml === '') {
                    detailsHtml = '<p>No detailed results available.</p>';
                }

                $('#details-content').html(detailsHtml);
            }

            // Helper function to update select options
            function updateSelectOptions(selectId, options) {
                const select = $(`#${selectId}`);
                select.empty();

                options.forEach(option => {
                    select.append(new Option(option, option));
                });
            }

            // Extend String prototype to convert a string to title case
            String.prototype.toTitleCase = function() {
                return this.replace(/\\w\\S*/g, function(txt) {
                    return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
                });
            };
        });
    </script>
</body>
</html>
            ''')

    # Create static directory for visualizations
    os.makedirs('static', exist_ok=True)

    # Create a default visualization if it doesn't exist
    if not os.path.exists('static/flow_map_latest.html'):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(
            text="No visualization generated yet. Please use the visualization controls to create one.",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Guangdong Population Flow Analysis",
            template="plotly_dark",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#2d2d2d',
            paper_bgcolor='#2d2d2d',
            font=dict(color='#e0e0e0')
        )
        fig.write_html('static/flow_map_latest.html')

    app.run(host='0.0.0.0', port=5000, debug=True)