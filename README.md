# Guangdong Population Flow Analysis

A data analysis and visualization application for studying population flow patterns in Guangdong Province, China.

## Features

- **Data Collection**: Scrape population data from multiple sources
- **Data Processing**: Clean and transform raw data for analysis
- **Analysis**: Perform various analyses on population flow data
  - Overall Population Flow
  - Inter-city Migration
  - Urban-Rural Movement
  - Demographic Analysis
- **Visualization**: Generate interactive visualizations
  - Flow Maps
  - Time Series Analysis
  - Regional Comparisons
  - Demographic Breakdowns
  - Flow Direction Analysis
- **Pagination**: Efficiently display large datasets with pagination

## Technologies

- Python
- Flask
- Pandas
- Plotly
- Bootstrap
- jQuery

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Access the web interface at `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `scrapers/`: Data collection modules
- `utils/`: Utility modules for data processing, analysis, and visualization
- `templates/`: HTML templates
- `static/`: Static files (CSS, JS, visualizations)
- `data/`: Data storage directory

## License

MIT
