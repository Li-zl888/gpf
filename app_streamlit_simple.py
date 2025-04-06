import streamlit as st
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Import our custom modules
from scrapers.guangdong_population_scraper import GuangdongPopulationScraper
from utils.data_processor import DataProcessor
from utils.data_analyzer import DataAnalyzer
from utils.visualizer import Visualizer

# Set page config
st.set_page_config(
    page_title="Guangdong Population Flow Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create instances of our classes
scraper = GuangdongPopulationScraper()
processor = DataProcessor()
analyzer = DataAnalyzer()
visualizer = Visualizer()

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
os.makedirs('data/visualizations', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Session state initialization
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'last_processed_data_path' not in st.session_state:
    st.session_state.last_processed_data_path = None
if 'available_regions' not in st.session_state:
    st.session_state.available_regions = []
if 'available_metrics' not in st.session_state:
    st.session_state.available_metrics = []

# Analysis types
analysis_types = [
    "Overall Population Flow",
    "Inter-city Migration",
    "Urban-Rural Movement",
    "Demographic Analysis"
]

# Main title
st.title("Guangdong Population Flow Analysis")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select a page",
        ["Data Collection & Processing", "Analysis Results", "Visualization"]
    )

# Main content area
if page == "Data Collection & Processing":
    st.header("Data Collection & Processing")
    
    # Create two columns for controls and data display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Data Collection section
        st.subheader("Data Collection")
        
        # Data sources selection
        st.write("Select Data Sources:")
        source_gsb = st.checkbox("Guangdong Statistical Bureau", value=True)
        source_cnbs = st.checkbox("China National Bureau of Statistics", value=True)
        source_gmr = st.checkbox("Guangdong Migration Reports", value=True)
        source_arp = st.checkbox("Academic Research Papers", value=True)
        
        # Collect selected sources
        selected_sources = []
        if source_gsb:
            selected_sources.append("Guangdong Statistical Bureau")
        if source_cnbs:
            selected_sources.append("China National Bureau of Statistics")
        if source_gmr:
            selected_sources.append("Guangdong Migration Reports")
        if source_arp:
            selected_sources.append("Academic Research Papers")
        
        # Scrape button
        if st.button("Scrape New Data"):
            with st.spinner("Scraping data..."):
                try:
                    # Scrape the data
                    raw_data = scraper.scrape_multiple_sources(selected_sources)
                    
                    # Automatically process the data after scraping
                    st.session_state.current_data = processor.process_data(raw_data)
                    
                    # Update available regions and metrics for UI
                    if 'region' in st.session_state.current_data.columns:
                        st.session_state.available_regions = sorted(st.session_state.current_data['region'].unique().tolist())
                    
                    if 'metric' in st.session_state.current_data.columns:
                        st.session_state.available_metrics = sorted(st.session_state.current_data['metric'].unique().tolist())
                    
                    # Get the path to the most recently processed data
                    processed_files = [f for f in os.listdir('data') if f.startswith('processed_data_')]
                    if processed_files:
                        st.session_state.last_processed_data_path = os.path.join('data', sorted(processed_files)[-1])
                    
                    st.success(f"Successfully scraped and processed {len(raw_data)} data points from {len(selected_sources)} sources.")
                except Exception as e:
                    st.error(f"Error during scraping: {str(e)}")
        
        # Data Processing section
        st.subheader("Data Processing")
        
        # Region selection
        if st.session_state.available_regions:
            selected_regions = st.multiselect(
                "Filter by Regions:",
                st.session_state.available_regions
            )
        else:
            selected_regions = []
            st.write("No regions available yet.")
        
        # Year range selection
        col1a, col1b = st.columns(2)
        with col1a:
            start_year = st.number_input("Start Year", min_value=2000, max_value=2030, value=2015)
        with col1b:
            end_year = st.number_input("End Year", min_value=2000, max_value=2030, value=2023)
        
        # Process button
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                try:
                    # Find the most recent raw data file
                    data_dir = 'data'
                    raw_data_files = [f for f in os.listdir(data_dir) if f.startswith('combined_raw_data_')]
                    
                    if not raw_data_files:
                        # No data files found, use sample data
                        raw_data = []
                    else:
                        # Sort files by name (which includes timestamp) and get the most recent
                        most_recent_file = sorted(raw_data_files)[-1]
                        file_path = os.path.join(data_dir, most_recent_file)
                        
                        with open(file_path, 'r', encoding='utf-8') as f:
                            raw_data = json.load(f)
                    
                    # Process the data
                    years = range(start_year, end_year + 1) if start_year <= end_year else None
                    st.session_state.current_data = processor.process_data(raw_data, regions=selected_regions if selected_regions else None, years=years)
                    
                    # Update available regions and metrics for UI
                    if 'region' in st.session_state.current_data.columns:
                        st.session_state.available_regions = sorted(st.session_state.current_data['region'].unique().tolist())
                    
                    if 'metric' in st.session_state.current_data.columns:
                        st.session_state.available_metrics = sorted(st.session_state.current_data['metric'].unique().tolist())
                    
                    # Get the path to the most recently processed data
                    processed_files = [f for f in os.listdir(data_dir) if f.startswith('processed_data_')]
                    if processed_files:
                        st.session_state.last_processed_data_path = os.path.join(data_dir, sorted(processed_files)[-1])
                    
                    st.success(f"Successfully processed {len(st.session_state.current_data)} data points.")
                except Exception as e:
                    st.error(f"Error during data processing: {str(e)}")
    
    with col2:
        # Data Preview section
        st.subheader("Data Preview")
        
        # Load data if available
        if st.session_state.current_data is None and st.session_state.last_processed_data_path:
            try:
                st.session_state.current_data = pd.read_csv(st.session_state.last_processed_data_path)
                st.info("Loaded previously processed data.")
            except Exception as e:
                st.error(f"Error loading processed data: {str(e)}")
        
        # Display data
        if st.session_state.current_data is not None:
            # Replace NaN with None for better display
            display_data = st.session_state.current_data.replace({np.nan: None})
            
            # Add pagination
            page_size = st.selectbox(
                "Rows per page",
                options=[10, 25, 50, 100],
                index=0
            )
            
            total_rows = len(display_data)
            total_pages = (total_rows + page_size - 1) // page_size
            
            if total_pages > 1:
                page_number = st.slider(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1
                )
                start_idx = (page_number - 1) * page_size
                end_idx = min(start_idx + page_size, total_rows)
                
                st.write(f"Showing {start_idx + 1} to {end_idx} of {total_rows} entries")
                
                # Display paginated data
                st.dataframe(display_data.iloc[start_idx:end_idx])
            else:
                # Display all data if it fits on one page
                st.dataframe(display_data)
        else:
            st.info("No data available. Please scrape or process data first.")

elif page == "Analysis Results":
    st.header("Analysis Results")
    
    # Create two columns for controls and results display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Analysis Parameters section
        st.subheader("Analysis Parameters")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            analysis_types
        )
        
        # Region selection
        if st.session_state.available_regions:
            selected_regions = st.multiselect(
                "Filter by Regions:",
                st.session_state.available_regions
            )
        else:
            selected_regions = []
            st.write("No regions available yet.")
        
        # Year range selection
        col1a, col1b = st.columns(2)
        with col1a:
            start_year = st.number_input("Start Year", min_value=2000, max_value=2030, value=2015, key="analysis_start_year")
        with col1b:
            end_year = st.number_input("End Year", min_value=2000, max_value=2030, value=2023, key="analysis_end_year")
        
        # Run analysis button
        if st.button("Run Analysis"):
            # Check if we have data to analyze
            if st.session_state.current_data is None and st.session_state.last_processed_data_path:
                # Load the last processed data if available
                try:
                    st.session_state.current_data = pd.read_csv(st.session_state.last_processed_data_path)
                except Exception as e:
                    st.error(f"Error loading processed data: {str(e)}")
            
            if st.session_state.current_data is None:
                st.error("No processed data available. Please process data first.")
            else:
                with st.spinner("Running analysis..."):
                    try:
                        # Prepare parameters
                        years = range(start_year, end_year + 1) if start_year <= end_year else None
                        
                        # Perform the analysis
                        st.session_state.current_analysis = analyzer.analyze(
                            st.session_state.current_data,
                            analysis_type=analysis_type,
                            regions=selected_regions if selected_regions else None,
                            years=years
                        )
                        
                        st.success(f"Successfully analyzed data with {analysis_type}.")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    
    with col2:
        # Analysis Results section
        if st.session_state.current_analysis:
            # Overview section
            st.subheader("Overview")
            if 'overview' in st.session_state.current_analysis:
                overview = st.session_state.current_analysis['overview']
                
                # Create metrics for key statistics
                metric_cols = st.columns(3)
                
                # Regions count
                if 'regions_count' in overview:
                    with metric_cols[0]:
                        st.metric("Regions", overview['regions_count'])
                
                # Years span
                if 'years_span' in overview:
                    with metric_cols[1]:
                        st.metric("Years Covered", f"{overview['years_span'][0]} - {overview['years_span'][1]}")
                
                # Total data points
                if 'metrics_distribution' in overview:
                    total_points = sum(overview['metrics_distribution'].values())
                    with metric_cols[2]:
                        st.metric("Data Points", total_points)
                
                # Metrics distribution
                if 'metrics_distribution' in overview:
                    st.write("Metrics Distribution")
                    
                    # Convert to DataFrame for better display
                    metrics_df = pd.DataFrame({
                        "Metric": list(overview['metrics_distribution'].keys()),
                        "Count": list(overview['metrics_distribution'].values())
                    })
                    
                    st.dataframe(metrics_df)
                
                # Regions list
                if 'regions' in overview and overview['regions']:
                    st.write("Regions")
                    st.write(", ".join(overview['regions']))
            
            # Insights section
            st.subheader("Key Insights")
            if 'insights' in st.session_state.current_analysis and st.session_state.current_analysis['insights']:
                for insight in st.session_state.current_analysis['insights']:
                    st.write(f"• {insight}")
            else:
                st.info("No insights available.")
            
            # Detailed Results section
            st.subheader("Detailed Results")
            
            # Top regions
            if 'top_regions' in st.session_state.current_analysis:
                top_regions = st.session_state.current_analysis['top_regions']
                
                if 'inflow' in top_regions and top_regions['inflow']:
                    st.write("Top Regions by Immigration")
                    
                    # Convert to DataFrame
                    inflow_df = pd.DataFrame(top_regions['inflow'])
                    st.dataframe(inflow_df)
                
                if 'outflow' in top_regions and top_regions['outflow']:
                    st.write("Top Regions by Emigration")
                    
                    # Convert to DataFrame
                    outflow_df = pd.DataFrame(top_regions['outflow'])
                    st.dataframe(outflow_df)
            
            # Net flows
            if 'net_flows' in st.session_state.current_analysis and st.session_state.current_analysis['net_flows']:
                st.write("Net Migration Flows by Region")
                
                # Convert to DataFrame
                net_flows_df = pd.DataFrame(st.session_state.current_analysis['net_flows'])
                st.dataframe(net_flows_df)
            
            # Statistical data
            if 'statistical_data' in st.session_state.current_analysis and st.session_state.current_analysis['statistical_data']:
                st.write("Statistical Data")
                
                # Convert to DataFrame if it's not already
                if isinstance(st.session_state.current_analysis['statistical_data'], list):
                    stat_df = pd.DataFrame(st.session_state.current_analysis['statistical_data'])
                else:
                    stat_df = st.session_state.current_analysis['statistical_data']
                
                st.dataframe(stat_df)
            
            # Flow matrix
            if 'flow_matrix' in st.session_state.current_analysis:
                st.write("Flow Matrix")
                
                # Check if it's a dictionary or DataFrame
                if isinstance(st.session_state.current_analysis['flow_matrix'], dict):
                    # Convert dictionary to DataFrame
                    flow_matrix_df = pd.DataFrame(st.session_state.current_analysis['flow_matrix'])
                    st.dataframe(flow_matrix_df)
                elif isinstance(st.session_state.current_analysis['flow_matrix'], pd.DataFrame):
                    st.dataframe(st.session_state.current_analysis['flow_matrix'])
                else:
                    st.info("Flow matrix data is not in a displayable format.")
            
            # Correlation matrix
            if 'correlation_matrix' in st.session_state.current_analysis:
                st.write("Correlation Matrix")
                
                # Check if it's a dictionary or DataFrame
                if isinstance(st.session_state.current_analysis['correlation_matrix'], dict):
                    # Convert dictionary to DataFrame
                    corr_matrix_df = pd.DataFrame(st.session_state.current_analysis['correlation_matrix'])
                    st.dataframe(corr_matrix_df)
                elif isinstance(st.session_state.current_analysis['correlation_matrix'], pd.DataFrame):
                    st.dataframe(st.session_state.current_analysis['correlation_matrix'])
                else:
                    st.info("Correlation matrix data is not in a displayable format.")
        else:
            st.info("No analysis results available. Please run an analysis first.")

elif page == "Visualization":
    st.header("Visualization")
    
    # Create two columns for controls and visualization display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Visualization Settings section
        st.subheader("Visualization Settings")
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Visualization Type:",
            [
                "flow_map",
                "time_series",
                "regional_comparison",
                "demographic_breakdown",
                "flow_direction"
            ],
            format_func=lambda x: {
                "flow_map": "Population Flow Map",
                "time_series": "Time Series Analysis",
                "regional_comparison": "Regional Comparison",
                "demographic_breakdown": "Demographic Breakdown",
                "flow_direction": "Flow Direction Analysis"
            }.get(x, x)
        )
        
        # Metric selection
        if st.session_state.available_metrics:
            selected_metric = st.selectbox(
                "Metric:",
                st.session_state.available_metrics
            )
        else:
            selected_metric = "Net Migration"  # Default
            st.write("No metrics available yet.")
        
        # Region selection
        if st.session_state.available_regions:
            selected_regions = st.multiselect(
                "Filter by Regions:",
                st.session_state.available_regions,
                key="viz_regions"
            )
        else:
            selected_regions = []
            st.write("No regions available yet.")
        
        # Year selection
        selected_year = st.number_input("Year:", min_value=2000, max_value=2030, value=2023)
        
        # Generate visualization button
        if st.button("Generate Visualization"):
            # Check if we have data to visualize
            if st.session_state.current_data is None and st.session_state.last_processed_data_path:
                # Load the last processed data if available
                try:
                    st.session_state.current_data = pd.read_csv(st.session_state.last_processed_data_path)
                except Exception as e:
                    st.error(f"Error loading processed data: {str(e)}")
            
            if st.session_state.current_data is None:
                st.error("No processed data available. Please process data first.")
            else:
                with st.spinner("Generating visualization..."):
                    try:
                        # Check if we need analysis results
                        if viz_type in ['flow_map', 'demographic_breakdown', 'flow_direction'] and st.session_state.current_analysis is None:
                            # Perform a default analysis if none exists
                            if viz_type == 'flow_map':
                                analysis_type = "Overall Population Flow"
                            elif viz_type == 'demographic_breakdown':
                                analysis_type = "Demographic Analysis"
                            elif viz_type == 'flow_direction':
                                analysis_type = "Inter-city Migration"
                            
                            st.session_state.current_analysis = analyzer.analyze(
                                st.session_state.current_data,
                                analysis_type=analysis_type,
                                regions=selected_regions if selected_regions else None,
                                years=[selected_year] if selected_year else None
                            )
                        
                        # Create the visualization based on type
                        if viz_type == 'flow_map':
                            fig = visualizer.create_flow_map(
                                st.session_state.current_data,
                                st.session_state.current_analysis,
                                regions=selected_regions if selected_regions else None,
                                year=selected_year
                            )
                        
                        elif viz_type == 'time_series':
                            fig = visualizer.create_time_series(
                                st.session_state.current_data,
                                metric=selected_metric,
                                regions=selected_regions if selected_regions else None
                            )
                        
                        elif viz_type == 'regional_comparison':
                            fig = visualizer.create_regional_comparison(
                                st.session_state.current_data,
                                comparison_type=selected_metric,
                                regions=selected_regions if selected_regions else None,
                                year=selected_year
                            )
                        
                        elif viz_type == 'demographic_breakdown':
                            fig = visualizer.create_demographic_breakdown(
                                st.session_state.current_data,
                                st.session_state.current_analysis,
                                demographic_factor=selected_metric,
                                regions=selected_regions if selected_regions else None
                            )
                        
                        elif viz_type == 'flow_direction':
                            fig = visualizer.create_flow_direction_analysis(
                                st.session_state.current_data,
                                st.session_state.current_analysis,
                                metric=selected_metric,
                                regions=selected_regions if selected_regions else None
                            )
                        
                        # Save the visualization to a file
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        filename = f"data/visualizations/{viz_type}_{timestamp}.html"
                        fig.write_html(filename)
                        
                        # Also save to static folder for web access
                        static_filename = f"static/{viz_type}_latest.html"
                        fig.write_html(static_filename)
                        
                        st.success(f"Successfully created {viz_type} visualization.")
                        
                        # Display the visualization
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
    
    with col2:
        # Visualization Display section
        st.subheader("Visualization")
        st.write("Generate a visualization using the controls on the left.")
        
        # Display a placeholder message
        st.info("Select visualization settings and click 'Generate Visualization' to create a visualization.")

# Footer
st.markdown("---")
st.markdown("Guangdong Population Flow Analysis - Streamlit Version")