import streamlit as st
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# Import our custom modules
from scrapers.guangdong_population_scraper import GuangdongPopulationScraper
from utils.data_processor import DataProcessor
from utils.data_analyzer import DataAnalyzer
from utils.visualizer import Visualizer

# Set page config
st.set_page_config(
    page_title="Guangdong Population Flow Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create instances of our classes
@st.cache_resource
def load_resources():
    scraper = GuangdongPopulationScraper()
    processor = DataProcessor()
    analyzer = DataAnalyzer()
    visualizer = Visualizer()
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/visualizations', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    return scraper, processor, analyzer, visualizer

scraper, processor, analyzer, visualizer = load_resources()

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
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # Default to English

# Analysis types
analysis_types = [
    "Overall Population Flow",  # 总体人口流动
    "Inter-city Migration",     # 城市间迁移
    "Urban-Rural Movement",     # 城乡流动
    "Demographic Analysis"      # 人口结构分析
]

# Function to translate text based on selected language
def translate(en_text, zh_text):
    if st.session_state.language == 'zh':
        return zh_text
    return en_text

# Sidebar for language selection and navigation
with st.sidebar:
    st.title(translate("Guangdong Population Flow Analysis", "广东人口流动分析"))
    
    # Language selector
    language = st.radio(
        translate("Language", "语言"),
        ["English", "中文"],
        index=0 if st.session_state.language == 'en' else 1,
        horizontal=True
    )
    st.session_state.language = 'en' if language == "English" else 'zh'
    
    # Navigation
    st.subheader(translate("Navigation", "导航"))
    page = st.radio(
        translate("Select a page", "选择页面"),
        [
            translate("Data Collection & Processing", "数据采集与处理"),
            translate("Analysis Results", "分析结果"),
            translate("Visualization", "可视化")
        ]
    )

# Main content area
if page == translate("Data Collection & Processing", "数据采集与处理"):
    st.header(translate("Data Collection & Processing", "数据采集与处理"))
    
    # Create two columns for controls and data display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Data Collection section
        st.subheader(translate("Data Collection", "数据采集"))
        
        # Data sources selection
        st.write(translate("Select Data Sources:", "选择数据源："))
        source_gsb = st.checkbox(translate("Guangdong Statistical Bureau", "广东省统计局"), value=True)
        source_cnbs = st.checkbox(translate("China National Bureau of Statistics", "中国国家统计局"), value=True)
        source_gmr = st.checkbox(translate("Guangdong Migration Reports", "广东省人口迁移报告"), value=True)
        source_arp = st.checkbox(translate("Academic Research Papers", "学术研究论文"), value=True)
        
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
        if st.button(translate("Scrape New Data", "采集新数据")):
            with st.spinner(translate("Scraping data...", "正在采集数据...")):
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
                    
                    st.success(translate(
                        f"Successfully scraped and processed {len(raw_data)} data points from {len(selected_sources)} sources.",
                        f"成功从 {len(selected_sources)} 个数据源采集并处理了 {len(raw_data)} 个数据点。"
                    ))
                except Exception as e:
                    st.error(translate(f"Error during scraping: {str(e)}", f"采集过程中出错：{str(e)}"))
        
        # Data Processing section
        st.subheader(translate("Data Processing", "数据处理"))
        
        # Region selection
        if st.session_state.available_regions:
            selected_regions = st.multiselect(
                translate("Filter by Regions:", "按地区筛选："),
                st.session_state.available_regions
            )
        else:
            selected_regions = []
            st.write(translate("No regions available yet.", "暂无可用地区。"))
        
        # Year range selection
        col1a, col1b = st.columns(2)
        with col1a:
            start_year = st.number_input(translate("Start Year", "开始年份"), min_value=2000, max_value=2030, value=2015)
        with col1b:
            end_year = st.number_input(translate("End Year", "结束年份"), min_value=2000, max_value=2030, value=2023)
        
        # Process button
        if st.button(translate("Process Data", "处理数据")):
            with st.spinner(translate("Processing data...", "正在处理数据...")):
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
                    
                    st.success(translate(
                        f"Successfully processed {len(st.session_state.current_data)} data points.",
                        f"成功处理了 {len(st.session_state.current_data)} 个数据点。"
                    ))
                except Exception as e:
                    st.error(translate(f"Error during data processing: {str(e)}", f"数据处理过程中出错：{str(e)}"))
    
    with col2:
        # Data Preview section
        st.subheader(translate("Data Preview", "数据预览"))
        
        # Load data if available
        if st.session_state.current_data is None and st.session_state.last_processed_data_path:
            try:
                st.session_state.current_data = pd.read_csv(st.session_state.last_processed_data_path)
                st.info(translate("Loaded previously processed data.", "已加载之前处理的数据。"))
            except Exception as e:
                st.error(translate(f"Error loading processed data: {str(e)}", f"加载处理数据时出错：{str(e)}"))
        
        # Display data
        if st.session_state.current_data is not None:
            # Replace NaN with None for better display
            display_data = st.session_state.current_data.replace({np.nan: None})
            
            # Add pagination
            page_size = st.selectbox(
                translate("Rows per page", "每页行数"),
                options=[10, 25, 50, 100],
                index=0
            )
            
            total_rows = len(display_data)
            total_pages = (total_rows + page_size - 1) // page_size
            
            if total_pages > 1:
                page_number = st.slider(
                    translate("Page", "页码"),
                    min_value=1,
                    max_value=total_pages,
                    value=1
                )
                start_idx = (page_number - 1) * page_size
                end_idx = min(start_idx + page_size, total_rows)
                
                st.write(translate(
                    f"Showing {start_idx + 1} to {end_idx} of {total_rows} entries",
                    f"显示第 {start_idx + 1} 至 {end_idx} 条，共 {total_rows} 条记录"
                ))
                
                # Display paginated data
                st.dataframe(display_data.iloc[start_idx:end_idx], use_container_width=True)
            else:
                # Display all data if it fits on one page
                st.dataframe(display_data, use_container_width=True)
        else:
            st.info(translate("No data available. Please scrape or process data first.", "暂无数据。请先采集或处理数据。"))

elif page == translate("Analysis Results", "分析结果"):
    st.header(translate("Analysis Results", "分析结果"))
    
    # Create two columns for controls and results display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Analysis Parameters section
        st.subheader(translate("Analysis Parameters", "分析参数"))
        
        # Analysis type selection
        analysis_type = st.selectbox(
            translate("Select Analysis Type:", "选择分析类型："),
            analysis_types
        )
        
        # Region selection
        if st.session_state.available_regions:
            selected_regions = st.multiselect(
                translate("Filter by Regions:", "按地区筛选："),
                st.session_state.available_regions
            )
        else:
            selected_regions = []
            st.write(translate("No regions available yet.", "暂无可用地区。"))
        
        # Year range selection
        col1a, col1b = st.columns(2)
        with col1a:
            start_year = st.number_input(translate("Start Year", "开始年份"), min_value=2000, max_value=2030, value=2015, key="analysis_start_year")
        with col1b:
            end_year = st.number_input(translate("End Year", "结束年份"), min_value=2000, max_value=2030, value=2023, key="analysis_end_year")
        
        # Run analysis button
        if st.button(translate("Run Analysis", "运行分析")):
            # Check if we have data to analyze
            if st.session_state.current_data is None and st.session_state.last_processed_data_path:
                # Load the last processed data if available
                try:
                    st.session_state.current_data = pd.read_csv(st.session_state.last_processed_data_path)
                except Exception as e:
                    st.error(translate(f"Error loading processed data: {str(e)}", f"加载处理数据时出错：{str(e)}"))
            
            if st.session_state.current_data is None:
                st.error(translate("No processed data available. Please process data first.", "暂无处理数据。请先处理数据。"))
            else:
                with st.spinner(translate("Running analysis...", "正在运行分析...")):
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
                        
                        st.success(translate(f"Successfully analyzed data with {analysis_type}.", f"成功使用 {analysis_type} 分析了数据。"))
                    except Exception as e:
                        st.error(translate(f"Error during analysis: {str(e)}", f"分析过程中出错：{str(e)}"))
    
    with col2:
        # Analysis Results section
        if st.session_state.current_analysis:
            # Overview section
            st.subheader(translate("Overview", "概览"))
            if 'overview' in st.session_state.current_analysis:
                overview = st.session_state.current_analysis['overview']
                
                # Create metrics for key statistics
                metric_cols = st.columns(3)
                
                # Regions count
                if 'regions_count' in overview:
                    with metric_cols[0]:
                        st.metric(translate("Regions", "地区数量"), overview['regions_count'])
                
                # Years span
                if 'years_span' in overview:
                    with metric_cols[1]:
                        st.metric(translate("Years Covered", "覆盖年份"), f"{overview['years_span'][0]} - {overview['years_span'][1]}")
                
                # Total data points
                if 'metrics_distribution' in overview:
                    total_points = sum(overview['metrics_distribution'].values())
                    with metric_cols[2]:
                        st.metric(translate("Data Points", "数据点数量"), total_points)
                
                # Metrics distribution
                if 'metrics_distribution' in overview:
                    st.write(translate("Metrics Distribution", "指标分布"))
                    
                    # Convert to DataFrame for better display
                    metrics_df = pd.DataFrame({
                        translate("Metric", "指标"): list(overview['metrics_distribution'].keys()),
                        translate("Count", "数量"): list(overview['metrics_distribution'].values())
                    })
                    
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Regions list
                if 'regions' in overview and overview['regions']:
                    st.write(translate("Regions", "地区"))
                    st.write(", ".join(overview['regions']))
            
            # Insights section
            st.subheader(translate("Key Insights", "关键洞察"))
            if 'insights' in st.session_state.current_analysis and st.session_state.current_analysis['insights']:
                for insight in st.session_state.current_analysis['insights']:
                    st.write(f"• {insight}")
            else:
                st.info(translate("No insights available.", "暂无可用洞察。"))
            
            # Detailed Results section
            st.subheader(translate("Detailed Results", "详细结果"))
            
            # Top regions
            if 'top_regions' in st.session_state.current_analysis:
                top_regions = st.session_state.current_analysis['top_regions']
                
                if 'inflow' in top_regions and top_regions['inflow']:
                    st.write(translate("Top Regions by Immigration", "按移入人口排名的前几个地区"))
                    
                    # Convert to DataFrame
                    inflow_df = pd.DataFrame(top_regions['inflow'])
                    st.dataframe(inflow_df, use_container_width=True)
                
                if 'outflow' in top_regions and top_regions['outflow']:
                    st.write(translate("Top Regions by Emigration", "按移出人口排名的前几个地区"))
                    
                    # Convert to DataFrame
                    outflow_df = pd.DataFrame(top_regions['outflow'])
                    st.dataframe(outflow_df, use_container_width=True)
            
            # Net flows
            if 'net_flows' in st.session_state.current_analysis and st.session_state.current_analysis['net_flows']:
                st.write(translate("Net Migration Flows by Region", "各地区净迁移流量"))
                
                # Convert to DataFrame
                net_flows_df = pd.DataFrame(st.session_state.current_analysis['net_flows'])
                st.dataframe(net_flows_df, use_container_width=True)
            
            # Statistical data
            if 'statistical_data' in st.session_state.current_analysis and st.session_state.current_analysis['statistical_data']:
                st.write(translate("Statistical Data", "统计数据"))
                
                # Convert to DataFrame if it's not already
                if isinstance(st.session_state.current_analysis['statistical_data'], list):
                    stat_df = pd.DataFrame(st.session_state.current_analysis['statistical_data'])
                else:
                    stat_df = st.session_state.current_analysis['statistical_data']
                
                st.dataframe(stat_df, use_container_width=True)
            
            # Flow matrix
            if 'flow_matrix' in st.session_state.current_analysis:
                st.write(translate("Flow Matrix", "流量矩阵"))
                
                # Check if it's a dictionary or DataFrame
                if isinstance(st.session_state.current_analysis['flow_matrix'], dict):
                    # Convert dictionary to DataFrame
                    flow_matrix_df = pd.DataFrame(st.session_state.current_analysis['flow_matrix'])
                    st.dataframe(flow_matrix_df, use_container_width=True)
                elif isinstance(st.session_state.current_analysis['flow_matrix'], pd.DataFrame):
                    st.dataframe(st.session_state.current_analysis['flow_matrix'], use_container_width=True)
                else:
                    st.info(translate("Flow matrix data is not in a displayable format.", "流量矩阵数据不是可显示的格式。"))
            
            # Correlation matrix
            if 'correlation_matrix' in st.session_state.current_analysis:
                st.write(translate("Correlation Matrix", "相关性矩阵"))
                
                # Check if it's a dictionary or DataFrame
                if isinstance(st.session_state.current_analysis['correlation_matrix'], dict):
                    # Convert dictionary to DataFrame
                    corr_matrix_df = pd.DataFrame(st.session_state.current_analysis['correlation_matrix'])
                    st.dataframe(corr_matrix_df, use_container_width=True)
                elif isinstance(st.session_state.current_analysis['correlation_matrix'], pd.DataFrame):
                    st.dataframe(st.session_state.current_analysis['correlation_matrix'], use_container_width=True)
                else:
                    st.info(translate("Correlation matrix data is not in a displayable format.", "相关性矩阵数据不是可显示的格式。"))
        else:
            st.info(translate("No analysis results available. Please run an analysis first.", "暂无分析结果。请先运行分析。"))

elif page == translate("Visualization", "可视化"):
    st.header(translate("Visualization", "可视化"))
    
    # Create two columns for controls and visualization display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Visualization Settings section
        st.subheader(translate("Visualization Settings", "可视化设置"))
        
        # Visualization type selection
        viz_type = st.selectbox(
            translate("Visualization Type:", "可视化类型："),
            [
                "flow_map",
                "time_series",
                "regional_comparison",
                "demographic_breakdown",
                "flow_direction"
            ],
            format_func=lambda x: {
                "flow_map": translate("Population Flow Map", "人口流动地图"),
                "time_series": translate("Time Series Analysis", "时间序列分析"),
                "regional_comparison": translate("Regional Comparison", "地区对比"),
                "demographic_breakdown": translate("Demographic Breakdown", "人口结构分析"),
                "flow_direction": translate("Flow Direction Analysis", "流动方向分析")
            }.get(x, x)
        )
        
        # Metric selection
        if st.session_state.available_metrics:
            selected_metric = st.selectbox(
                translate("Metric:", "指标："),
                st.session_state.available_metrics
            )
        else:
            selected_metric = "Net Migration"  # Default
            st.write(translate("No metrics available yet.", "暂无可用指标。"))
        
        # Region selection
        if st.session_state.available_regions:
            selected_regions = st.multiselect(
                translate("Filter by Regions:", "按地区筛选："),
                st.session_state.available_regions,
                key="viz_regions"
            )
        else:
            selected_regions = []
            st.write(translate("No regions available yet.", "暂无可用地区。"))
        
        # Year selection
        selected_year = st.number_input(translate("Year:", "年份："), min_value=2000, max_value=2030, value=2023)
        
        # Generate visualization button
        if st.button(translate("Generate Visualization", "生成可视化")):
            # Check if we have data to visualize
            if st.session_state.current_data is None and st.session_state.last_processed_data_path:
                # Load the last processed data if available
                try:
                    st.session_state.current_data = pd.read_csv(st.session_state.last_processed_data_path)
                except Exception as e:
                    st.error(translate(f"Error loading processed data: {str(e)}", f"加载处理数据时出错：{str(e)}"))
            
            if st.session_state.current_data is None:
                st.error(translate("No processed data available. Please process data first.", "暂无处理数据。请先处理数据。"))
            else:
                with st.spinner(translate("Generating visualization...", "正在生成可视化...")):
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
                        
                        # Store the figure in session state for display
                        st.session_state.current_figure = fig
                        
                        st.success(translate(f"Successfully created {viz_type} visualization.", f"成功创建了 {viz_type} 可视化。"))
                    except Exception as e:
                        st.error(translate(f"Error creating visualization: {str(e)}", f"创建可视化时出错：{str(e)}"))
    
    with col2:
        # Visualization Display section
        st.subheader(translate("Visualization", "可视化"))
        
        if 'current_figure' in st.session_state and st.session_state.current_figure is not None:
            # Display the visualization
            st.plotly_chart(st.session_state.current_figure, use_container_width=True)
        else:
            # Create a placeholder visualization
            fig = go.Figure()
            fig.add_annotation(
                text=translate("No visualization generated yet. Please use the visualization controls to create one.", 
                              "尚未生成可视化。请使用可视化控件创建一个。"),
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=translate("Guangdong Population Flow Analysis", "广东人口流动分析"),
                template="plotly_dark",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font=dict(color='#e0e0e0'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    translate(
        "Guangdong Population Flow Analysis - Streamlit Version",
        "广东人口流动分析 - Streamlit 版本"
    )
)