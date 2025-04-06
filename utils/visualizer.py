import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os
from datetime import datetime

class Visualizer:
    """
    A class to create visualizations for population flow data in Guangdong Province.
    用于创建广东省人口流动数据可视化的类。
    """
    
    def __init__(self):
        # Create basic color scheme for consistent visualizations
        self.color_scheme = {
            'primary': '#636EFA',  # Plotly default blue
            'secondary': '#EF553B',  # Plotly default red
            'tertiary': '#00CC96',  # Plotly default green
            'quaternary': '#AB63FA',  # Plotly default purple
            'background': '#F8F9FA',
            'text': '#2A2A2A'
        }
        
        # Color maps for different regions
        self.region_colors = px.colors.qualitative.Plotly
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/visualizations", exist_ok=True)
    
    def create_flow_map(self, data, analysis_results, regions=None, year=None):
        """
        Create a map visualization of population flows between regions.
        创建地区间人口流动的地图可视化。
        
        Args:
            data (pd.DataFrame): Processed data
            analysis_results (dict): Results from data analysis
            regions (list): List of regions to include (if None, use all)
            year (int): Year to use for the visualization (if None, use the most recent)
            
        参数：
            data (pd.DataFrame): 处理后的数据
            analysis_results (dict): 数据分析结果
            regions (list): 要包含的地区列表（如果为None，则使用所有地区）
            year (int): 用于可视化的年份（如果为None，则使用最近的年份）
            
        Returns:
            plotly.graph_objects.Figure: Flow map visualization
            
        返回：
            plotly.graph_objects.Figure: 流动地图可视化
        """
        # Filter data if needed
        df = data.copy()
        
        if regions:
            # Filter for specific regions
            if 'region' in df.columns:
                region_mask = df['region'].isin(regions)
            else:
                region_mask = pd.Series(False, index=df.index)
            
            if 'from_region' in df.columns and 'to_region' in df.columns:
                flow_mask = df['from_region'].isin(regions) | df['to_region'].isin(regions)
                region_mask = region_mask | flow_mask
            
            df = df[region_mask].copy()
        
        if year:
            # Filter for specific year
            if 'year' in df.columns:
                df = df[df['year'] == year].copy()
        
        # Check if we have inter-city flow data
        has_flow_data = 'from_region' in df.columns and 'to_region' in df.columns
        
        # Create the map
        if has_flow_data and 'Inter-city Migration' in df['metric'].values:
            # Use flow data to create migration map
            flow_df = df[df['metric'] == 'Inter-city Migration'].copy()
            
            # Dictionary of approximate coordinates for major Guangdong cities
            # These are approximate coordinates for illustration
            city_coordinates = {
                'Guangzhou': [113.2644, 23.1291],
                'Shenzhen': [114.0579, 22.5431],
                'Dongguan': [113.7518, 23.0207],
                'Foshan': [113.1221, 23.0291],
                'Zhongshan': [113.3924, 22.5160],
                'Zhuhai': [113.5767, 22.2769],
                'Huizhou': [114.4126, 23.1118],
                'Jiangmen': [113.0823, 22.5900],
                'Zhaoqing': [112.4669, 23.0471],
                'Shantou': [116.6781, 23.3535],
                'Chaozhou': [116.6220, 23.6567],
                'Jieyang': [116.3653, 23.5440],
                'Meizhou': [116.1184, 24.2882],
                'Shanwei': [115.3751, 22.7787],
                'Heyuan': [114.6978, 23.7461],
                'Yangjiang': [111.9797, 21.8581],
                'Maoming': [110.9251, 21.6631],
                'Zhanjiang': [110.3594, 21.2707],
                'Qingyuan': [113.0303, 23.6843],
                'Yunfu': [112.0440, 22.9286]
            }
            
            # Node positions
            nodes = []
            for city, coords in city_coordinates.items():
                if city in flow_df['from_region'].values or city in flow_df['to_region'].values:
                    nodes.append({
                        'name': city,
                        'x': coords[0],
                        'y': coords[1]
                    })
            
            # Create node dataframe
            node_df = pd.DataFrame(nodes)
            
            # Create edge dataframe
            edges = []
            for _, row in flow_df.iterrows():
                from_city = row['from_region']
                to_city = row['to_region']
                value = row['value']
                
                if from_city in city_coordinates and to_city in city_coordinates:
                    edges.append({
                        'from': from_city,
                        'to': to_city,
                        'value': value
                    })
            
            edge_df = pd.DataFrame(edges)
            
            # Normalize edge values for better visualization
            if not edge_df.empty:
                edge_df['normalized_value'] = edge_df['value'] / edge_df['value'].max()
            
            # Create figure
            fig = go.Figure()
            
            # Add map background (simplified - in a real app, would use actual geographic data)
            fig.add_trace(go.Scattermapbox(
                mode="markers",
                lon=node_df['x'],
                lat=node_df['y'],
                marker=dict(
                    size=10,
                    color=self.color_scheme['primary'],
                    opacity=0.8
                ),
                text=node_df['name'],
                hoverinfo='text',
                name='Cities'
            ))
            
            # Add flow lines
            if not edge_df.empty:
                for _, edge in edge_df.iterrows():
                    from_city = edge['from']
                    to_city = edge['to']
                    
                    if from_city in city_coordinates and to_city in city_coordinates:
                        from_coords = city_coordinates[from_city]
                        to_coords = city_coordinates[to_city]
                        
                        fig.add_trace(go.Scattergeo(
                            locationmode='country names',
                            lon=[from_coords[0], to_coords[0]],
                            lat=[from_coords[1], to_coords[1]],
                            mode='lines',
                            line=dict(
                                width=edge['normalized_value'] * 10,  # Scale line width
                                color=self.color_scheme['secondary']
                            ),
                            opacity=0.8,
                            text=f"{from_city} to {to_city}: {edge['value']:.2f} million",
                            hoverinfo='text',
                            name=''
                        ))
            
            # Update layout for map
            fig.update_layout(
                title=f"Population Flow in Guangdong Province ({year})" if year else "Population Flow in Guangdong Province",
                autosize=True,
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=22.8, lon=113.5),  # Center on Guangdong
                    zoom=7
                ),
                showlegend=False,
                height=700
            )
        else:
            # Use region data to create choropleth-like map
            # For regions that have data
            region_df = df[df['metric'].isin(['Immigration', 'Net Migration', 'Total Population'])]
            
            if 'region' in region_df.columns:
                region_df = region_df.groupby('region').agg({
                    'value': 'mean',
                    'metric': 'first'
                }).reset_index()
            
            # Dictionary of approximate coordinates for major Guangdong cities
            city_coordinates = {
                'Guangzhou': [113.2644, 23.1291],
                'Shenzhen': [114.0579, 22.5431],
                'Dongguan': [113.7518, 23.0207],
                'Foshan': [113.1221, 23.0291],
                'Zhongshan': [113.3924, 22.5160],
                'Zhuhai': [113.5767, 22.2769],
                'Huizhou': [114.4126, 23.1118],
                'Jiangmen': [113.0823, 22.5900],
                'Zhaoqing': [112.4669, 23.0471],
                'Shantou': [116.6781, 23.3535],
                'Chaozhou': [116.6220, 23.6567],
                'Jieyang': [116.3653, 23.5440],
                'Meizhou': [116.1184, 24.2882],
                'Shanwei': [115.3751, 22.7787],
                'Heyuan': [114.6978, 23.7461],
                'Yangjiang': [111.9797, 21.8581],
                'Maoming': [110.9251, 21.6631],
                'Zhanjiang': [110.3594, 21.2707],
                'Qingyuan': [113.0303, 23.6843],
                'Yunfu': [112.0440, 22.9286]
            }
            
            # Create a simplified map visualization
            fig = go.Figure()
            
            # Add points for each region with size based on value
            for i, row in region_df.iterrows():
                region = row['region']
                value = row['value']
                
                if region in city_coordinates:
                    coords = city_coordinates[region]
                    
                    # Determine color based on metric
                    metric = row['metric']
                    color = self.color_scheme['primary']  # Default
                    
                    if metric == 'Immigration':
                        color = self.color_scheme['tertiary']  # Green for immigration
                    elif metric == 'Emigration':
                        color = self.color_scheme['secondary']  # Red for emigration
                    
                    # Size based on value (normalized)
                    size = 10 + 40 * (value / region_df['value'].max())
                    
                    fig.add_trace(go.Scattermapbox(
                        mode="markers",
                        lon=[coords[0]],
                        lat=[coords[1]],
                        marker=dict(
                            size=size,
                            color=color,
                            opacity=0.7
                        ),
                        text=f"{region}: {value:.2f} million",
                        hoverinfo='text',
                        name=region
                    ))
            
            # Update layout for map
            fig.update_layout(
                title=f"Population Distribution in Guangdong Province ({year})" if year else "Population Distribution in Guangdong Province",
                autosize=True,
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=22.8, lon=113.5),  # Center on Guangdong
                    zoom=7
                ),
                height=700
            )
        
        # Save the visualization
        if year:
            filename = f"data/visualizations/flow_map_{year}.html"
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"data/visualizations/flow_map_{timestamp}.html"
        
        fig.write_html(filename)
        print(f"Saved flow map visualization to {filename}")
        
        return fig
    
    def create_time_series(self, data, metric='Net Migration', regions=None, years=None):
        """
        Create a time series visualization for the specified metric.
        创建指定指标的时间序列可视化。
        
        Args:
            data (pd.DataFrame): Processed data
            metric (str): Metric to visualize
            regions (list): List of regions to include
            years (range): Range of years to include
            
        参数：
            data (pd.DataFrame): 处理后的数据
            metric (str): 要可视化的指标
            regions (list): 要包含的地区列表
            years (range): 要包含的年份范围
            
        Returns:
            plotly.graph_objects.Figure: Time series visualization
            
        返回：
            plotly.graph_objects.Figure: 时间序列可视化
        """
        # Filter data if needed
        df = data.copy()
        
        if regions:
            # Filter for specific regions
            if 'region' in df.columns:
                df = df[df['region'].isin(regions)].copy()
        
        if years:
            # Filter for specific years
            if 'year' in df.columns:
                df = df[df['year'].isin(years)].copy()
        
        # Filter for the specified metric
        if 'metric' in df.columns:
            df = df[df['metric'] == metric].copy()
        
        # Check if we have data
        if df.empty or 'year' not in df.columns or 'region' not in df.columns:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for the selected metric and filters",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=f"Time Series Analysis: {metric}",
                xaxis_title="Year",
                yaxis_title="Value (millions)",
                height=600
            )
            return fig
        
        # Aggregate data by region and year
        time_series = df.groupby(['region', 'year']).agg({
            'value': 'mean'
        }).reset_index()
        
        # Sort by year
        time_series = time_series.sort_values(['region', 'year'])
        
        # Create figure
        fig = go.Figure()
        
        # Add line for each region
        for i, region in enumerate(time_series['region'].unique()):
            region_data = time_series[time_series['region'] == region]
            
            color_index = i % len(self.region_colors)
            
            fig.add_trace(go.Scatter(
                x=region_data['year'],
                y=region_data['value'],
                mode='lines+markers',
                name=region,
                line=dict(color=self.region_colors[color_index], width=2),
                marker=dict(size=8)
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Time Series Analysis: {metric} ({', '.join(regions) if regions else 'All Regions'})",
            xaxis_title="Year",
            yaxis_title=f"{metric} (millions)",
            legend_title="Region",
            hovermode="x unified",
            height=600
        )
        
        # Improve x-axis formatting
        fig.update_xaxes(
            tickmode='linear',
            tick0=time_series['year'].min(),
            dtick=1
        )
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"data/visualizations/time_series_{metric.replace(' ', '_').lower()}_{timestamp}.html"
        fig.write_html(filename)
        print(f"Saved time series visualization to {filename}")
        
        return fig
    
    def create_regional_comparison(self, data, comparison_type='Total Population', regions=None, year=None):
        """
        Create a regional comparison visualization.
        创建地区对比可视化。
        
        Args:
            data (pd.DataFrame): Processed data
            comparison_type (str): Type of comparison to visualize
            regions (list): List of regions to include
            year (int): Year to use for the visualization
            
        参数：
            data (pd.DataFrame): 处理后的数据
            comparison_type (str): 要可视化的对比类型
            regions (list): 要包含的地区列表
            year (int): 用于可视化的年份
            
        Returns:
            plotly.graph_objects.Figure: Regional comparison visualization
            
        返回：
            plotly.graph_objects.Figure: 地区对比可视化
        """
        # Filter data if needed
        df = data.copy()
        
        if regions:
            # Filter for specific regions
            if 'region' in df.columns:
                df = df[df['region'].isin(regions)].copy()
        
        if year:
            # Filter for specific year
            if 'year' in df.columns:
                df = df[df['year'] == year].copy()
        
        # Map comparison type to metric
        metric_mapping = {
            'Total Population': 'Total Population',
            'Population Density': 'Total Population',  # Would need area data for true density
            'Migration Rate': 'Migration Intensity',
            'Urban-Rural Ratio': None  # Would need specific data
        }
        
        metric = metric_mapping.get(comparison_type)
        
        # Filter for the specified metric
        if metric and 'metric' in df.columns:
            df = df[df['metric'] == metric].copy()
        
        # Check if we have data
        if df.empty or 'region' not in df.columns:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {comparison_type} with the selected filters",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=f"Regional Comparison: {comparison_type}",
                height=600
            )
            return fig
        
        # Aggregate data by region
        if 'value' in df.columns:
            region_data = df.groupby('region')['value'].mean().reset_index()
            region_data = region_data.sort_values('value', ascending=False)
        
            # Create horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=region_data['region'],
                x=region_data['value'],
                orientation='h',
                marker_color=self.color_scheme['primary'],
                text=region_data['value'].round(2),
                textposition='auto'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Regional Comparison: {comparison_type} ({year if year else 'All Years'})",
                xaxis_title=f"{comparison_type} (millions)",
                yaxis_title="Region",
                height=max(400, len(region_data) * 30)  # Scale height based on number of regions
            )
            
            # Improve y-axis ordering
            fig.update_yaxes(
                categoryorder='total ascending'
            )
        else:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No value data available for {comparison_type}",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=f"Regional Comparison: {comparison_type}",
                height=600
            )
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"data/visualizations/regional_comparison_{comparison_type.replace(' ', '_').lower()}_{timestamp}.html"
        fig.write_html(filename)
        print(f"Saved regional comparison visualization to {filename}")
        
        return fig
    
    def create_demographic_breakdown(self, data, analysis_results, demographic_factor='Age Groups', regions=None):
        """
        Create a demographic breakdown visualization.
        创建人口统计分析可视化。
        
        Args:
            data (pd.DataFrame): Processed data
            analysis_results (dict): Results from data analysis
            demographic_factor (str): Demographic factor to visualize
            regions (list): List of regions to include
            
        参数：
            data (pd.DataFrame): 处理后的数据
            analysis_results (dict): 数据分析结果
            demographic_factor (str): 要可视化的人口统计因素
            regions (list): 要包含的地区列表
            
        Returns:
            plotly.graph_objects.Figure: Demographic breakdown visualization
            
        返回：
            plotly.graph_objects.Figure: 人口统计分析可视化
        """
        # Check analysis results for demographic data
        demo_data = None
        
        # Map user-friendly terms to analysis result keys
        demographic_mapping = {
            'Age Groups': 'age_distribution',
            'Education Level': 'education_distribution',
            'Income Level': 'income_distribution',
            'Occupation': 'occupation_distribution'
        }
        
        result_key = demographic_mapping.get(demographic_factor)
        
        if result_key and result_key in analysis_results:
            demo_data = analysis_results[result_key]
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(demo_data, list):
                demo_data = pd.DataFrame(demo_data)
        
        # If no data in analysis results, try to extract from raw data
        if demo_data is None or demo_data.empty:
            df = data.copy()
            
            # Filter data if needed
            if regions:
                # Filter for specific regions
                if 'region' in df.columns:
                    df = df[df['region'].isin(regions)].copy()
            
            # Look for demographic information
            if 'demographic' in df.columns and 'group' in df.columns:
                # Map demographic factor to values in the data
                factor_mapping = {
                    'Age Groups': 'Age',
                    'Education Level': 'Education',
                    'Income Level': 'Income',
                    'Occupation': 'Occupation'
                }
                
                factor = factor_mapping.get(demographic_factor)
                
                if factor:
                    demo_df = df[df['demographic'] == factor].copy()
                    
                    if not demo_df.empty:
                        demo_data = demo_df.groupby('group')['value'].mean().reset_index()
                        
                        # Try to sort appropriately
                        if factor == 'Age':
                            # Try to sort by age range start
                            try:
                                demo_data['sort_key'] = demo_data['group'].apply(
                                    lambda x: int(x.split('-')[0]) if '-' in x else int(x.replace('+', ''))
                                )
                                demo_data = demo_data.sort_values('sort_key')
                                demo_data.drop('sort_key', axis=1, inplace=True)
                            except:
                                # If parsing fails, sort by value
                                demo_data = demo_data.sort_values('value', ascending=False)
                        else:
                            # Sort by value for other factors
                            demo_data = demo_data.sort_values('value', ascending=False)
            
            # Alternative: Look for metrics that might contain demographic information
            if (demo_data is None or demo_data.empty) and 'metric' in df.columns:
                factor_keywords = {
                    'Age Groups': ['Age'],
                    'Education Level': ['Education'],
                    'Income Level': ['Income'],
                    'Occupation': ['Occupation']
                }
                
                keywords = factor_keywords.get(demographic_factor, [])
                
                demographic_metrics = [
                    m for m in df['metric'].unique() 
                    if any(keyword in m for keyword in keywords)
                ]
                
                if demographic_metrics:
                    demo_df = df[df['metric'].isin(demographic_metrics)].copy()
                    
                    if not demo_df.empty:
                        # Try to extract group from metric name
                        demo_df['group'] = demo_df['metric'].apply(
                            lambda x: x.split()[-1] if len(x.split()) > 1 else x
                        )
                        
                        demo_data = demo_df.groupby('group')['value'].mean().reset_index()
                        demo_data = demo_data.sort_values('value', ascending=False)
        
        # Check if we have data to visualize
        if demo_data is None or demo_data.empty or 'group' not in demo_data.columns or 'value' not in demo_data.columns:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {demographic_factor} breakdown",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=f"Demographic Breakdown: {demographic_factor}",
                height=600
            )
            return fig
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=demo_data['group'],
            y=demo_data['value'],
            marker_color=self.region_colors[:len(demo_data)],
            text=demo_data['value'].round(3),
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Demographic Breakdown: {demographic_factor}",
            xaxis_title=demographic_factor,
            yaxis_title="Value (millions)",
            height=600
        )
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"data/visualizations/demographic_{demographic_factor.replace(' ', '_').lower()}_{timestamp}.html"
        fig.write_html(filename)
        print(f"Saved demographic breakdown visualization to {filename}")
        
        return fig
    
    def create_flow_direction_analysis(self, data, analysis_results, metric='Net Flow Volume', regions=None):
        """
        Create a visualization of flow directions.
        创建流动方向可视化。
        
        Args:
            data (pd.DataFrame): Processed data
            analysis_results (dict): Results from data analysis
            metric (str): Flow metric to visualize
            regions (list): List of regions to include
            
        参数：
            data (pd.DataFrame): 处理后的数据
            analysis_results (dict): 数据分析结果
            metric (str): 要可视化的流动指标
            regions (list): 要包含的地区列表
            
        Returns:
            plotly.graph_objects.Figure: Flow direction visualization
            
        返回：
            plotly.graph_objects.Figure: 流动方向可视化
        """
        # Check analysis results for flow data
        flow_matrix = None
        net_flows = None
        
        if 'flow_matrix' in analysis_results:
            flow_matrix = analysis_results['flow_matrix']
            
            # Convert to DataFrame if it's a dictionary
            if isinstance(flow_matrix, dict):
                flow_matrix = pd.DataFrame(flow_matrix)
        
        if 'net_flows' in analysis_results:
            net_flows = analysis_results['net_flows']
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(net_flows, list):
                net_flows = pd.DataFrame(net_flows)
        
        # If no data in analysis results, try to extract from raw data
        if (flow_matrix is None or flow_matrix.empty) and (net_flows is None or net_flows.empty):
            df = data.copy()
            
            # Filter data if needed
            if regions:
                # Filter for specific regions
                if 'region' in df.columns:
                    region_mask = df['region'].isin(regions)
                else:
                    region_mask = pd.Series(False, index=df.index)
                
                if 'from_region' in df.columns and 'to_region' in df.columns:
                    flow_mask = df['from_region'].isin(regions) | df['to_region'].isin(regions)
                    region_mask = region_mask | flow_mask
                
                df = df[region_mask].copy()
            
            # Check if we have inter-city flow data
            if 'from_region' in df.columns and 'to_region' in df.columns:
                flow_df = df[df['metric'] == 'Inter-city Migration'].copy() if 'metric' in df.columns else df
                
                if not flow_df.empty:
                    # Create flow matrix
                    all_regions = sorted(list(set(flow_df['from_region'].unique()) | set(flow_df['to_region'].unique())))
                    flow_matrix = pd.DataFrame(0, index=all_regions, columns=all_regions)
                    
                    # Populate flow matrix
                    for _, row in flow_df.iterrows():
                        from_region = row['from_region']
                        to_region = row['to_region']
                        value = row['value']
                        
                        if from_region in flow_matrix.index and to_region in flow_matrix.columns:
                            flow_matrix.at[from_region, to_region] = value
                    
                    # Calculate net flows
                    net_flows = []
                    
                    for region in all_regions:
                        # Total outflow
                        outflow = flow_df[flow_df['from_region'] == region]['value'].sum() if region in flow_df['from_region'].values else 0
                        
                        # Total inflow
                        inflow = flow_df[flow_df['to_region'] == region]['value'].sum() if region in flow_df['to_region'].values else 0
                        
                        # Net flow
                        net_flow = inflow - outflow
                        
                        net_flows.append({
                            'region': region,
                            'inflow': inflow,
                            'outflow': outflow,
                            'net_flow': net_flow
                        })
                    
                    net_flows = pd.DataFrame(net_flows)
                    net_flows = net_flows.sort_values('net_flow', ascending=False)
            
            # Alternative: Try to derive flow information from immigration/emigration data
            elif (flow_matrix is None or flow_matrix.empty) and 'metric' in df.columns:
                immigration_df = df[df['metric'] == 'Immigration'].copy() if 'Immigration' in df['metric'].values else None
                emigration_df = df[df['metric'] == 'Emigration'].copy() if 'Emigration' in df['metric'].values else None
                
                if immigration_df is not None and emigration_df is not None and 'region' in df.columns:
                    # Calculate total immigration and emigration by region
                    immigration_by_region = immigration_df.groupby('region')['value'].mean().to_dict()
                    emigration_by_region = emigration_df.groupby('region')['value'].mean().to_dict()
                    
                    # Calculate net migration
                    net_flows_list = []
                    
                    for region in set(immigration_by_region.keys()) | set(emigration_by_region.keys()):
                        inflow = immigration_by_region.get(region, 0)
                        outflow = emigration_by_region.get(region, 0)
                        net_flow = inflow - outflow
                        
                        net_flows_list.append({
                            'region': region,
                            'inflow': inflow,
                            'outflow': outflow,
                            'net_flow': net_flow
                        })
                    
                    net_flows = pd.DataFrame(net_flows_list)
                    net_flows = net_flows.sort_values('net_flow', ascending=False)
        
        # Decide which visualization to create based on available data and selected metric
        if metric == 'Net Flow Volume' and net_flows is not None and not net_flows.empty:
            # Create horizontal bar chart of net flows
            fig = go.Figure()
            
            # Filter if needed
            if regions and 'region' in net_flows.columns:
                net_flows = net_flows[net_flows['region'].isin(regions)]
            
            # Sort values
            net_flows = net_flows.sort_values('net_flow')
            
            # Create colors based on net flow direction
            colors = [self.color_scheme['tertiary'] if val >= 0 else self.color_scheme['secondary'] for val in net_flows['net_flow']]
            
            fig.add_trace(go.Bar(
                y=net_flows['region'],
                x=net_flows['net_flow'],
                orientation='h',
                marker_color=colors,
                text=net_flows['net_flow'].round(2),
                textposition='auto'
            ))
            
            # Add zero line
            fig.add_shape(
                type="line",
                x0=0,
                y0=-0.5,
                x1=0,
                y1=len(net_flows) - 0.5,
                line=dict(color="black", width=1, dash="dash")
            )
            
            # Update layout
            fig.update_layout(
                title="Net Population Flow by Region",
                xaxis_title="Net Flow (millions)",
                yaxis_title="Region",
                height=max(400, len(net_flows) * 30)  # Scale height based on number of regions
            )
            
        elif metric == 'Flow Percentage' and net_flows is not None and not net_flows.empty:
            # Create flow percentage visualization
            fig = go.Figure()
            
            # Filter if needed
            if regions and 'region' in net_flows.columns:
                net_flows = net_flows[net_flows['region'].isin(regions)]
            
            # Calculate percentages
            net_flows['total_flow'] = net_flows['inflow'] + net_flows['outflow']
            net_flows['inflow_pct'] = net_flows['inflow'] / net_flows['total_flow'] * 100
            net_flows['outflow_pct'] = net_flows['outflow'] / net_flows['total_flow'] * 100
            
            # Sort by total flow
            net_flows = net_flows.sort_values('total_flow', ascending=False)
            
            # Create stacked bar chart
            fig.add_trace(go.Bar(
                y=net_flows['region'],
                x=net_flows['inflow_pct'],
                orientation='h',
                name='Inflow %',
                marker_color=self.color_scheme['tertiary'],
                text=net_flows['inflow_pct'].round(1).astype(str) + '%',
                textposition='auto'
            ))
            
            fig.add_trace(go.Bar(
                y=net_flows['region'],
                x=net_flows['outflow_pct'],
                orientation='h',
                name='Outflow %',
                marker_color=self.color_scheme['secondary'],
                text=net_flows['outflow_pct'].round(1).astype(str) + '%',
                textposition='auto'
            ))
            
            # Update layout
            fig.update_layout(
                title="Population Flow Percentages by Region",
                xaxis_title="Percentage of Total Flow",
                yaxis_title="Region",
                barmode='stack',
                height=max(400, len(net_flows) * 30)  # Scale height based on number of regions
            )
            
        elif metric == 'Year-over-Year Change' and 'year' in data.columns:
            # Create year-over-year change visualization
            df = data.copy()
            
            # Filter data if needed
            if regions:
                # Filter for specific regions
                if 'region' in df.columns:
                    df = df[df['region'].isin(regions)].copy()
            
            # Check if we have appropriate metrics
            migration_metrics = ['Immigration', 'Emigration', 'Net Migration']
            available_metrics = [m for m in migration_metrics if m in df['metric'].values] if 'metric' in df.columns else []
            
            if available_metrics:
                # Focus on one metric (preferably Net Migration)
                target_metric = 'Net Migration' if 'Net Migration' in available_metrics else available_metrics[0]
                metric_df = df[df['metric'] == target_metric].copy()
                
                # Group by region and year
                if 'region' in metric_df.columns and 'year' in metric_df.columns:
                    pivot_data = metric_df.pivot_table(
                        values='value',
                        index='year',
                        columns='region',
                        aggfunc='mean'
                    )
                    
                    # Calculate year-over-year percentage change
                    pct_change = pivot_data.pct_change() * 100
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=pct_change.values,
                        x=pct_change.columns,
                        y=pct_change.index,
                        colorscale='RdBu_r',  # Red for negative, Blue for positive
                        zmid=0,  # Center colorscale at zero
                        text=np.round(pct_change.values, 1),
                        texttemplate='%{text}%',
                        colorbar=dict(title='Change %')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Year-over-Year Change in {target_metric} (%)",
                        xaxis_title="Region",
                        yaxis_title="Year",
                        height=600
                    )
                else:
                    # Create empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Insufficient data for year-over-year analysis",
                        showarrow=False,
                        font=dict(size=20)
                    )
                    fig.update_layout(
                        title="Year-over-Year Change Analysis",
                        height=600
                    )
            else:
                # Create empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    text="No migration metrics available for year-over-year analysis",
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(
                    title="Year-over-Year Change Analysis",
                    height=600
                )
        elif flow_matrix is not None and not flow_matrix.empty:
            # Create flow matrix heatmap
            fig = go.Figure(data=go.Heatmap(
                z=flow_matrix.values,
                x=flow_matrix.columns,
                y=flow_matrix.index,
                colorscale='Viridis',
                text=np.round(flow_matrix.values, 2),
                texttemplate='%{text}',
                colorbar=dict(title='Flow Volume (millions)')
            ))
            
            # Update layout
            fig.update_layout(
                title="Inter-region Population Flow Matrix",
                xaxis_title="To Region",
                yaxis_title="From Region",
                height=max(500, len(flow_matrix) * 40)  # Scale height based on matrix size
            )
        else:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {metric} analysis",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=f"Flow Direction Analysis: {metric}",
                height=600
            )
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"data/visualizations/flow_direction_{metric.replace(' ', '_').lower()}_{timestamp}.html"
        fig.write_html(filename)
        print(f"Saved flow direction visualization to {filename}")
        
        return fig
