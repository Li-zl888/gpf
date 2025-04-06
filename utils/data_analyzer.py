import re
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

class DataAnalyzer:
    """
    A class to analyze population flow data for Guangdong Province.
    用于分析广东省人口流动数据的类。
    """
    
    def __init__(self):
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
    
    def analyze(self, data, analysis_type="Overall Population Flow", regions=None, years=None):
        """
        Analyze the processed data based on the specified analysis type.
        根据指定的分析类型分析处理后的数据。
        
        Args:
            data (pd.DataFrame): Processed data
            analysis_type (str): Type of analysis to perform
            regions (list): List of regions to analyze (if None, use all regions)
            years (range): Range of years to analyze (if None, use all years)
            
        参数：
            data (pd.DataFrame): 处理后的数据
            analysis_type (str): 要执行的分析类型
            regions (list): 要分析的地区列表（如果为None，则使用所有地区）
            years (range): 要分析的年份范围（如果为None，则使用所有年份）
            
        Returns:
            dict: Analysis results
            
        返回：
            dict: 分析结果
        """
        print(f"Performing {analysis_type} analysis...")
        
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Filter data if regions or years specified
        if regions:
            if 'region' in df.columns:
                region_mask = df['region'].isin(regions)
            else:
                region_mask = pd.Series(False, index=df.index)
            
            if 'from_region' in df.columns and 'to_region' in df.columns:
                flow_mask = df['from_region'].isin(regions) | df['to_region'].isin(regions)
                region_mask = region_mask | flow_mask
            
            df = df[region_mask].copy()
        
        if years:
            if 'year' in df.columns:
                df = df[df['year'].isin(years)].copy()
        
        # Perform different analyses based on the specified type
        if analysis_type == "Overall Population Flow":
            results = self._analyze_overall_flow(df)
        elif analysis_type == "Inter-city Migration":
            results = self._analyze_intercity_migration(df)
        elif analysis_type == "Urban-Rural Movement":
            results = self._analyze_urban_rural_movement(df)
        elif analysis_type == "Demographic Analysis":
            results = self._analyze_demographics(df)
        else:
            # Default to overall analysis
            results = self._analyze_overall_flow(df)
        
        # Save analysis results
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"data/analysis_results_{analysis_type.replace(' ', '_').lower()}_{timestamp}.json"
        
        # Convert any non-serializable objects to strings
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict(orient='records')
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        # Save as JSON
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Saved analysis results to {filename}")
        
        return results
    
    def _analyze_overall_flow(self, df):
        """
        Analyze overall population flow patterns.
        分析总体人口流动模式。
        
        Args:
            df (pd.DataFrame): Filtered data
            
        参数：
            df (pd.DataFrame): 过滤后的数据
            
        Returns:
            dict: Analysis results
            
        返回：
            dict: 分析结果
        """
        results = {
            'overview': {},
            'insights': [],
            'statistical_data': None,
            'top_regions': {'inflow': None, 'outflow': None},
            'time_trends': None
        }
        
        # Get basic overview statistics
        if 'metric' in df.columns:
            # Total records by metric
            metric_counts = df['metric'].value_counts().to_dict()
            results['overview']['metrics_distribution'] = metric_counts
            
            # Average values by metric
            metric_avgs = df.groupby('metric')['value'].mean().to_dict()
            results['overview']['metric_averages'] = metric_avgs
        
        # Get regions with data
        if 'region' in df.columns:
            results['overview']['regions_count'] = len(df['region'].unique())
            results['overview']['regions'] = sorted(df['region'].unique().tolist())
        
        # Get years with data
        if 'year' in df.columns and not df['year'].empty:
            # Handle potential NaN values safely
            min_year = df['year'].dropna().min() if not df['year'].dropna().empty else 2010
            max_year = df['year'].dropna().max() if not df['year'].dropna().empty else 2023
            
            # Safely convert to integers
            try:
                results['overview']['years_span'] = [int(min_year), int(max_year)]
            except (ValueError, TypeError):
                results['overview']['years_span'] = [2010, 2023]  # Default if conversion fails
                
            results['overview']['years_count'] = len(df['year'].unique())
        
        # Analyze immigration patterns
        if 'metric' in df.columns and 'Immigration' in df['metric'].values:
            immigration_df = df[df['metric'] == 'Immigration'].copy()
            
            if not immigration_df.empty and 'region' in immigration_df.columns and 'year' in immigration_df.columns:
                # Calculate average immigration by region
                region_immigration = immigration_df.groupby('region')['value'].mean().reset_index()
                region_immigration = region_immigration.sort_values('value', ascending=False)
                
                # Top regions by immigration
                results['top_regions']['inflow'] = region_immigration.head(10).to_dict(orient='records')
                
                # Check for time trends in immigration
                if len(immigration_df['year'].unique()) > 1:
                    immigration_trend = immigration_df.groupby('year')['value'].sum().reset_index()
                    immigration_trend = immigration_trend.sort_values('year')
                    
                    # Calculate year-over-year change
                    immigration_trend['yoy_change'] = immigration_trend['value'].pct_change() * 100
                    
                    # Add to results
                    results['time_trends'] = {
                        'immigration_trend': immigration_trend.to_dict(orient='records')
                    }
                    
                    # Add insight about trend
                    trend_direction = "increasing" if immigration_trend['yoy_change'].mean() > 0 else "decreasing"
                    results['insights'].append(
                        f"Overall immigration to Guangdong Province has been {trend_direction} "
                        f"at an average rate of {abs(immigration_trend['yoy_change'].mean()):.2f}% per year."
                    )
        
        # Analyze emigration patterns
        if 'metric' in df.columns and 'Emigration' in df['metric'].values:
            emigration_df = df[df['metric'] == 'Emigration'].copy()
            
            if not emigration_df.empty and 'region' in emigration_df.columns and 'year' in emigration_df.columns:
                # Calculate average emigration by region
                region_emigration = emigration_df.groupby('region')['value'].mean().reset_index()
                region_emigration = region_emigration.sort_values('value', ascending=False)
                
                # Top regions by emigration
                results['top_regions']['outflow'] = region_emigration.head(10).to_dict(orient='records')
                
                # Check for time trends in emigration
                if len(emigration_df['year'].unique()) > 1:
                    emigration_trend = emigration_df.groupby('year')['value'].sum().reset_index()
                    emigration_trend = emigration_trend.sort_values('year')
                    
                    # Calculate year-over-year change
                    emigration_trend['yoy_change'] = emigration_trend['value'].pct_change() * 100
                    
                    # Add to results
                    if 'time_trends' not in results:
                        results['time_trends'] = {}
                    results['time_trends']['emigration_trend'] = emigration_trend.to_dict(orient='records')
                    
                    # Add insight about trend
                    trend_direction = "increasing" if emigration_trend['yoy_change'].mean() > 0 else "decreasing"
                    results['insights'].append(
                        f"Overall emigration from Guangdong Province has been {trend_direction} "
                        f"at an average rate of {abs(emigration_trend['yoy_change'].mean()):.2f}% per year."
                    )
        
        # Analyze net migration
        if 'metric' in df.columns and 'Net Migration' in df['metric'].values:
            net_migration_df = df[df['metric'] == 'Net Migration'].copy()
            
            if not net_migration_df.empty and 'region' in net_migration_df.columns:
                # Average net migration by region
                region_net = net_migration_df.groupby('region')['value'].mean().reset_index()
                region_net = region_net.sort_values('value', ascending=False)
                
                # Identify regions with positive and negative net migration
                positive_regions = region_net[region_net['value'] > 0]
                negative_regions = region_net[region_net['value'] < 0]
                
                # Add insights about net migration
                if not positive_regions.empty:
                    top_gaining = positive_regions.iloc[0]['region']
                    results['insights'].append(
                        f"{top_gaining} has the highest positive net migration, "
                        f"gaining an average of {positive_regions.iloc[0]['value']:.2f} million people."
                    )
                
                if not negative_regions.empty:
                    top_losing = negative_regions.iloc[-1]['region']
                    results['insights'].append(
                        f"{top_losing} has the highest negative net migration, "
                        f"losing an average of {abs(negative_regions.iloc[-1]['value']):.2f} million people."
                    )
        
        # Analyze population growth rate
        if 'metric' in df.columns and 'Population Growth Rate' in df['metric'].values:
            growth_df = df[df['metric'] == 'Population Growth Rate'].copy()
            
            if not growth_df.empty and 'region' in growth_df.columns:
                # Average growth rate by region
                region_growth = growth_df.groupby('region')['value'].mean().reset_index()
                region_growth = region_growth.sort_values('value', ascending=False)
                
                # Add statistical data
                results['statistical_data'] = region_growth.to_dict(orient='records')
                
                # Add insights about growth rates
                fastest_growing = region_growth.iloc[0]['region']
                slowest_growing = region_growth.iloc[-1]['region']
                
                results['insights'].append(
                    f"{fastest_growing} has the highest population growth rate at {region_growth.iloc[0]['value']:.2f}% annually."
                )
                
                if region_growth.iloc[-1]['value'] < 0:
                    results['insights'].append(
                        f"{slowest_growing} is experiencing population decline at {region_growth.iloc[-1]['value']:.2f}% annually."
                    )
                else:
                    results['insights'].append(
                        f"{slowest_growing} has the lowest population growth rate at {region_growth.iloc[-1]['value']:.2f}% annually."
                    )
        
        # Calculate correlation matrix between different metrics
        pivot_metrics = ['Immigration', 'Emigration', 'Net Migration', 'Population Growth Rate', 'Total Population']
        available_metrics = [m for m in pivot_metrics if m in df['metric'].values]
        
        if len(available_metrics) > 1 and 'region' in df.columns and 'year' in df.columns:
            # Create a pivot table with regions as rows, metrics as columns, and average values
            pivot_df = pd.pivot_table(
                df[df['metric'].isin(available_metrics)],
                values='value',
                index=['region'],
                columns=['metric'],
                aggfunc='mean'
            )
            
            # Calculate correlation matrix
            corr_matrix = pivot_df.corr()
            
            # Add to results
            results['correlation_matrix'] = corr_matrix.to_dict()
            
            # Add insights based on correlations
            for i in range(len(available_metrics)):
                for j in range(i+1, len(available_metrics)):
                    metric1 = available_metrics[i]
                    metric2 = available_metrics[j]
                    
                    if metric1 in corr_matrix.index and metric2 in corr_matrix.columns:
                        corr_value = corr_matrix.loc[metric1, metric2]
                        
                        if abs(corr_value) > 0.7:
                            direction = "positive" if corr_value > 0 else "negative"
                            strength = "strong"
                        elif abs(corr_value) > 0.4:
                            direction = "positive" if corr_value > 0 else "negative"
                            strength = "moderate"
                        else:
                            continue  # Skip weak correlations
                        
                        results['insights'].append(
                            f"There is a {strength} {direction} correlation ({corr_value:.2f}) between {metric1} and {metric2}."
                        )
        
        # Clustering analysis to identify similar regions
        if 'region' in df.columns and len(df['region'].unique()) >= 3:
            # Create features for clustering
            clustering_metrics = ['Immigration', 'Emigration', 'Net Migration', 'Total Population']
            available_clustering_metrics = [m for m in clustering_metrics if m in df['metric'].values]
            
            if len(available_clustering_metrics) >= 2:
                # Create a pivot table for clustering
                cluster_pivot = pd.pivot_table(
                    df[df['metric'].isin(available_clustering_metrics)],
                    values='value',
                    index=['region'],
                    columns=['metric'],
                    aggfunc='mean'
                )
                
                # Fill NaN values with column means
                cluster_pivot = cluster_pivot.fillna(cluster_pivot.mean())
                
                # Only proceed if we have enough data
                if len(cluster_pivot) >= 3 and not cluster_pivot.empty:
                    # Standardize the features
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(cluster_pivot)
                    
                    # Determine optimal number of clusters (simplified)
                    n_clusters = min(3, len(cluster_pivot) - 1)  # Use at most 3 clusters
                    
                    # Apply KMeans clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(scaled_features)
                    
                    # Add cluster labels to the pivot table
                    cluster_pivot['Cluster'] = cluster_labels
                    
                    # Get cluster characteristics
                    cluster_summary = cluster_pivot.groupby('Cluster').mean()
                    
                    # Identify cluster patterns
                    cluster_insights = []
                    
                    for cluster_id in range(n_clusters):
                        regions_in_cluster = cluster_pivot[cluster_pivot['Cluster'] == cluster_id].index.tolist()
                        
                        # Skip if no regions in this cluster
                        if not regions_in_cluster:
                            continue
                        
                        # Get characteristics of this cluster
                        cluster_chars = {}
                        for metric in available_clustering_metrics:
                            if metric in cluster_summary.columns:
                                cluster_chars[metric] = cluster_summary.loc[cluster_id, metric]
                        
                        # Determine dominant characteristics
                        high_metrics = []
                        low_metrics = []
                        
                        for metric, value in cluster_chars.items():
                            # Compare to overall average
                            overall_avg = cluster_pivot[metric].mean()
                            
                            if value > overall_avg * 1.2:  # 20% above average
                                high_metrics.append(metric)
                            elif value < overall_avg * 0.8:  # 20% below average
                                low_metrics.append(metric)
                        
                        # Create cluster description
                        description = f"Cluster {cluster_id+1} regions "
                        
                        if high_metrics:
                            description += f"have high {', '.join(high_metrics)}"
                            
                            if low_metrics:
                                description += f" and low {', '.join(low_metrics)}"
                        elif low_metrics:
                            description += f"have low {', '.join(low_metrics)}"
                        else:
                            description += "have average values across all metrics"
                        
                        description += f". Regions in this cluster: {', '.join(regions_in_cluster[:5])}"
                        
                        if len(regions_in_cluster) > 5:
                            description += f" and {len(regions_in_cluster) - 5} more"
                        
                        cluster_insights.append(description)
                    
                    # Add cluster insights
                    results['insights'].extend(cluster_insights)
        
        return results
    
    def _analyze_intercity_migration(self, df):
        """
        Analyze inter-city migration patterns.
        分析城市间迁移模式。
        
        Args:
            df (pd.DataFrame): Filtered data
            
        参数：
            df (pd.DataFrame): 过滤后的数据
            
        Returns:
            dict: Analysis results
            
        返回：
            dict: 分析结果
        """
        results = {
            'overview': {},
            'insights': [],
            'flow_matrix': None,
            'top_flows': None,
            'net_flows': None
        }
        
        # Check if we have inter-city flow data
        if 'from_region' in df.columns and 'to_region' in df.columns:
            flow_df = df[df['metric'] == 'Inter-city Migration'].copy()
            
            if not flow_df.empty:
                # Get basic statistics
                results['overview']['total_flows'] = len(flow_df)
                results['overview']['from_regions'] = sorted(flow_df['from_region'].unique().tolist())
                results['overview']['to_regions'] = sorted(flow_df['to_region'].unique().tolist())
                results['overview']['total_volume'] = flow_df['value'].sum()
                
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
                
                # Add flow matrix to results
                results['flow_matrix'] = flow_matrix.to_dict()
                
                # Find top flows
                flow_df['flow_name'] = flow_df['from_region'] + ' to ' + flow_df['to_region']
                top_flows = flow_df.sort_values('value', ascending=False).head(10)
                results['top_flows'] = top_flows[['flow_name', 'value', 'year']].to_dict(orient='records')
                
                # Calculate net flows
                net_flows = {}
                
                for region in all_regions:
                    # Total outflow
                    if region in flow_df['from_region'].values:
                        outflow = flow_df[flow_df['from_region'] == region]['value'].sum()
                    else:
                        outflow = 0
                    
                    # Total inflow
                    if region in flow_df['to_region'].values:
                        inflow = flow_df[flow_df['to_region'] == region]['value'].sum()
                    else:
                        inflow = 0
                    
                    # Net flow
                    net_flow = inflow - outflow
                    
                    net_flows[region] = {
                        'inflow': inflow,
                        'outflow': outflow,
                        'net_flow': net_flow
                    }
                
                # Sort regions by net flow
                sorted_regions = sorted(net_flows.keys(), key=lambda r: net_flows[r]['net_flow'], reverse=True)
                
                # Create sorted net flow results
                sorted_net_flows = []
                for region in sorted_regions:
                    sorted_net_flows.append({
                        'region': region,
                        'inflow': net_flows[region]['inflow'],
                        'outflow': net_flows[region]['outflow'],
                        'net_flow': net_flows[region]['net_flow']
                    })
                
                results['net_flows'] = sorted_net_flows
                
                # Add insights
                # Top inflow regions
                top_inflow_region = sorted_net_flows[0]['region']
                results['insights'].append(
                    f"{top_inflow_region} has the highest net inflow with "
                    f"{sorted_net_flows[0]['net_flow']:.2f} million more inflow than outflow."
                )
                
                # Top outflow regions
                bottom_region = sorted_net_flows[-1]['region']
                if sorted_net_flows[-1]['net_flow'] < 0:
                    results['insights'].append(
                        f"{bottom_region} has the highest net outflow with "
                        f"{abs(sorted_net_flows[-1]['net_flow']):.2f} million more outflow than inflow."
                    )
                
                # Strongest connections
                if len(top_flows) > 0:
                    strongest_flow = top_flows.iloc[0]
                    results['insights'].append(
                        f"The strongest migration flow is from {strongest_flow['from_region']} to "
                        f"{strongest_flow['to_region']} with {strongest_flow['value']:.2f} million people."
                    )
                
                # Identify major migration corridors
                if len(top_flows) >= 3:
                    results['insights'].append(
                        f"Major migration corridors include {top_flows.iloc[0]['flow_name']}, "
                        f"{top_flows.iloc[1]['flow_name']}, and {top_flows.iloc[2]['flow_name']}."
                    )
        else:
            # Try to derive flow patterns from immigration/emigration data
            if 'metric' in df.columns and 'region' in df.columns:
                # Get immigration data
                immigration_df = df[df['metric'] == 'Immigration'].copy() if 'Immigration' in df['metric'].values else None
                
                # Get emigration data
                emigration_df = df[df['metric'] == 'Emigration'].copy() if 'Emigration' in df['metric'].values else None
                
                if immigration_df is not None and emigration_df is not None:
                    # Basic overview
                    results['overview']['regions'] = sorted(df['region'].unique().tolist())
                    
                    # Calculate total immigration and emigration by region
                    immigration_by_region = immigration_df.groupby('region')['value'].mean().to_dict()
                    emigration_by_region = emigration_df.groupby('region')['value'].mean().to_dict()
                    
                    # Calculate net migration
                    net_flows = []
                    
                    for region in set(immigration_by_region.keys()) | set(emigration_by_region.keys()):
                        inflow = immigration_by_region.get(region, 0)
                        outflow = emigration_by_region.get(region, 0)
                        net_flow = inflow - outflow
                        
                        net_flows.append({
                            'region': region,
                            'inflow': inflow,
                            'outflow': outflow,
                            'net_flow': net_flow
                        })
                    
                    # Sort by net flow
                    net_flows = sorted(net_flows, key=lambda x: x['net_flow'], reverse=True)
                    
                    results['net_flows'] = net_flows
                    
                    # Add insights
                    if net_flows:
                        top_inflow_region = net_flows[0]['region']
                        results['insights'].append(
                            f"{top_inflow_region} has the highest net migration with "
                            f"{net_flows[0]['net_flow']:.2f} million more immigration than emigration."
                        )
                        
                        bottom_region = net_flows[-1]['region']
                        if net_flows[-1]['net_flow'] < 0:
                            results['insights'].append(
                                f"{bottom_region} has the highest negative net migration with "
                                f"{abs(net_flows[-1]['net_flow']):.2f} million more emigration than immigration."
                            )
        
        # Add general insights if none specific to flow data
        if not results['insights']:
            # Add generic insights based on available data
            if 'metric' in df.columns and 'region' in df.columns:
                regions = df['region'].unique()
                
                if len(regions) > 0:
                    results['insights'].append(
                        f"Analysis includes data for {len(regions)} regions in Guangdong Province."
                    )
                
                years_range = None
                if 'year' in df.columns:
                    min_year = df['year'].min()
                    max_year = df['year'].max()
                    
                    if min_year and max_year:
                        years_range = f"from {min_year} to {max_year}"
                        results['insights'].append(
                            f"Data covers population flows {years_range}."
                        )
                
                # Add placeholder insight about major cities
                major_cities = [r for r in regions if r in ["Guangzhou", "Shenzhen", "Dongguan", "Foshan"]]
                if major_cities:
                    results['insights'].append(
                        f"Major urban centers in the analysis include {', '.join(major_cities)}."
                    )
        
        return results
    
    def _analyze_urban_rural_movement(self, df):
        """
        Analyze urban-rural population movement patterns.
        分析城乡人口流动模式。
        
        Args:
            df (pd.DataFrame): Filtered data
            
        参数：
            df (pd.DataFrame): 过滤后的数据
            
        Returns:
            dict: Analysis results
            
        返回：
            dict: 分析结果
        """
        results = {
            'overview': {},
            'insights': [],
            'urban_rural_flows': None,
            'rural_urban_flows': None,
            'time_trends': None
        }
        
        # Check for specific urban-rural flow metrics
        urban_rural_metrics = [
            'Urban to Rural Flow', 
            'Rural to Urban Flow'
        ]
        
        has_urban_rural_data = False
        if 'metric' in df.columns:
            available_metrics = [m for m in urban_rural_metrics if m in df['metric'].values]
            has_urban_rural_data = len(available_metrics) > 0
        
        if has_urban_rural_data:
            # Extract urban to rural flow data
            if 'Urban to Rural Flow' in df['metric'].values:
                urban_rural_df = df[df['metric'] == 'Urban to Rural Flow'].copy()
                
                # Aggregate by year
                if 'year' in urban_rural_df.columns:
                    urban_rural_trend = urban_rural_df.groupby('year')['value'].sum().reset_index()
                    urban_rural_trend = urban_rural_trend.sort_values('year')
                    
                    results['urban_rural_flows'] = urban_rural_trend.to_dict(orient='records')
            
            # Extract rural to urban flow data
            if 'Rural to Urban Flow' in df['metric'].values:
                rural_urban_df = df[df['metric'] == 'Rural to Urban Flow'].copy()
                
                # Aggregate by year
                if 'year' in rural_urban_df.columns:
                    rural_urban_trend = rural_urban_df.groupby('year')['value'].sum().reset_index()
                    rural_urban_trend = rural_urban_trend.sort_values('year')
                    
                    results['rural_urban_flows'] = rural_urban_trend.to_dict(orient='records')
            
            # Calculate net urban-rural flow
            if 'urban_rural_flows' in results and 'rural_urban_flows' in results:
                urban_rural_years = {item['year']: item['value'] for item in results['urban_rural_flows']}
                rural_urban_years = {item['year']: item['value'] for item in results['rural_urban_flows']}
                
                # Find common years
                common_years = sorted(set(urban_rural_years.keys()) & set(rural_urban_years.keys()))
                
                if common_years:
                    # Calculate net flow (positive = net urbanization, negative = net ruralization)
                    net_flow = [
                        {
                            'year': year,
                            'urban_to_rural': urban_rural_years[year],
                            'rural_to_urban': rural_urban_years[year],
                            'net_flow': rural_urban_years[year] - urban_rural_years[year]
                        }
                        for year in common_years
                    ]
                    
                    results['time_trends'] = net_flow
                    
                    # Calculate average direction
                    avg_net_flow = sum(item['net_flow'] for item in net_flow) / len(net_flow)
                    
                    # Add insights
                    if avg_net_flow > 0:
                        results['insights'].append(
                            f"Guangdong Province is experiencing net urbanization with an average of "
                            f"{avg_net_flow:.2f} million more people moving from rural to urban areas than the reverse."
                        )
                    else:
                        results['insights'].append(
                            f"Guangdong Province is experiencing net ruralization with an average of "
                            f"{abs(avg_net_flow):.2f} million more people moving from urban to rural areas than the reverse."
                        )
                    
                    # Check for trends over time
                    if len(common_years) > 1:
                        first_net = net_flow[0]['net_flow']
                        last_net = net_flow[-1]['net_flow']
                        
                        if (last_net > 0 and first_net < 0) or (last_net < 0 and first_net > 0):
                            # Flow direction has changed
                            change_year = None
                            for i in range(1, len(net_flow)):
                                if (net_flow[i-1]['net_flow'] * net_flow[i]['net_flow']) <= 0:  # Sign change
                                    change_year = net_flow[i]['year']
                                    break
                            
                            if change_year:
                                if last_net > 0:
                                    results['insights'].append(
                                        f"The flow direction changed from net ruralization to net urbanization around {change_year}."
                                    )
                                else:
                                    results['insights'].append(
                                        f"The flow direction changed from net urbanization to net ruralization around {change_year}."
                                    )
                        elif last_net > first_net:
                            results['insights'].append(
                                "The trend shows increasing urbanization over time."
                            )
                        elif last_net < first_net:
                            results['insights'].append(
                                "The trend shows decreasing urbanization over time."
                            )
        else:
            # Try to infer urban-rural patterns from city-level data
            if 'region' in df.columns and 'metric' in df.columns:
                # Identify major urban centers
                major_cities = ["Guangzhou", "Shenzhen", "Dongguan", "Foshan", "Zhuhai"]
                
                # Identify regions likely to be more rural
                rural_regions = [
                    r for r in df['region'].unique() 
                    if r not in major_cities and r in [
                        "Qingyuan", "Yunfu", "Heyuan", "Meizhou", "Shanwei", 
                        "Yangjiang", "Maoming", "Zhanjiang"
                    ]
                ]
                
                if major_cities and rural_regions and 'Immigration' in df['metric'].values and 'Emigration' in df['metric'].values:
                    # Calculate net migration for urban centers
                    urban_immigration = df[(df['region'].isin(major_cities)) & (df['metric'] == 'Immigration')]['value'].sum()
                    urban_emigration = df[(df['region'].isin(major_cities)) & (df['metric'] == 'Emigration')]['value'].sum()
                    urban_net = urban_immigration - urban_emigration
                    
                    # Calculate net migration for rural regions
                    rural_immigration = df[(df['region'].isin(rural_regions)) & (df['metric'] == 'Immigration')]['value'].sum()
                    rural_emigration = df[(df['region'].isin(rural_regions)) & (df['metric'] == 'Emigration')]['value'].sum()
                    rural_net = rural_immigration - rural_emigration
                    
                    # Add to overview
                    results['overview']['urban_regions'] = major_cities
                    results['overview']['rural_regions'] = rural_regions
                    results['overview']['urban_net_migration'] = urban_net
                    results['overview']['rural_net_migration'] = rural_net
                    
                    # Add insights
                    if urban_net > 0 and rural_net < 0:
                        results['insights'].append(
                            f"Analysis suggests urbanization trend with major urban centers gaining {urban_net:.2f} million "
                            f"people while rural areas losing {abs(rural_net):.2f} million people."
                        )
                    elif urban_net < 0 and rural_net > 0:
                        results['insights'].append(
                            f"Analysis suggests counter-urbanization trend with major urban centers losing {abs(urban_net):.2f} million "
                            f"people while rural areas gaining {rural_net:.2f} million people."
                        )
                    elif urban_net > 0 and rural_net > 0:
                        results['insights'].append(
                            f"Both urban centers (+{urban_net:.2f} million) and rural areas (+{rural_net:.2f} million) "
                            f"are experiencing population growth, suggesting overall population growth in the province."
                        )
                    elif urban_net < 0 and rural_net < 0:
                        results['insights'].append(
                            f"Both urban centers (-{abs(urban_net):.2f} million) and rural areas (-{abs(rural_net):.2f} million) "
                            f"are experiencing population decline, suggesting overall population decline in the province."
                        )
        
        # Add general insights if none specific to urban-rural data
        if not results['insights']:
            # Generic insights
            results['insights'].append(
                "Guangdong Province has been experiencing rapid urbanization as part of China's economic development."
            )
            
            results['insights'].append(
                "Urban centers like Guangzhou, Shenzhen, and Dongguan attract migrants due to job opportunities and higher wages."
            )
            
            results['insights'].append(
                "Rural-to-urban migration is a significant factor in population flow patterns in Guangdong Province."
            )
        
        return results
    
    def _analyze_demographics(self, df):
        """
        Analyze demographic factors in population flow.
        分析人口流动中的人口统计因素。
        
        Args:
            df (pd.DataFrame): Filtered data
            
        参数：
            df (pd.DataFrame): 过滤后的数据
            
        Returns:
            dict: Analysis results
            
        返回：
            dict: 分析结果
        """
        results = {
            'overview': {},
            'insights': [],
            'age_distribution': None,
            'education_distribution': None,
            'demographic_trends': None
        }
        
        # Check for demographic data
        has_demographic_data = False
        demographic_columns = ['demographic', 'group']
        
        if all(col in df.columns for col in demographic_columns):
            has_demographic_data = True
        
        if has_demographic_data:
            # Group data by demographic factors
            demographic_types = df['demographic'].unique()
            
            for demo_type in demographic_types:
                demo_df = df[df['demographic'] == demo_type].copy()
                
                # Group by the demographic group
                if 'group' in demo_df.columns:
                    group_data = demo_df.groupby('group')['value'].mean().reset_index()
                    group_data = group_data.sort_values('value', ascending=False)
                    
                    # Add to results
                    key_name = f"{demo_type.lower()}_distribution"
                    results[key_name] = group_data.to_dict(orient='records')
                    
                    # Add insights
                    top_group = group_data.iloc[0]['group']
                    bottom_group = group_data.iloc[-1]['group']
                    
                    results['insights'].append(
                        f"The {demo_type} group with the highest migration rate is {top_group} "
                        f"with an average of {group_data.iloc[0]['value']:.2f} million."
                    )
                    
                    results['insights'].append(
                        f"The {demo_type} group with the lowest migration rate is {bottom_group} "
                        f"with an average of {group_data.iloc[-1]['value']:.2f} million."
                    )
                
                # Check for time trends by demographic group
                if 'year' in demo_df.columns and 'group' in demo_df.columns:
                    # Get unique groups
                    groups = demo_df['group'].unique()
                    
                    # Create time trends for each group
                    time_trends = {}
                    
                    for group in groups:
                        group_time_df = demo_df[demo_df['group'] == group].copy()
                        time_trend = group_time_df.groupby('year')['value'].mean().reset_index()
                        time_trend = time_trend.sort_values('year')
                        
                        time_trends[group] = time_trend.to_dict(orient='records')
                    
                    # Add to results
                    key_name = f"{demo_type.lower()}_trends"
                    results[key_name] = time_trends
                    
                    # Identify trends
                    trend_insights = []
                    
                    for group, trend_data in time_trends.items():
                        if len(trend_data) > 1:
                            first_value = trend_data[0]['value']
                            last_value = trend_data[-1]['value']
                            
                            if last_value > first_value * 1.2:  # 20% increase
                                trend_insights.append(f"{group} shows significant increase over time")
                            elif first_value > last_value * 1.2:  # 20% decrease
                                trend_insights.append(f"{group} shows significant decrease over time")
                    
                    if trend_insights:
                        results['insights'].append(
                            f"Notable {demo_type} trends: {', '.join(trend_insights)}."
                        )
        else:
            # Try to infer demographic patterns from available data
            # Look for metrics that might contain demographic information
            demographic_metrics = [
                m for m in df['metric'].unique() 
                if any(keyword in m for keyword in ['Age', 'Education', 'Income', 'Occupation'])
            ] if 'metric' in df.columns else []
            
            for metric in demographic_metrics:
                metric_df = df[df['metric'] == metric].copy()
                
                # Extract demographic category from metric name
                category = metric.split()[0] if ' ' in metric else metric
                
                # Try to aggregate and analyze
                if 'region' in metric_df.columns and 'value' in metric_df.columns:
                    agg_data = metric_df.groupby('region')['value'].mean().reset_index()
                    agg_data = agg_data.sort_values('value', ascending=False)
                    
                    # Add to results
                    key_name = f"{category.lower()}_regional_distribution"
                    results[key_name] = agg_data.to_dict(orient='records')
                    
                    # Add insights
                    top_region = agg_data.iloc[0]['region']
                    results['insights'].append(
                        f"{top_region} has the highest {metric} with an average of {agg_data.iloc[0]['value']:.2f} million."
                    )
        
        # Add age-specific migration patterns if available
        age_metrics = [m for m in df['metric'].unique() if 'Age' in m] if 'metric' in df.columns else []
        
        if age_metrics:
            age_data = {}
            
            for metric in age_metrics:
                # Try to extract age group from metric name
                match = re.search(r'Age\s+(\S+)', metric)
                
                if match:
                    age_group = match.group(1)
                    age_df = df[df['metric'] == metric].copy()
                    
                    if 'value' in age_df.columns:
                        avg_value = age_df['value'].mean()
                        age_data[age_group] = avg_value
            
            if age_data:
                # Sort by age group if possible
                try:
                    # For simple formats like "0-14", "15-24", etc.
                    sorted_ages = sorted(age_data.keys(), key=lambda x: int(x.split('-')[0]) if '-' in x else int(x.replace('+', '')))
                except:
                    # If parsing fails, use alphabetical order
                    sorted_ages = sorted(age_data.keys())
                
                age_distribution = [{'group': age, 'value': age_data[age]} for age in sorted_ages]
                results['age_distribution'] = age_distribution
                
                # Add insights about age distribution
                young_adult_groups = [g for g in sorted_ages if ('15-24' in g or '25-34' in g)]
                older_groups = [g for g in sorted_ages if ('45-59' in g or '60+' in g)]
                
                if young_adult_groups and older_groups:
                    young_avg = sum(age_data[g] for g in young_adult_groups) / len(young_adult_groups)
                    older_avg = sum(age_data[g] for g in older_groups) / len(older_groups)
                    
                    if young_avg > older_avg * 1.5:  # Young adults migrate 50% more
                        results['insights'].append(
                            f"Young adults ({', '.join(young_adult_groups)}) have significantly higher migration rates "
                            f"than older populations ({', '.join(older_groups)})."
                        )
        
        # Add education-specific migration patterns if available
        edu_metrics = [m for m in df['metric'].unique() if 'Education' in m] if 'metric' in df.columns else []
        
        if edu_metrics:
            edu_data = {}
            
            for metric in edu_metrics:
                # Extract from the data
                edu_df = df[df['metric'] == metric].copy()
                
                if 'group' in edu_df.columns:
                    # Group already separated in data
                    for group in edu_df['group'].unique():
                        group_data = edu_df[edu_df['group'] == group]
                        edu_data[group] = group_data['value'].mean()
                else:
                    # Try a generic approach
                    if 'value' in edu_df.columns:
                        avg_value = edu_df['value'].mean()
                        edu_data[metric] = avg_value
            
            if edu_data:
                # Create education distribution
                edu_distribution = [{'group': edu, 'value': edu_data[edu]} for edu in edu_data.keys()]
                sorted_edu = sorted(edu_distribution, key=lambda x: x['value'], reverse=True)
                results['education_distribution'] = sorted_edu
                
                # Add insights about education
                if len(sorted_edu) >= 2:
                    highest_edu = sorted_edu[0]['group']
                    lowest_edu = sorted_edu[-1]['group']
                    
                    results['insights'].append(
                        f"People with {highest_edu} education have the highest migration rate, "
                        f"while those with {lowest_edu} education have the lowest."
                    )
        
        # Add general insights if none specific to demographic data
        if not results['insights']:
            results['insights'] = [
                "Young adults (ages 18-34) typically have the highest migration rates due to education and early career mobility.",
                "Higher education levels are generally associated with increased mobility and migration rates.",
                "Migration patterns differ significantly by age group, with young adults being the most mobile population segment.",
                "Family formation and children's education are significant factors in migration decisions for middle-aged adults."
            ]
        
        return results
