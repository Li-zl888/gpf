import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
import json

class DataProcessor:
    """
    A class to process and clean the raw population flow data.
    用于处理和清洗原始人口流动数据的类。
    """

    def __init__(self):
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

    def process_data(self, raw_data, regions=None, years=None):
        """
        Process and clean the raw data.
        处理和清洗原始数据。

        Args:
            raw_data (list): List of dictionaries containing raw data
                             包含原始数据的字典列表
            regions (list): List of regions to filter for (if None, use all regions)
                           要筛选的地区列表（如果为None，则使用所有地区）
            years (range): Range of years to filter for (if None, use all years)
                          要筛选的年份范围（如果为None，则使用所有年份）

        Returns:
            pd.DataFrame: Processed data
                         处理后的数据
        """
        print("Processing data...")

        # Check if we have any data to process
        if not raw_data or len(raw_data) == 0:
            print("No data to process. Generating sample dataset...")
            # Create a minimal dataset to allow the app to function
            raw_data = self._generate_minimal_dataset()

        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(raw_data)

        # Ensure we have the required columns
        required_columns = ['region', 'year', 'value', 'metric']
        for col in required_columns:
            if col not in df.columns:
                if col == 'region' and 'from_region' in df.columns:
                    # For flow data, use from_region as region
                    df['region'] = df['from_region']
                else:
                    # Add a default column if missing
                    if col == 'region':
                        df[col] = 'Guangdong'
                    elif col == 'year':
                        df[col] = 2023
                    elif col == 'value':
                        df[col] = 1.0
                    elif col == 'metric':
                        df[col] = 'Total Population'

        # Basic data cleaning
        df = self._clean_data(df)

        # Filter data if regions or years specified
        if regions:
            # Filter for specific regions, considering both 'region' and potentially 'from_region'/'to_region'
            if 'region' in df.columns:
                region_mask = df['region'].isin(regions)
            else:
                region_mask = pd.Series(False, index=df.index)

            # Also include inter-region flows if both from and to regions are in the list
            if 'from_region' in df.columns and 'to_region' in df.columns:
                flow_mask = df['from_region'].isin(regions) | df['to_region'].isin(regions)
                region_mask = region_mask | flow_mask

            df = df[region_mask].copy()

        if years:
            # Convert years to integers if they're not already
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
                df = df[df['year'].isin(years)].copy()

        # Calculate derived metrics
        df = self._calculate_derived_metrics(df)

        # Final data cleaning to ensure JSON compatibility
        # Replace NaN values with None for JSON serialization
        df = df.replace({np.nan: None})

        # Ensure numeric columns are properly formatted
        if 'value' in df.columns:
            # Convert any string numbers to float
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"data/processed_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved processed data to {filename}")

        return df

    def _generate_minimal_dataset(self):
        """
        Generate a minimal dataset with required fields to enable app functionality.
        Used only if no real data is available.
        生成具有必要字段的最小数据集，以启用应用功能。
        仅在没有真实数据可用时使用。

        Returns:
            list: List of dictionaries containing minimal data
                 包含最小数据的字典列表
        """
        minimal_data = []

        # Basic cities
        cities = ["Guangzhou", "Shenzhen", "Dongguan", "Foshan"]

        # Generate some basic data points
        for city in cities:
            for year in range(2015, 2024):
                # Simple population model
                base_pop = {"Guangzhou": 14.5, "Shenzhen": 12.6, "Dongguan": 8.3, "Foshan": 7.9}
                population = base_pop.get(city, 5.0) * (1 + (year - 2015) * 0.02)

                # Add total population
                minimal_data.append({
                    'region': city,
                    'year': year,
                    'value': round(population, 2),
                    'metric': 'Total Population',
                    'source': 'Generated'
                })

                # Add immigration
                minimal_data.append({
                    'region': city,
                    'year': year,
                    'value': round(population * 0.1, 2),
                    'metric': 'Immigration',
                    'source': 'Generated'
                })

                # Add emigration
                minimal_data.append({
                    'region': city,
                    'year': year,
                    'value': round(population * 0.08, 2),
                    'metric': 'Emigration',
                    'source': 'Generated'
                })

        return minimal_data

    def _clean_data(self, df):
        """
        Clean the raw data.
        清洗原始数据。

        Args:
            df (pd.DataFrame): Raw data DataFrame
                              原始数据DataFrame

        Returns:
            pd.DataFrame: Cleaned data
                         清洗后的数据
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Handle missing values
        if 'value' in df.columns:
            # Convert value column to numeric, forcing errors to NaN
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            # Drop rows with missing values in critical columns
            df = df.dropna(subset=['value'])

        # Standardize region names
        if 'region' in df.columns:
            df['region'] = df['region'].str.strip()

            # Handle possible variations in region names
            region_mapping = {
                'GuangZhou': 'Guangzhou',
                'ShenZhen': 'Shenzhen',
                'DongGuan': 'Dongguan',
                'FoShan': 'Foshan',
                'ZhongShan': 'Zhongshan',
                'ZhuHai': 'Zhuhai',
                'HuiZhou': 'Huizhou',
                'JiangMen': 'Jiangmen',
                'guangzhou': 'Guangzhou',
                'shenzhen': 'Shenzhen',
                'dongguan': 'Dongguan',
                'foshan': 'Foshan',
                'zhongshan': 'Zhongshan',
                'zhuhai': 'Zhuhai',
                'huizhou': 'Huizhou',
                'jiangmen': 'Jiangmen',
                # Add other variations as needed
            }

            df['region'] = df['region'].replace(region_mapping)

        # Also standardize from_region and to_region if they exist
        for col in ['from_region', 'to_region']:
            if col in df.columns:
                df[col] = df[col].str.strip()
                df[col] = df[col].replace(region_mapping)

        # Standardize year format
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(int)

        # Standardize metric names
        if 'metric' in df.columns:
            # Convert to title case and strip whitespace
            df['metric'] = df['metric'].str.strip().str.title()

            # Map similar metrics to standard names
            metric_mapping = {
                'Immigration': 'Immigration',
                'Inflow': 'Immigration',
                'In-Migration': 'Immigration',
                'Inmigration': 'Immigration',
                'Emigration': 'Emigration',
                'Outflow': 'Emigration',
                'Out-Migration': 'Emigration',
                'Outmigration': 'Emigration',
                'Total Population': 'Total Population',
                'Population': 'Total Population',
                'Residents': 'Total Population',
                'Inter-City Migration': 'Inter-city Migration',
                'Intercity Migration': 'Inter-city Migration',
                'Migration Flow': 'Migration Flow',
                'Provincial Immigration': 'Provincial Immigration',
                'Provincial Emigration': 'Provincial Emigration',
                'Urban To Rural Flow': 'Urban to Rural Flow',
                'Rural To Urban Flow': 'Rural to Urban Flow',
                'Migration By Education': 'Migration by Education',
                # Add other mappings as needed
            }

            df['metric'] = df['metric'].replace(metric_mapping)

        # Handle outliers
        # Simple example: remove values that are more than 3 standard deviations from the mean
        if 'value' in df.columns:
            # Group by metric to handle different scales
            for metric in df['metric'].unique():
                metric_mask = df['metric'] == metric
                metric_values = df.loc[metric_mask, 'value']

                mean = metric_values.mean()
                std = metric_values.std()

                # Only apply if we have enough data points
                if len(metric_values) > 10:
                    # Identify outliers
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std

                    # Replace outliers with NaN
                    outlier_mask = (metric_values < lower_bound) | (metric_values > upper_bound)
                    df.loc[metric_mask & outlier_mask, 'value'] = np.nan

            # Drop rows with NaN values after outlier removal
            df = df.dropna(subset=['value'])

        # Add source reliability column (for demonstration)
        if 'source' in df.columns:
            source_reliability = {
                'Guangdong Statistical Bureau': 0.9,
                'China National Bureau of Statistics': 0.95,
                'Guangdong Migration Reports': 0.85,
                'Academic Research Papers': 0.8
            }

            df['source_reliability'] = df['source'].map(source_reliability)

        return df

    def _calculate_derived_metrics(self, df):
        """
        Calculate derived metrics from the base data.
        从基础数据计算派生指标。

        Args:
            df (pd.DataFrame): Cleaned data
                              清洗后的数据

        Returns:
            pd.DataFrame: Data with additional derived metrics
                         包含额外派生指标的数据
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Group data by region and year to calculate net migration
        if 'metric' in df.columns and ('Immigration' in df['metric'].values and 'Emigration' in df['metric'].values):
            # Process cities with both immigration and emigration data
            regions = df[df['metric'].isin(['Immigration', 'Emigration'])]['region'].unique()
            years = df['year'].unique()

            net_migration_data = []

            for region in regions:
                for year in years:
                    # Get immigration and emigration values
                    immigration = df[(df['region'] == region) & (df['year'] == year) & (df['metric'] == 'Immigration')]['value'].sum()
                    emigration = df[(df['region'] == region) & (df['year'] == year) & (df['metric'] == 'Emigration')]['value'].sum()

                    # Only calculate if both values exist
                    if immigration > 0 and emigration > 0:
                        net_migration = immigration - emigration

                        # Add to new data
                        net_migration_data.append({
                            'region': region,
                            'year': year,
                            'value': net_migration,
                            'metric': 'Net Migration',
                            'source': 'Calculated',
                            'source_reliability': 0.9  # Reliability of calculated metrics
                        })

            # Append net migration data to the DataFrame
            if net_migration_data:
                net_df = pd.DataFrame(net_migration_data)
                df = pd.concat([df, net_df], ignore_index=True)

        # Calculate population growth rate for regions with total population data
        if 'metric' in df.columns and 'Total Population' in df['metric'].values:
            regions = df[df['metric'] == 'Total Population']['region'].unique()

            growth_rate_data = []

            for region in regions:
                # Get population data sorted by year
                pop_data = df[(df['region'] == region) & (df['metric'] == 'Total Population')].sort_values('year')

                if len(pop_data) >= 2:  # Need at least two years to calculate growth
                    for i in range(1, len(pop_data)):
                        current_year = pop_data.iloc[i]['year']
                        current_pop = pop_data.iloc[i]['value']
                        prev_pop = pop_data.iloc[i-1]['value']
                        prev_year = pop_data.iloc[i-1]['year']

                        # Calculate annual growth rate
                        years_diff = current_year - prev_year
                        if years_diff > 0 and prev_pop > 0:
                            growth_rate = (current_pop / prev_pop) ** (1 / years_diff) - 1

                            growth_rate_data.append({
                                'region': region,
                                'year': current_year,
                                'value': growth_rate * 100,  # Convert to percentage
                                'metric': 'Population Growth Rate',
                                'source': 'Calculated',
                                'source_reliability': 0.85
                            })

            # Append growth rate data to the DataFrame
            if growth_rate_data:
                growth_df = pd.DataFrame(growth_rate_data)
                df = pd.concat([df, growth_df], ignore_index=True)

        # Calculate migration intensity (sum of immigration and emigration divided by population)
        if 'metric' in df.columns and ('Immigration' in df['metric'].values and
                                      'Emigration' in df['metric'].values and
                                      'Total Population' in df['metric'].values):
            regions = df['region'].unique()
            years = df['year'].unique()

            intensity_data = []

            for region in regions:
                for year in years:
                    # Get immigration, emigration, and population values
                    immigration = df[(df['region'] == region) & (df['year'] == year) & (df['metric'] == 'Immigration')]['value'].sum()
                    emigration = df[(df['region'] == region) & (df['year'] == year) & (df['metric'] == 'Emigration')]['value'].sum()
                    population = df[(df['region'] == region) & (df['year'] == year) & (df['metric'] == 'Total Population')]['value'].sum()

                    # Only calculate if all values exist and population is not zero
                    if immigration > 0 and emigration > 0 and population > 0:
                        migration_intensity = (immigration + emigration) / population

                        intensity_data.append({
                            'region': region,
                            'year': year,
                            'value': migration_intensity * 100,  # Convert to percentage
                            'metric': 'Migration Intensity',
                            'source': 'Calculated',
                            'source_reliability': 0.85
                        })

            # Append migration intensity data to the DataFrame
            if intensity_data:
                intensity_df = pd.DataFrame(intensity_data)
                df = pd.concat([df, intensity_df], ignore_index=True)

        return df
