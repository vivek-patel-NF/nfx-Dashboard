import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
import os

class SocialMediaAnalytics:
    def __init__(self, mongo_uri: str, database: str, collection: str):
        """
        Initialize the Social Media Analytics application
        
        Args:
            mongo_uri (str): MongoDB connection URI
            database (str): Database name
            collection (str): Collection name
        """
        self.mongo_uri = mongo_uri
        self.database = database
        self.collection = collection
    
    def get_mongo_connection(self, _mongo_uri=None, _database=None):
        """
        Establish a secure MongoDB connection with robust error handling
        
        Args:
            _mongo_uri (str, optional): MongoDB URI to override instance URI
            _database (str, optional): Database name to override instance database
        
        Returns:
            pymongo.database.Database: MongoDB database connection
        """
        # Use provided URI or instance URI
        mongo_uri = _mongo_uri or self.mongo_uri
        database = _database or self.database

        try:
            client = pymongo.MongoClient(
                mongo_uri, 
                connectTimeoutMS=10000, 
                serverSelectionTimeoutMS=10000,
                maxPoolSize=50, 
                maxIdleTimeMS=30000
            )
            
            # Verify connection
            client.admin.command('ismaster')
            
            db = client[database]
            st.write("MongoDB connection established successfully")
            return db
        
        except pymongo.errors.ConnectionFailure as e:
            st.error(f"Failed to connect to MongoDB: {e}")
            return None
        
        except Exception as e:
            st.error(f"Unexpected error in MongoDB connection: {e}")
            return None
    
    def fetch_social_media_data(self) -> List[Dict[str, Any]]:
        """
        Fetch and aggregate social media channel state data with temporal information
        
        Returns:
            List of aggregated social media data
        """
        # Use the modified get_mongo_connection method
        db = self.get_mongo_connection()
        
        if db is None:  # Explicitly check if db is None
            st.error("Database connection failed")
            return []
        
        try:
            collection = db[self.collection]
            
            # Advanced Aggregation Pipeline with date-wise analytics
            pipeline = [
                {
                    # Convert created_at to date
                    '$addFields': {
                        'created_date': {
                            '$dateToString': {
                                'format': '%Y-%m-%d', 
                                'date': '$created_at'
                            }
                        }
                    }
                },
                {
                    '$group': {
                        '_id': { 
                            'channel': '$channel', 
                            'state': '$state',
                            'operation': '$operation',
                            'created_date': '$created_date'
                        },
                        'stateCount': { '$sum': 1 },
                        'totalBoostPriority': { '$sum': '$boost_priority' },
                        'earliestTimestamp': { '$min': '$created_at' },
                        'latestTimestamp': { '$max': '$created_at' }
                    }
                },
                {
                    '$group': {
                        '_id': '$_id.channel',
                        'statuses': {
                            '$push': {
                                'state': '$_id.state',
                                'operation': '$_id.operation',
                                'created_date': '$_id.created_date',
                                'count': '$stateCount',
                                'totalBoostPriority': '$totalBoostPriority',
                                'earliestTimestamp': '$earliestTimestamp',
                                'latestTimestamp': '$latestTimestamp'
                            }
                        },
                        'uniqueStates': { '$sum': 1 },
                        'totalChannelEntries': { '$sum': '$stateCount' }
                    }
                },
                {
                    '$sort': { 'totalChannelEntries': -1 }
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            st.write(f"Successfully fetched {len(results)} channel data records")
            return results
    
        except Exception as e:
            st.error(f"Error fetching social media data: {e}")
            return []
    def create_channel_state_chart(self, df: pd.DataFrame):
            """
            Create an interactive stacked bar chart
            
            Args:
                df (pd.DataFrame): Transformed dataframe
            
            Returns:
                Plotly figure
            """
            fig = px.bar(
                df, 
                x='Channel', 
                y='Count', 
                color='StateMeaning',
                title='Social Media Channel States Distribution',
                labels={'Count': 'Number of Entries', 'StateMeaning': 'State Description'},
                color_discrete_map={
                    'Initial/Pending': 'lightblue', 
                    'Processing': 'orange', 
                    'Completed': 'green'
                }
            )
            
            return fig
    def create_pie_chart(self, df: pd.DataFrame):
        """
        Create a pie chart showing state distribution
        
        Args:
            df (pd.DataFrame): Transformed dataframe
        
        Returns:
            Plotly figure
        """
        state_totals = df.groupby('StateMeaning')['Count'].sum().reset_index()
        fig = px.pie(
            state_totals, 
            values='Count', 
            names='StateMeaning', 
            title='Overall State Distribution',
            color_discrete_map={
                'Initial/Pending': 'lightblue', 
                'Successful': 'green',
                "failed": 'red'
            }
        )
        return fig
    def transform_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Transform raw MongoDB results into a pandas DataFrame
        
        Args:
            raw_data (List[Dict]): Raw aggregated data
        
        Returns:
            pd.DataFrame: Transformed and clean dataframe
        """
        try:
            transformed_data = []
            for channel_data in raw_data:
                channel = channel_data['_id']
                for status in channel_data['statuses']:
                    transformed_data.append({
                        'Channel': channel,
                        'State': status['state'],
                        'Operation': status.get('operation', 'Unknown'),
                        'CreatedDate': status.get('created_date', 'Unknown'),
                        'Count': status['count'],
                        'BoostPriority': status.get('totalBoostPriority', 0),
                        'EarliestTimestamp': status.get('earliestTimestamp'),
                        'LatestTimestamp': status.get('latestTimestamp')
                    })
            
            df = pd.DataFrame(transformed_data)
            
            # Convert date columns to datetime
            df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])
            
            # Add State Meaning
            state_meanings = {
                0: 'Initial/Pending',
                1: 'Successful',
                2: 'Processing',
                3: 'Failed',
            }
            df['StateMeaning'] = df['State'].map(state_meanings)
            
            return df
        
        except Exception as e:
            st.error(f"Data transformation error: {e}")
            return pd.DataFrame()
    def create_boost_priority_chart(self, df: pd.DataFrame):
        """
        Create a bar chart showing boost priority across channels
        
        Args:
            df (pd.DataFrame): Transformed dataframe
        
        Returns:
            Plotly figure
        """
        boost_data = df.groupby('Channel')['BoostPriority'].sum().reset_index()
        fig = px.bar(
            boost_data, 
            x='Channel', 
            y='BoostPriority',
            title='Boost Priority by Social Media Channel',
            labels={'BoostPriority': 'Total Boost Priority', 'Channel': 'Social Media Channel'},
            color='Channel',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        return fig
    def create_daily_trend_chart(self, df: pd.DataFrame):
        """
        Create a line chart showing daily trends of social media entries
        
        Args:
            df (pd.DataFrame): Transformed dataframe
        
        Returns:
            Plotly figure
        """
        # Group by date and state, sum the counts
        daily_trends = df.groupby([df['CreatedDate'].dt.date, 'StateMeaning'])['Count'].sum().reset_index()
        
        fig = px.line(
            daily_trends, 
            x='CreatedDate', 
            y='Count', 
            color='StateMeaning',
            title='Daily Social Media Entry Trends',
            labels={'CreatedDate': 'Date', 'Count': 'Number of Entries'},
            color_discrete_map={
                'Initial/Pending': 'lightblue', 
                'Processing': 'orange', 
                'Successful': 'green',
                'Failed': 'red'
            }
        )
        
        return fig
    def create_operation_distribution_chart(self, df: pd.DataFrame):
        """
        Create a stacked bar chart showing operation distribution
        
        Args:
            df (pd.DataFrame): Transformed dataframe
        
        Returns:
            Plotly figure
        """
        operation_data = df.groupby(['Channel', 'Operation'])['Count'].sum().reset_index()
        fig = px.bar(
            operation_data, 
            x='Channel', 
            y='Count', 
            color='Operation',
            title='Social Media Channel Operations Distribution',
            labels={'Count': 'Number of Entries', 'Operation': 'Operation Type'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        return fig
    def create_channel_daily_performance(self, df: pd.DataFrame):
        """
        Create a heatmap showing channel performance by date
        
        Args:
            df (pd.DataFrame): Transformed dataframe
        
        Returns:
            Plotly figure
        """
        # Pivot the data for heatmap
        daily_channel_performance = df.pivot_table(
            index='CreatedDate', 
            columns='Channel', 
            values='Count', 
            aggfunc='sum'
        ).fillna(0)
        
        # Create heatmap
        fig = px.imshow(
            daily_channel_performance, 
            labels=dict(x="Channel", y="Date", color="Number of Entries"),
            title="Channel Performance Heatmap",
            aspect="auto"
        )
        
        return fig

def main():
    # Set page configuration
    st.set_page_config(
        page_title="🌐 Advanced Social Media Analytics Dashboard",
        page_icon=":bar_chart:",
        layout="wide"
    )

    # Access secrets
    mongo_uri = st.secrets["mongo"]["uri"]
    database = st.secrets["mongo"]["database"]
    collection = st.secrets["mongo"]["collection"]

    # Initialize Analytics
    analytics = SocialMediaAnalytics(
        mongo_uri=mongo_uri,
        database=database,
        collection=collection
    )
    
    # Title and Description
    st.title("🌐 Advanced Social Media Channel State Analytics")
    st.markdown("Comprehensive insights into social media channel performance and engagement.")
    
    # Fetch and Transform Data
    raw_data = analytics.fetch_social_media_data()
    
    # Check if data is available
    if not raw_data:
        st.error("No data available. Please check your database connection.")
        return
    
    # Transform data
    df = analytics.transform_data(raw_data)
    # Date-wise Insights Section
    st.markdown("### 🕒 Date-wise Insights")
    
    # Date range selection
    min_date = df['CreatedDate'].min().date()
    max_date = df['CreatedDate'].max().date()
    
    # Allow date range selection
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on selected date range
    filtered_df = df[
        (df['CreatedDate'].dt.date >= date_range[0]) & 
        (df['CreatedDate'].dt.date <= date_range[1])
    ]
    
    # Compute date-wise metrics
    total_entries = filtered_df['Count'].sum()
    successful_entries = filtered_df[filtered_df['StateMeaning'] == 'Successful']['Count'].sum()
    success_rate = (successful_entries / total_entries) * 100 if total_entries > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Entries", f"{total_entries:,}")
    
    with col2:
        st.metric("Successful Entries", f"{successful_entries:,}")
    
    with col3:
        st.metric("Success Rate", f"{success_rate:.2f}%")
    
    # Daily breakdown
    st.subheader("📆 Daily Breakdown")
    daily_breakdown = filtered_df.groupby(
        [filtered_df['CreatedDate'].dt.date, 'StateMeaning']
    )['Count'].sum().unstack(fill_value=0)
    
    st.dataframe(daily_breakdown, use_container_width=True)
    # Create Rows of Visualizations
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.plotly_chart(
            analytics.create_channel_state_chart(df), 
            use_container_width=True
        )
    
    with row1_col2:
        st.plotly_chart(
            analytics.create_pie_chart(df), 
            use_container_width=True
        )
    
    # Second Row
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.plotly_chart(
            analytics.create_boost_priority_chart(df), 
            use_container_width=True
        )
    
    with row2_col2:
        st.plotly_chart(
            analytics.create_operation_distribution_chart(df), 
            use_container_width=True
        )
    
    # Date-wise Insights Row
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        st.plotly_chart(
            analytics.create_daily_trend_chart(df), 
            use_container_width=True
        )
    
    with row3_col2:
        st.plotly_chart(
            analytics.create_channel_daily_performance(df), 
            use_container_width=True
        )
    
    # Detailed Data Table
    st.subheader("📊 Detailed Social Media Channel Data")
    st.dataframe(df, use_container_width=True)

    

if __name__ == "__main__":
    main()
