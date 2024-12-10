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
import io

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
    def create_state_color_map(self):
            """
            Create a color mapping for different states
            
            Returns:
                dict: Color mapping for states
            """
            return {
                0: 'red',    # Failed
                1: 'green',  # Successful
                2: 'red',    # Failed
                3: 'red'     # Failed
            }

    def render_social_media_card(self, record):
        """
        Render a card for a social media record
        
        Args:
            record (dict): Social media record
        
        Returns:
            str: HTML for the card
        """
        # Determine state color
        state_colors = self.create_state_color_map()
        state_color = state_colors.get(record.get('state', 0), 'gray')
        
        # Extract relevant information with proper fallbacks
        object_id = str(record.get('_id', 'N/A'))
        channel = record.get('channel', 'N/A')
        state = record.get('state', 'N/A')
        operation = record.get('operation', 'N/A')
        
        # Social data extraction
        social_data = record.get('social_data', {})
        message = social_data.get('message', 'No message')
        picture = social_data.get('picture', '')
        link = social_data.get('link', '')
        
        # Truncate message if too long
        truncated_message = message[:200] + '...' if len(message) > 200 else message
        
        # Create card HTML
        # Create card HTML
        card_html = f'''
        <div style="border: 2px solid {state_color}; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0; color: {state_color};">Record ID: {object_id}</h3>
                <a href="{link}" target="_blank" style="color: blue;"><span style="background-color: {state_color}; color: white; padding: 5px 10px; border-radius: 5px;">
                    {channel.upper()} | {operation.upper()}
                </span>
            </div>
        '''
        
        # Example usage of truncated message
        truncated_message = message[:250] + ('...' if len(message) > 250 else '')

        return card_html

    def render_paginated_cards(self, page=1, records_per_page=5):
        """
        Render paginated cards for both successful and failed states.
        
        Args:
            page (int): Current page number.
            records_per_page (int): Number of records to display per type.
        
        Returns:
            dict: A dictionary with cards and metadata for pagination.
        """
        db = self.get_mongo_connection()
        if db is None:
            st.error("Database connection failed")
            return {"cards": [], "total_records": 0}
        
        try:
            collection = db[self.collection]
            
            # Query for successful and failed records
            successful_query = {'state': 1}
            failed_query = {'state': {'$in': [0, 2, 3]}}
            
            # Sort and pagination
            sort = [('created_at', -1)]
            skip = (page - 1) * records_per_page
            limit = records_per_page
            
            # Fetch records
            successful_records = list(collection.find(successful_query).sort(sort).skip(skip).limit(limit))
            failed_records = list(collection.find(failed_query).sort(sort).skip(skip).limit(limit))
            
            # Combine records
            records = successful_records + failed_records
            total_records = collection.count_documents(successful_query) + collection.count_documents(failed_query)
            
            # Render cards
            cards = [self.render_social_media_card(record) for record in records]
            
            return {"cards": cards, "total_records": total_records}
        
        except Exception as e:
            st.error(f"Error rendering paginated cards: {e}")
            return {"cards": [], "total_records": 0}


    def render_records_as_cards(self, df, page=1, records_per_page=5):
        """
        Render social media records as cards with pagination.
        
        Args:
            df (pd.DataFrame): DataFrame containing filters.
            page (int): Current page number.
            records_per_page (int): Number of records to display per page.
        
        Returns:
            dict: A dictionary with cards and metadata for pagination.
        """
        # Fetch MongoDB connection
        db = self.get_mongo_connection()
        if db is None:
            st.error("Database connection failed")
            return {"cards": [], "total_records": 0}
        
        try:
            collection = db[self.collection]
            
            # Build query for successful and failed states
            query = {}
            if not df.empty:
                # If specific channels are selected
                if 'Channel' in df.columns and len(df['Channel'].unique()) > 0:
                    query['channel'] = {'$in': df['Channel'].unique().tolist()}
                
                # If specific states are selected
                if 'StateMeaning' in df.columns:
                    state_map = {'Successful': 1, 'Failed': [0, 2, 3]}
                    query['state'] = {'$in': state_map['Successful'] if 'Successful' in df['StateMeaning'].unique() 
                                    else state_map['Failed']}
            
            # Sort by `created_at` descending
            sort = [('created_at', -1)]
            
            # Calculate pagination details
            skip = (page - 1) * records_per_page
            limit = records_per_page
            
            # Fetch records with the query
            records = list(collection.find(query).sort(sort).skip(skip).limit(limit))
            total_records = collection.count_documents(query)
            
            # Render cards
            cards = [self.render_social_media_card(record) for record in records]
            
            return {"cards": cards, "total_records": total_records}
        
        except Exception as e:
            st.error(f"Error rendering records as cards: {e}")
            return {"cards": [], "total_records": 0}


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
                    'Failed': 'red',
                    'Completed': 'green',
                    'Successful': 'green',
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
                0: 'Failed',
                1: 'Successful',
                2: 'Failed',
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
    def fetch_top_customers(self) -> pd.DataFrame:
        """
        Fetch top 10 customers who have posted the most along with their favorite channel.

        Returns:
            pd.DataFrame: DataFrame containing top customers and their favorite channels.
        """
        # Establish MongoDB connection
        db = self.get_mongo_connection()
        if db is None:
            st.error("Database connection failed.")
            return pd.DataFrame()

        try:
            collection = db[self.collection]

            # Aggregation pipeline to find top customers
            pipeline = [
                # Step 1: Group by `nowfloats_id` and `channel` to count posts per channel
                {
                    '$group': {
                        '_id': {
                            'nowfloats_id': '$nowfloats_id',
                            'channel': '$channel'
                        },
                        'postCount': {'$sum': 1}
                    }
                },
                # Step 2: Group by `nowfloats_id` to find the total posts and all channel data
                {
                    '$group': {
                        '_id': '$_id.nowfloats_id',
                        'totalPosts': {'$sum': '$postCount'},
                        'channels': {
                            '$push': {
                                'channel': '$_id.channel',
                                'postCount': '$postCount'
                            }
                        }
                    }
                },
                # Step 3: Sort channels by post count in descending order
                {
                    '$addFields': {
                        'favoriteChannel': {
                            '$arrayElemAt': [
                                {
                                    '$sortArray': {
                                        'input': '$channels',
                                        'sortBy': {'postCount': -1}
                                    }
                                },
                                0
                            ]
                        }
                    }
                },
                # Step 4: Sort customers by total posts in descending order
                {
                    '$sort': {'totalPosts': -1}
                },
                # Step 5: Limit to the top 10 customers
                {
                    '$limit': 10
                }
            ]

            # Execute aggregation pipeline
            results = list(collection.aggregate(pipeline))

            # Transform the results into a pandas DataFrame
            data = [
                {
                    'nowfloats_id': item['_id'],
                    'TotalPosts': item['totalPosts'],
                    'FavoriteChannel': item['favoriteChannel']['channel'],
                    'ChannelPostCount': item['favoriteChannel']['postCount']
                }
                for item in results
            ]

            return pd.DataFrame(data)

        except Exception as e:
            st.error(f"Error fetching top customers: {e}")
            return pd.DataFrame()

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
    
    # Fetch initial data (ensure df is defined)
    raw_data = analytics.fetch_social_media_data()
    if not raw_data:  # Check if data fetching failed
        st.error("No data available. Please check the database connection or data source.")
        return

    df = analytics.transform_data(raw_data)
    if df.empty:  # Check if the transformation resulted in an empty DataFrame
        st.error("Data transformation failed. Please check the raw data.")
        return

    if df.empty:
        st.error("No data available. Please check the database connection or data source.")
        return  # Exit the function if no data is available

    # Title and Description
    st.title("🌐 Advanced Social Media Channel State Analytics")
    



    # Add sidebar with interactive options
    st.sidebar.header("🔧 Filters and Options")
    selected_channel = st.sidebar.selectbox("Select Channel", ["All"] + list(df["Channel"].unique()))
    selected_operation = st.sidebar.selectbox("Select Operation", ["All"] + list(df["Operation"].unique()))
    selected_state = st.sidebar.selectbox("Select State", ["All"] + list(df["StateMeaning"].unique()))
    st.sidebar.markdown("---")
    filtered_df = df.copy()  # Create a copy to avoid modifying the original data

    if selected_channel != "All":
        filtered_df = filtered_df[filtered_df["Channel"] == selected_channel]
    if selected_operation != "All":
        filtered_df = filtered_df[filtered_df["Operation"] == selected_operation]
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["StateMeaning"] == selected_state]
    min_date = filtered_df["CreatedDate"].min().date()
    max_date = filtered_df["CreatedDate"].max().date()
    selected_date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date)

    # Filter data based on the selected date range
    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        filtered_df = filtered_df[
            (filtered_df["CreatedDate"] >= pd.Timestamp(start_date)) &
            (filtered_df["CreatedDate"] <= pd.Timestamp(end_date))
        ]
    df = filtered_df.copy()  # Update the main dataframe with filtered data
    # Real-time Metrics
    st.markdown("### 📊 Real-time Metrics")
    st.markdown(
        """
        <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
            <div style="text-align: center;">
                <h3 style="font-size: 2rem;">{:,}</h3>
                <p style="font-size: 1rem;">Total Entries</p>
            </div>
            <div style="text-align: center;">
                <h3 style="font-size: 2rem;">{:,}</h3>
                <p style="font-size: 1rem;">Unique Channels</p>
            </div>
            <div style="text-align: center;">
                <h3 style="font-size: 2rem;">{:.2f}%</h3>
                <p style="font-size: 1rem;">Success Rate</p>
            </div>
        </div>
        """.format(
            df["Count"].sum(),
            df["Channel"].nunique(),
            (df[df["StateMeaning"] == "Successful"]["Count"].sum() / df["Count"].sum() * 100) if not df.empty else 0
        ),
        unsafe_allow_html=True
    )
    
    # Help Button
    if st.sidebar.button("❓ Help/Guide"):
        st.sidebar.info("""
        - Use filters to customize the data view.
        - Hover over charts for detailed insights.
        - Download data using the export options below.
        """)

    # Data Table and Export Options
    st.markdown("### 📊 Detailed Social Media Channel Data")
    st.dataframe(filtered_df, use_container_width=True)

    st.download_button(
        label="Download Data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name="social_media_analytics.csv",
        mime="text/csv"
    )

    # Visualization Options
    st.markdown("### 📈 Visualizations")
    st.plotly_chart(analytics.create_channel_state_chart(filtered_df), use_container_width=True)
    st.plotly_chart(analytics.create_pie_chart(filtered_df), use_container_width=True)
    st.plotly_chart(analytics.create_daily_trend_chart(filtered_df), use_container_width=True)
    st.plotly_chart(analytics.create_channel_daily_performance(filtered_df), use_container_width=True)
    st.markdown("### 📈 Visualizations & Records")

    st.markdown("#### 🗂️ Record Cards") # Error rendering records as cards: in needs an array, full error: {'ok': 0.0, 'errmsg': 'in needs an array', 'code': 2, 'codeName': 'BadValue', '$clusterTime': {'clusterTime': Timestamp(1733799069, 1), 'signature': {'hash': b'\x91\x8e\xca\x8d\x82\x8b\xaf\xec\x17I\x1fV\xaa\xed1W\xdc\x02\xcf\xfa', 'keyId': 7407672065555169293}}, 'operationTime': Timestamp(1733799069, 1)} No records found matching the current filters.
        
    #     # Render cards
    #     cards = analytics.render_records_as_cards(filtered_df)
        
    #     if cards:
    #         # Use st.markdown to render HTML cards
    #         for card in cards:
    #             st.markdown(card, unsafe_allow_html=True)
    #     else:
    #         st.warning("No records found matching the current filters.")
    # # Top Customers Section

        # Pagination settings
    page = st.session_state.get('page', 1)
    records_per_page = 5

    # Fetch cards and total records
    result = analytics.render_paginated_cards(page=page, records_per_page=records_per_page)

    # Display cards
    if result["cards"]:
        for card in result["cards"]:
            st.markdown(card, unsafe_allow_html=True)

    # Handle pagination buttons
    col1, col2 = st.columns(2)
    with col1:
        if page > 1 and st.button("Previous Page"):
            st.session_state['page'] = page - 1
            st.experimental_rerun()

    with col2:
        if page * records_per_page < result["total_records"] and st.button("Next Page"):
            st.session_state['page'] = page + 1
            st.experimental_rerun()

    st.markdown("### 🏆 Top Customers")
    top_customers_df = analytics.fetch_top_customers()

    if not top_customers_df.empty:
        st.dataframe(top_customers_df)
        st.bar_chart(data=top_customers_df, x="nowfloats_id", y="TotalPosts", use_container_width=True)
    else:
        st.warning("No data available for top customers.")


if __name__ == "__main__":
    main()
