"""
Data visualization utilities for EcoGuard AI
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

def create_impact_charts(waste_history):
    """
    Create visualization charts for environmental impact
    
    Args:
        waste_history: List of classification records
    
    Returns:
        Dictionary of Plotly figures
    """
    charts = {}
    
    if not waste_history:
        return charts
    
    # Waste type distribution
    waste_types = [item['waste_type'] for item in waste_history]
    df_waste = pd.DataFrame({'Type': waste_types}).value_counts().reset_index()
    df_waste.columns = ['Type', 'Count']
    
    charts['pie_chart'] = px.pie(
        df_waste, 
        values='Count', 
        names='Type',
        title='Waste Type Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Impact over time
    dates = [item['timestamp'] for item in waste_history]
    co2_saved = [item['impact']['co2_saved'] for item in waste_history]
    water_saved = [item['impact']['water_saved'] for item in waste_history]
    
    df_timeline = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'CO2_Saved': co2_saved,
        'Water_Saved': water_saved
    }).sort_values('Date')
    
    df_timeline['Cumulative_CO2'] = df_timeline['CO2_Saved'].cumsum()
    df_timeline['Cumulative_Water'] = df_timeline['Water_Saved'].cumsum()
    
    # CO2 timeline chart
    charts['co2_timeline'] = px.area(
        df_timeline,
        x='Date',
        y='Cumulative_CO2',
        title='Cumulative CO2 Savings Over Time',
        labels={'Cumulative_CO2': 'CO2 Saved (kg)'}
    )
    
    # Water timeline chart
    charts['water_timeline'] = px.area(
        df_timeline,
        x='Date',
        y='Cumulative_Water',
        title='Cumulative Water Savings Over Time',
        labels={'Cumulative_Water': 'Water Saved (L)'}
    )
    
    # Bar chart for waste types
    type_counts = pd.DataFrame({
        'Waste Type': [item['waste_type'] for item in waste_history]
    }).value_counts().reset_index()
    type_counts.columns = ['Waste Type', 'Count']
    
    charts['bar_chart'] = px.bar(
        type_counts,
        x='Waste Type',
        y='Count',
        title='Items by Waste Type',
        color='Waste Type',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    return charts

def create_leaderboard(user_items=0, user_co2_saved=0):
    """
    Create sample community leaderboard
    
    Args:
        user_items: Number of items recycled by the user (default: 0)
        user_co2_saved: Amount of CO2 saved by the user (default: 0)
    
    Returns:
        DataFrame with leaderboard data
    """
    # Sample data for demonstration
    # In production, this would fetch from a database
    sample_users = [
        {'Name': 'EcoWarrior', 'Items Recycled': 245, 'CO2 Saved': 367.5, 'Badge': '🏆'},
        {'Name': 'GreenGuardian', 'Items Recycled': 189, 'CO2 Saved': 283.5, 'Badge': '🥇'},
        {'Name': 'RecyclePro', 'Items Recycled': 156, 'CO2 Saved': 234.0, 'Badge': '🥈'},
        {'Name': 'EarthLover', 'Items Recycled': 134, 'CO2 Saved': 201.0, 'Badge': '🥉'},
        {'Name': 'Sustainability', 'Items Recycled': 98, 'CO2 Saved': 147.0, 'Badge': '🌱'},
        {'Name': 'EcoFriend', 'Items Recycled': 87, 'CO2 Saved': 130.5, 'Badge': '🌿'},
        {'Name': 'GreenTechie', 'Items Recycled': 76, 'CO2 Saved': 114.0, 'Badge': '🍃'},
        {'Name': 'PlanetSaver', 'Items Recycled': 65, 'CO2 Saved': 97.5, 'Badge': '🌍'},
        {'Name': 'WasteWarrior', 'Items Recycled': 54, 'CO2 Saved': 81.0, 'Badge': '♻️'},
        {'Name': 'RecycleRookie', 'Items Recycled': 43, 'CO2 Saved': 64.5, 'Badge': '🌱'},
    ]
    
    # Add the current user
    user_entry = {
        'Name': '⭐ You',
        'Items Recycled': user_items,
        'CO2 Saved': round(user_co2_saved, 2),
        'Badge': '🌟'
    }
    
    # Combine and sort by items recycled
    all_users = sample_users + [user_entry]
    leaderboard_df = pd.DataFrame(all_users)
    leaderboard_df = leaderboard_df.sort_values('Items Recycled', ascending=False)
    leaderboard_df = leaderboard_df.reset_index(drop=True)
    
    # Add rank column
    leaderboard_df.insert(0, 'Rank', range(1, len(leaderboard_df) + 1))
    
    return leaderboard_df

def create_weekly_summary(waste_history):
    """
    Create a weekly summary of recycling activities
    
    Args:
        waste_history: List of classification records
    
    Returns:
        DataFrame with weekly summary
    """
    if not waste_history:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(waste_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by date and waste type
    df['date'] = df['timestamp'].dt.date
    daily_summary = df.groupby(['date', 'waste_type']).size().reset_index(name='count')
    
    return daily_summary

def create_impact_comparison_chart(user_impact, average_impact):
    """
    Create a comparison chart of user impact vs average
    
    Args:
        user_impact: Dict with user's impact metrics
        average_impact: Dict with average impact metrics
    
    Returns:
        Plotly figure
    """
    metrics = ['co2_saved', 'water_saved', 'energy_saved']
    labels = ['CO2 Saved (kg)', 'Water Saved (L)', 'Energy Saved (kWh)']
    
    user_values = [user_impact.get(m, 0) for m in metrics]
    avg_values = [average_impact.get(m, 0) for m in metrics]
    
    fig = go.Figure(data=[
        go.Bar(name='Your Impact', x=labels, y=user_values, marker_color='#2E7D32'),
        go.Bar(name='Average User', x=labels, y=avg_values, marker_color='#90A4AE')
    ])
    
    fig.update_layout(
        title='Your Impact vs Average User',
        barmode='group',
        xaxis_title='Impact Metric',
        yaxis_title='Amount Saved',
        showlegend=True
    )
    
    return fig