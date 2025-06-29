"""
===============================================================================
INTERACTIVE VISUALIZATIONS TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Advanced interactive visualizations using Plotly and other tools

This template covers:
- Interactive scatter plots and line charts
- 3D visualizations
- Animated plots
- Interactive dashboards
- Geographic visualizations
- Network graphs

Prerequisites:
- pandas, numpy, matplotlib, seaborn, plotly
- Dataset loaded as 'df'
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import warnings
warnings.filterwarnings('ignore')

# Set plotting preferences
plt.style.use('default')
pyo.init_notebook_mode(connected=True)

# ===============================================================================
# LOAD AND PREPARE DATA
# ===============================================================================

# Load your dataset
# df = pd.read_csv('your_data.csv')

# Sample dataset creation for demonstration
np.random.seed(42)
n_samples = 500

# Create comprehensive synthetic dataset
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
    'category': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Education', 'Retail'], n_samples),
    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples),
    'value': np.random.exponential(100, n_samples),
    'score': np.random.normal(75, 15, n_samples),
    'count': np.random.poisson(50, n_samples),
    'rating': np.random.uniform(1, 5, n_samples),
    'latitude': np.random.uniform(25, 50, n_samples),
    'longitude': np.random.uniform(-125, -65, n_samples),
    'size': np.random.uniform(10, 100, n_samples),
    'profit': np.random.normal(0, 50, n_samples)
})

# Add some correlations
df['revenue'] = df['value'] * df['count'] / 100 + np.random.normal(0, 10, n_samples)
df['growth_rate'] = df['score'] / 100 + np.random.normal(0, 0.1, n_samples)
df['efficiency'] = df['rating'] * 20 + np.random.normal(0, 5, n_samples)

print("Dataset Shape:", df.shape)
print("Data Types:")
print(df.dtypes)
print("\nSample Data:")
print(df.head())

# ===============================================================================
# 1. INTERACTIVE SCATTER PLOTS
# ===============================================================================

print("\n" + "="*60)
print("1. INTERACTIVE SCATTER PLOTS")
print("="*60)

# Basic interactive scatter plot
fig1 = px.scatter(
    df, 
    x='value', 
    y='score',
    color='category',
    size='size',
    hover_data=['revenue', 'rating'],
    title='Interactive Scatter Plot: Value vs Score by Category'
)
fig1.update_layout(width=800, height=600)
fig1.show()

# Scatter plot with animation over time
# Prepare monthly aggregated data for animation
df['year_month'] = df['date'].dt.to_period('M').astype(str)
monthly_data = df.groupby(['year_month', 'category']).agg({
    'value': 'mean',
    'score': 'mean',
    'revenue': 'sum',
    'count': 'sum'
}).reset_index()

fig2 = px.scatter(
    monthly_data, 
    x='value', 
    y='score',
    size='revenue',
    color='category',
    animation_frame='year_month',
    hover_name='category',
    title='Animated Scatter Plot: Evolution Over Time'
)
fig2.update_layout(width=800, height=600)
fig2.show()

# 3D scatter plot
fig3 = px.scatter_3d(
    df, 
    x='value', 
    y='score', 
    z='revenue',
    color='category',
    size='rating',
    hover_data=['region'],
    title='3D Scatter Plot: Value, Score, and Revenue'
)
fig3.update_layout(width=800, height=600)
fig3.show()

print("Interactive scatter plots created with:")
print("- Hover information")
print("- Color coding by category")
print("- Size mapping")
print("- Animation over time")
print("- 3D visualization")

# ===============================================================================
# 2. INTERACTIVE LINE CHARTS AND TIME SERIES
# ===============================================================================

print("\n" + "="*60)
print("2. INTERACTIVE LINE CHARTS AND TIME SERIES")
print("="*60)

# Time series with multiple traces
time_series_data = df.groupby(['date', 'category'])['value'].mean().reset_index()

fig4 = px.line(
    time_series_data, 
    x='date', 
    y='value',
    color='category',
    title='Interactive Time Series: Value by Category Over Time'
)
fig4.update_layout(width=900, height=500)
fig4.show()

# Multiple y-axes plot
fig5 = make_subplots(specs=[[{"secondary_y": True}]])

# Add revenue trace
fig5.add_trace(
    go.Scatter(x=df['date'], y=df['revenue'], name='Revenue'),
    secondary_y=False,
)

# Add score trace on secondary y-axis
fig5.add_trace(
    go.Scatter(x=df['date'], y=df['score'], name='Score'),
    secondary_y=True,
)

fig5.update_yaxes(title_text="Revenue", secondary_y=False)
fig5.update_yaxes(title_text="Score", secondary_y=True)
fig5.update_layout(title_text="Dual Y-Axis: Revenue and Score Over Time")
fig5.show()

# Candlestick-style plot (using daily aggregates)
daily_agg = df.groupby('date').agg({
    'value': ['min', 'max', 'mean'],
    'score': 'mean'
}).reset_index()
daily_agg.columns = ['date', 'value_min', 'value_max', 'value_mean', 'score_mean']

fig6 = go.Figure()

# Add range area
fig6.add_trace(go.Scatter(
    x=daily_agg['date'], 
    y=daily_agg['value_max'],
    fill=None,
    mode='lines',
    line_color='rgba(0,100,80,0)',
    showlegend=False
))

fig6.add_trace(go.Scatter(
    x=daily_agg['date'], 
    y=daily_agg['value_min'],
    fill='tonexty',
    mode='lines',
    line_color='rgba(0,100,80,0)',
    name='Value Range'
))

# Add mean line
fig6.add_trace(go.Scatter(
    x=daily_agg['date'], 
    y=daily_agg['value_mean'],
    mode='lines',
    name='Value Mean',
    line=dict(color='red', width=2)
))

fig6.update_layout(title='Range Plot: Daily Value Min, Max, and Mean')
fig6.show()

print("Interactive time series created with:")
print("- Multiple category traces")
print("- Dual y-axes")
print("- Range/area plots")
print("- Zoom and pan capabilities")

# ===============================================================================
# 3. ADVANCED STATISTICAL VISUALIZATIONS
# ===============================================================================

print("\n" + "="*60)
print("3. ADVANCED STATISTICAL VISUALIZATIONS")
print("="*60)

# Interactive violin plot
fig7 = px.violin(
    df, 
    y='value', 
    x='category',
    color='region',
    box=True,
    title='Interactive Violin Plot: Value Distribution by Category and Region'
)
fig7.show()

# Parallel coordinates plot
fig8 = px.parallel_coordinates(
    df.sample(100),  # Sample for better performance
    color='score',
    dimensions=['value', 'revenue', 'rating', 'efficiency'],
    title='Parallel Coordinates: Multi-dimensional Analysis'
)
fig8.show()

# Interactive correlation heatmap
corr_matrix = df.select_dtypes(include=[np.number]).corr()

fig9 = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    title='Interactive Correlation Heatmap'
)
fig9.show()

# Sunburst chart (hierarchical data)
fig10 = px.sunburst(
    df.groupby(['region', 'category']).size().reset_index(name='count'),
    path=['region', 'category'],
    values='count',
    title='Sunburst Chart: Hierarchical Data (Region > Category)'
)
fig10.show()

print("Advanced statistical visualizations created:")
print("- Interactive violin plots")
print("- Parallel coordinates")
print("- Interactive correlation heatmap")
print("- Sunburst hierarchical chart")

# ===============================================================================
# 4. GEOGRAPHIC VISUALIZATIONS
# ===============================================================================

print("\n" + "="*60)
print("4. GEOGRAPHIC VISUALIZATIONS")
print("="*60)

# Scatter map
fig11 = px.scatter_mapbox(
    df, 
    lat='latitude', 
    lon='longitude',
    color='category',
    size='value',
    hover_data=['revenue', 'score'],
    mapbox_style='open-street-map',
    zoom=3,
    title='Geographic Scatter Map: Data Points by Location'
)
fig11.update_layout(height=600)
fig11.show()

# Density heatmap
fig12 = px.density_mapbox(
    df, 
    lat='latitude', 
    lon='longitude',
    z='value',
    radius=20,
    mapbox_style='open-street-map',
    zoom=3,
    title='Density Heatmap: Value Concentration'
)
fig12.update_layout(height=600)
fig12.show()

# Choropleth map (using random state data)
# Create state-level aggregated data
state_data = pd.DataFrame({
    'state': ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI'],
    'value': np.random.uniform(50, 200, 10),
    'count': np.random.randint(10, 100, 10)
})

fig13 = px.choropleth(
    state_data,
    locations='state',
    locationmode='USA-states',
    color='value',
    hover_data=['count'],
    scope='usa',
    title='Choropleth Map: Value by State'
)
fig13.show()

print("Geographic visualizations created:")
print("- Scatter map with location data")
print("- Density heatmap")
print("- Choropleth map")

# ===============================================================================
# 5. DASHBOARD-STYLE LAYOUTS
# ===============================================================================

print("\n" + "="*60)
print("5. DASHBOARD-STYLE LAYOUTS")
print("="*60)

# Create a comprehensive dashboard
fig14 = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Time Series', 'Distribution', 'Scatter Plot', 
                   'Bar Chart', 'Box Plot', 'Pie Chart'),
    specs=[[{"type": "scatter"}, {"type": "histogram"}],
           [{"type": "scatter"}, {"type": "bar"}],
           [{"type": "box"}, {"type": "pie"}]]
)

# Time series
time_agg = df.groupby('date')['value'].mean()
fig14.add_trace(
    go.Scatter(x=time_agg.index, y=time_agg.values, name='Value'),
    row=1, col=1
)

# Histogram
fig14.add_trace(
    go.Histogram(x=df['score'], name='Score Distribution'),
    row=1, col=2
)

# Scatter plot
fig14.add_trace(
    go.Scatter(x=df['value'], y=df['revenue'], mode='markers', name='Value vs Revenue'),
    row=2, col=1
)

# Bar chart
category_avg = df.groupby('category')['value'].mean()
fig14.add_trace(
    go.Bar(x=category_avg.index, y=category_avg.values, name='Avg Value by Category'),
    row=2, col=2
)

# Box plot
for i, cat in enumerate(df['category'].unique()[:3]):  # Limit for clarity
    cat_data = df[df['category'] == cat]['score']
    fig14.add_trace(
        go.Box(y=cat_data, name=cat),
        row=3, col=1
    )

# Pie chart
category_counts = df['category'].value_counts()
fig14.add_trace(
    go.Pie(labels=category_counts.index, values=category_counts.values, name='Category Distribution'),
    row=3, col=2
)

fig14.update_layout(height=900, title_text="Comprehensive Dashboard")
fig14.show()

print("Dashboard created with:")
print("- Multiple subplot types")
print("- Coordinated layout")
print("- Comprehensive data overview")

# ===============================================================================
# 6. ANIMATION AND INTERACTIVE CONTROLS
# ===============================================================================

print("\n" + "="*60)
print("6. ANIMATION AND INTERACTIVE CONTROLS")
print("="*60)

# Animated bubble chart
bubble_data = df.copy()
bubble_data['week'] = bubble_data['date'].dt.isocalendar().week

weekly_agg = bubble_data.groupby(['week', 'category']).agg({
    'value': 'mean',
    'revenue': 'sum',
    'count': 'sum'
}).reset_index()

fig15 = px.scatter(
    weekly_agg,
    x='value',
    y='revenue',
    size='count',
    color='category',
    animation_frame='week',
    hover_name='category',
    title='Animated Bubble Chart: Weekly Performance',
    range_x=[0, weekly_agg['value'].max() * 1.1],
    range_y=[0, weekly_agg['revenue'].max() * 1.1]
)
fig15.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 300
fig15.show()

# Interactive filters with dropdowns
fig16 = px.scatter(
    df, 
    x='value', 
    y='score',
    color='category',
    size='rating',
    title='Scatter Plot with Dropdown Filter'
)

# Add dropdown for region filtering
buttons = []
for region in ['All'] + list(df['region'].unique()):
    if region == 'All':
        visible_data = [True] * len(df)
    else:
        visible_data = df['region'] == region
    
    buttons.append(dict(
        label=region,
        method='restyle',
        args=[{'visible': visible_data}]
    ))

fig16.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top"
        ),
    ]
)
fig16.show()

# Range slider for time series
time_series = df.groupby('date')['value'].mean().reset_index()

fig17 = px.line(
    time_series, 
    x='date', 
    y='value',
    title='Time Series with Range Slider'
)

fig17.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)
fig17.show()

print("Interactive animations created:")
print("- Animated bubble charts")
print("- Dropdown filters")
print("- Range sliders")
print("- Interactive controls")

# ===============================================================================
# 7. NETWORK AND RELATIONSHIP VISUALIZATIONS
# ===============================================================================

print("\n" + "="*60)
print("7. NETWORK AND RELATIONSHIP VISUALIZATIONS")
print("="*60)

# Create sample network data
network_data = []
categories = df['category'].unique()
regions = df['region'].unique()

# Create connections between categories and regions
for cat in categories:
    for reg in regions:
        weight = len(df[(df['category'] == cat) & (df['region'] == reg)])
        if weight > 0:
            network_data.append({'source': cat, 'target': reg, 'weight': weight})

network_df = pd.DataFrame(network_data)

# Sankey diagram
fig18 = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=list(categories) + list(regions),
        color="blue"
    ),
    link=dict(
        source=[list(categories).index(x) for x in network_df['source']],
        target=[len(categories) + list(regions).index(x) for x in network_df['target']],
        value=network_df['weight']
    )
)])

fig18.update_layout(title_text="Sankey Diagram: Category-Region Relationships", font_size=10)
fig18.show()

# Treemap
treemap_data = df.groupby(['region', 'category']).agg({
    'value': 'sum',
    'count': 'sum'
}).reset_index()

fig19 = px.treemap(
    treemap_data,
    path=[px.Constant("All"), 'region', 'category'],
    values='value',
    color='count',
    title='Treemap: Hierarchical Value Distribution'
)
fig19.show()

print("Network visualizations created:")
print("- Sankey flow diagrams")
print("- Treemap hierarchical views")

# ===============================================================================
# 8. STATISTICAL MODELING VISUALIZATIONS
# ===============================================================================

print("\n" + "="*60)
print("8. STATISTICAL MODELING VISUALIZATIONS")
print("="*60)

# Regression plot with confidence intervals
fig20 = px.scatter(
    df, 
    x='value', 
    y='revenue',
    trendline='ols',
    title='Regression Analysis: Value vs Revenue'
)
fig20.show()

# Residual plot
from sklearn.linear_model import LinearRegression

# Fit simple regression
X = df[['value']].values
y = df['revenue'].values
model = LinearRegression().fit(X, y)
predictions = model.predict(X)
residuals = y - predictions

residual_df = pd.DataFrame({
    'fitted': predictions,
    'residuals': residuals,
    'category': df['category']
})

fig21 = px.scatter(
    residual_df,
    x='fitted',
    y='residuals',
    color='category',
    title='Residual Plot: Model Diagnostics'
)
fig21.add_hline(y=0, line_dash="dash", line_color="red")
fig21.show()

# Distribution comparison
fig22 = px.histogram(
    df, 
    x='score', 
    color='category',
    marginal='box',
    hover_data=df.columns,
    title='Distribution Comparison with Marginal Box Plot'
)
fig22.show()

print("Statistical modeling visualizations:")
print("- Regression with trendlines")
print("- Residual analysis")
print("- Distribution comparisons")

# ===============================================================================
# 9. EXPORT AND SAVE INTERACTIVE PLOTS
# ===============================================================================

print("\n" + "="*60)
print("9. EXPORT AND SAVE INTERACTIVE PLOTS")
print("="*60)

# Save plots as HTML files
fig1.write_html("interactive_scatter.html")
fig4.write_html("time_series.html")
fig14.write_html("dashboard.html")

# Save static images
fig1.write_image("scatter_plot.png", width=800, height=600)
fig14.write_image("dashboard.png", width=1200, height=900)

print("Plots exported as:")
print("- HTML files (interactive)")
print("- PNG files (static)")
print("- Ready for web deployment")

# Summary of created visualizations
print("\n" + "="*60)
print("VISUALIZATION SUMMARY")
print("="*60)

visualizations = [
    "Interactive scatter plots with hover and animation",
    "3D scatter plots",
    "Multi-trace time series",
    "Dual y-axis plots",
    "Range and area plots",
    "Interactive violin plots",
    "Parallel coordinates",
    "Interactive correlation heatmaps",
    "Sunburst hierarchical charts",
    "Geographic scatter maps",
    "Density heatmaps",
    "Choropleth maps",
    "Comprehensive dashboards",
    "Animated bubble charts",
    "Interactive filters and controls",
    "Sankey diagrams",
    "Treemaps",
    "Regression analysis plots",
    "Statistical diagnostic plots"
]

print(f"Created {len(visualizations)} different visualization types:")
for i, viz in enumerate(visualizations, 1):
    print(f"{i:2d}. {viz}")

print(f"\nAll visualizations are interactive and ready for analysis!")
print(f"Use .show() to display plots in Jupyter notebooks")
print(f"Use .write_html() to save as standalone HTML files")
