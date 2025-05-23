
# exploratory_healthcare_geo_eda.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
from sklearn.cluster import KMeans
import contextily as ctx

# Load healthcare facility data
health_df = pd.read_csv("healthcare_locations.csv")  # ['Name', 'Type', 'Latitude', 'Longitude']

# Load regional health metrics
metrics_df = pd.read_csv("regional_health_data.csv")  # ['Region', 'DiseaseRate', 'Income', 'Pollution']

# Load shapefile or GeoJSON for regions
gdf = gpd.read_file("regions_shapefile.shp")  # Must include a 'Region' column

# Merge geospatial and metric data
combined = gdf.merge(metrics_df, on="Region")

# Convert health facilities to GeoDataFrame
health_gdf = gpd.GeoDataFrame(
    health_df,
    geometry=gpd.points_from_xy(health_df.Longitude, health_df.Latitude),
    crs="EPSG:4326"
)

# Plot healthcare facilities
fig, ax = plt.subplots(figsize=(12, 10))
combined.to_crs(epsg=3857).plot(ax=ax, color='lightgrey', edgecolor='black')
health_gdf.to_crs(epsg=3857).plot(ax=ax, color='red', markersize=10, label='Healthcare Facilities')
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
plt.title("Healthcare Facilities on Regional Map")
plt.legend()
plt.show()

# Choropleth map of disease rates
combined.plot(column='DiseaseRate', cmap='Reds', legend=True, figsize=(12, 10), edgecolor='black')
plt.title("Disease Rate by Region")
plt.show()

# Pairplot and correlation matrix
sns.pairplot(combined[['DiseaseRate', 'Income', 'Pollution']])
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

corr = combined[['DiseaseRate', 'Income', 'Pollution']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Interactive map with Folium
map_center = [health_df.Latitude.mean(), health_df.Longitude.mean()]
m = folium.Map(location=map_center, zoom_start=6)
for _, row in health_df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Name'],
        icon=folium.Icon(color='red', icon='plus-sign')
    ).add_to(m)
m.save("interactive_health_map.html")

# KMeans clustering
features = combined[['DiseaseRate', 'Income', 'Pollution']].dropna()
kmeans = KMeans(n_clusters=4, random_state=0).fit(features)
combined['Cluster'] = kmeans.labels_

# Visualize clusters
combined.plot(column='Cluster', cmap='Set2', legend=True, figsize=(12, 10), edgecolor='black')
plt.title("Clustered Regions Based on Health Indicators")
plt.show()

# Count healthcare facilities per region
facilities_per_region = gpd.sjoin(health_gdf, combined, how="inner", predicate="within")
counts = facilities_per_region.groupby("Region").size().reset_index(name="FacilityCount")
combined = combined.merge(counts, on="Region", how="left").fillna(0)

# Visualize facilities per region
combined.plot(column='FacilityCount', cmap='Blues', legend=True, figsize=(12, 10), edgecolor='black')
plt.title("Number of Healthcare Facilities per Region")
plt.show()
