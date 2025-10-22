import math

import numpy as np
import pandas as pd
import shapely as shp
import geopandas as gpd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase


def open_processed_gtfs(filepath) -> gpd.GeoDataFrame:
    gtfs_stations = gpd.read_file(filepath)

    gtfs_stations['connections'] = [set_string.replace(' ', '') for set_string in gtfs_stations['connections'].values]
    gtfs_stations['connections'] = [set_string.replace('[', '') for set_string in gtfs_stations['connections'].values]
    gtfs_stations['connections'] = [set_string.replace(']', '') for set_string in gtfs_stations['connections'].values]
    gtfs_stations['connections'] = [set_string.split(',') for set_string in gtfs_stations['connections'].values]

    total = []
    for value in gtfs_stations['connections'].values:
        connections = []
        for index, val in enumerate(value):
            if val[1:-1] == "None":
                value[index] = 'None'
            else:
                value[index] = int(val)
        for i in range(len(value) // 2):
            connections.append([value[i*2], value[i*2 + 1]])
        total.append(connections)
    gtfs_stations['connections'] = total

    return gtfs_stations


def open_wmata_station_data(filepath) -> gpd.geodataframe:
    wmata_stations = pd.read_csv(filepath, sep=',', usecols=['NAME', 'X_COORD', 'Y_COORD', 'DISPLAY_NAME'])
    wmata_stations = gpd.GeoDataFrame(wmata_stations, geometry=gpd.points_from_xy(wmata_stations['X_COORD'], wmata_stations['Y_COORD']))
    wmata_stations['STATION_ID'] = [name.split('(')[-1][:-1] for name in wmata_stations['DISPLAY_NAME']]
    wmata_stations = wmata_stations.drop(columns=['X_COORD', 'Y_COORD', 'DISPLAY_NAME'])
    wmata_stations = wmata_stations.set_crs('EPSG:4326')
    
    return wmata_stations


def open_stations(filepath) -> gpd.GeoDataFrame:
    gtfs_stations = gpd.read_file(filepath)
    gtfs_stations = gtfs_stations.set_crs('EPSG:4326')
    gtfs_stations = gtfs_stations.to_crs('EPSG:2248')

    # Convert from-to set strings to lists
    for column in ['connections']:
        gtfs_stations[column] = [set_string.replace(' ', '') for set_string in gtfs_stations[column].values]
        gtfs_stations[column] = [set_string.replace('[', '') for set_string in gtfs_stations[column].values]
        gtfs_stations[column] = [set_string.replace(']', '') for set_string in gtfs_stations[column].values]
        gtfs_stations[column] = [set_string.split(',') for set_string in gtfs_stations[column].values]

        total = []
        for value in gtfs_stations[column].values:
            connections = []
            for i in range(len(value) // 2):
                connections.append([value[i*2][1:-1], value[i*2 + 1][1:-1]])
            total.append(connections)
        gtfs_stations[column] = total

    for index, station in gtfs_stations.iterrows():
        connections = station['connections']
        connections = [[station.replace('\\', '') for station in connection] for connection in connections]
        gtfs_stations.at[index, 'connections'] = connections

    return gtfs_stations


def open_delays(filepath) -> pd.DataFrame:
    delays = pd.read_csv(filepath)
    if 'arrival_time' in delays.columns:
        delays['arrival_time'] = pd.to_datetime(delays['arrival_time'])
    if 'real_arrival_time' in delays.columns:
        delays['real_arrival_time'] = pd.to_datetime(delays['real_arrival_time'])
    if 'sched_arrival_time' in delays.columns:
        delays['sched_arrival_time'] = pd.to_datetime(delays['sched_arrival_time'])

    print(f"Amount of delay values: {len(delays):_}")
    return delays

def create_timesteps(timestep_size, timeperiod_start, timeperiod_end) -> list[tuple[int, int]]:
    timesteps = []
    current_step = (timeperiod_start, 0)
    while current_step[0] < timeperiod_end:
        timesteps.append(current_step)
        next_step = (current_step[0] + (current_step[1] + timestep_size) // 60,
                    (current_step[1] + timestep_size) % 60)
        current_step = next_step
    timesteps.append(current_step)

    return timesteps


def calculate_initial_compass_bearing(pointA, pointB):
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def determine_pie_chart_sections(gtfs_stations) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Return total shares dict and starting angles dict of map station pie charts"""
    total_shares = {}
    starting_angles = {}
    geodesic_stations = gtfs_stations.to_crs('EPSG:4326')
    for _, station in geodesic_stations.iterrows():
        station_coords = (station['geometry'].y, station['geometry'].x)
        connections = station['connections']

        bearings = {}
        for connection in [conn for conn in connections if conn[1] != 'None']:
            other_coords = geodesic_stations[geodesic_stations['stop_id'] == connection[1]].geometry.values[0]
            other_coords = (other_coords.y, other_coords.x)
            bearings[connection[1]] = calculate_initial_compass_bearing(station_coords, other_coords)
        directions_sorted = [direction for _, direction in sorted(zip(bearings.values(), bearings.keys()))]
        bearings_sorted = sorted(bearings.values())

        border_bearings = {}
        for i in range(len(directions_sorted)):
            if i == 0:
                first_bearing = (bearings_sorted[-1] + ((360 - bearings_sorted[-1] + bearings_sorted[0]) / 2)) % 360
                start_bearing = first_bearing
            else:
                start_bearing = (bearings_sorted[i - 1] + bearings_sorted[i]) / 2
            if i == (len(directions_sorted) - 1):
                end_bearing = first_bearing
            else:
                end_bearing = (bearings_sorted[i] + bearings_sorted[i + 1]) / 2
            border_bearings[directions_sorted[i]] = (start_bearing, end_bearing)
        starting_angles[station['stop_id']] = first_bearing

        shares = {}
        for direction, border_bearings_i in border_bearings.items():
            if border_bearings_i[1] < border_bearings_i[0]:
                share = (360 - border_bearings_i[0] + border_bearings_i[1]) / 360
            else:
                share = (border_bearings_i[1] - border_bearings_i[0]) / 360
            if share == 0:
                share = 1.0
            shares[direction] = round(share, 2)
        total_shares[station['stop_id']] = shares
    
    return total_shares, starting_angles


def draw_pie(dist, start_angle, colors, xpos, ypos, size, ax=None):
    """A function to plot pie charts"""
    """https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()
    start_shift = (start_angle * (np.pi/180))

    for r1, r2, color in zip(pie[:-1], pie[1:], colors):
        angles = np.linspace((2 * np.pi * r1) + start_shift, (2 * np.pi * r2) + start_shift)
        
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()
        xy = np.column_stack([y, x])


        ax.scatter([xpos], [ypos], marker=xy, s=size, color=color)

    return ax


def plot_map_view(gtfs_stations: gpd.GeoDataFrame, links: gpd.GeoDataFrame,
                  link_values: dict[tuple[str, str], float]=None,
                  station_values: dict[str, dict[str, float]]=None,
                  originator_connection: tuple[str, str]=None,
                  cmap_bounds: tuple[int, int, int]=[-60, 0, 60],
                  non_pie_station_size: tuple[int, int]=[4, 5],
                  station_names_y_offset: tuple[float, float]=[0.003, 0.0010],
                  link_direction_split_offset: tuple[float, float]=[0.002, 0.0005]
                  ) -> None:
    # Start figure, determine color scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    cmap = LinearSegmentedColormap.from_list('', ['green', 'limegreen', 'white', 'pink', 'red'])
    norm = TwoSlopeNorm(vmin=cmap_bounds[0], vcenter=cmap_bounds[1], vmax=cmap_bounds[2])

    # Plot station piecharts
    if station_values is not None:
        station_pie_shares, station_pie_start_angles = determine_pie_chart_sections(gtfs_stations)

    geodesic_stations = gtfs_stations.to_crs('EPSG:4326')
    for _, station in geodesic_stations.iterrows():
        if station_values is not None:
            stop_id = station['stop_id']
            starting_angle = station_pie_start_angles[stop_id]
            shares = station_pie_shares[stop_id]
            means = station_values[stop_id]
            means_sorted = {direction_id: means[direction_id] for direction_id in shares.keys()}
            colors = [cmap(norm(mean)) if mean != None else 'gray' for mean in means_sorted.values()]

            if originator_connection is not None:
                colors = [color if (stop_id, direction_id) != originator_connection else 'blue' for color, direction_id in zip(colors, means_sorted.keys())]

            for ax in [ax1, ax2]:
                draw_pie(list(shares.values()),
                        start_angle=starting_angle,
                        colors=colors,
                        xpos=station['geometry'].x, ypos=station['geometry'].y,
                        ax=ax, size=300)

        else:
            ax1.plot(station['geometry'].x, station['geometry'].y, **{'color': 'grey', 'marker': 'o', 'markersize': non_pie_station_size[0]})
            ax2.plot(station['geometry'].x, station['geometry'].y, **{'color': 'grey', 'marker': 'o', 'markersize': non_pie_station_size[1]})

        # Plot station names
        x, y = station.geometry.x, station.geometry.y
        label = station['stop_id']
        ax1.text(x+0.001, y+station_names_y_offset[0], label, zorder=1, size=8, clip_on=True)
        ax2.text(x+0.001, y+station_names_y_offset[1], label, zorder=1, size=8, clip_on=True)

    # Plot links
    geodesic_links = links.copy()
    geodesic_links = geodesic_links.to_crs('EPSG:4326')
    if link_values is not None:
        for ax, offset in zip([ax1, ax2], [link_direction_split_offset[0], link_direction_split_offset[1]]):
            for _, link in geodesic_links.iterrows():
                mean = link_values[(link['from_stop'], link['to_stop'])]
                color = 'gray' if mean is None else cmap(norm(mean))

                link_geom = shp.offset_curve(link['geometry'], -offset)
                x, y = link_geom.xy
                ax.quiver(x[0], y[0], x[-1]-x[0], y[-1]-y[0], color=color, width=0.004, zorder=0, scale_units='xy', angles='xy', scale=1)
    else:
        for ax in [ax1, ax2]:
            for _, link in geodesic_links.iterrows():
                color = 'gray'
                x, y = link['geometry'].xy
                ax.plot(x, y, color=color, linewidth=4, zorder=0)


    # Add colorbar in the middle
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    cbar = ColorbarBase(cax, cmap=cmap, norm=norm, label='Average delay [seconds]')

    # Plot make-up
    for ax_i in [ax1, ax2]:
        ax_i.set_aspect('equal', adjustable='box')
        ax_i.axis('off')

        # ax_i.set_xticks([])
        # ax_i.set_yticks([])
        # ax_i.set_ylabel('Latitude')
        # ax_i.set_xlabel('Longitude')

    ax2.set_ylim(38.8385, 38.9340)
    ax2.set_xlim(-77.0915, -76.9619)

    # fig.suptitle(f"Mean propagation effects on delays at {int(timestep)}:00")
    # ax.set_facecolor('lightgray')
    # ax.axis('off')
    plt.tight_layout()
    fig.patch.set_facecolor('lightgray')
    return fig
