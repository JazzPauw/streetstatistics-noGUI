from flask import Flask, render_template
import json
import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
# Neccesary imports

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
overall_stats_path = os.path.join(BASE_DIR, 'Internal', 'overall_stats.json')
log_dir = os.path.join(BASE_DIR, 'logs')

# Define the dynamic paths 
logger = logging.getLogger()


def load_data_as_dataframe():
    try:
        with open(overall_stats_path, 'r') as file:
            data = json.load(file)

        if not data:
            return pd.DataFrame(columns=['timestamp', 'direction', 'class_id'])

        df = pd.DataFrame(data, columns=['timestamp', 'direction', 'class_id'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=['timestamp', 'direction', 'class_id'])  # Return empty DataFrame on error

def calculate_text_stats(df):
    if df.empty:
        return {
            'total_direction_a': 0,
            'total_direction_b': 0,
            'total_targets': 0,
            'total_humans': 0,
            'total_vehicles': 0
        }

    total_direction_a = df[df['direction'] == 1].shape[0]
    total_direction_b = df[df['direction'] == 2].shape[0]

    total_targets = total_direction_a + total_direction_b

    total_humans = df[df['class_id'] == '0'].shape[0]
    total_vehicles = df[df['class_id'].isin(['2', '3', '5', '7'])].shape[0]

    return {
        'total_direction_a': total_direction_a,
        'total_direction_b': total_direction_b,
        'total_targets': total_targets,
        'total_humans': total_humans,
        'total_vehicles': total_vehicles
    }

def generate_hourly_activity_plot(df):
    if df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot([], [], marker='o', color='b', linestyle='-')
        plt.title('Average Hourly Activity (No Data)')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Average Number of Target Counts')
        plt.grid(True)
    else:
        df['hour'] = df['timestamp'].dt.hour
        
        hourly_counts = df.groupby('hour').size()
        
        hourly_counts = hourly_counts.reindex(range(24), fill_value=0)

        plt.figure(figsize=(10, 6))
        plt.plot(hourly_counts.index, hourly_counts.values, marker='o', color='b', linestyle='-')
        plt.title('Average Hourly Activity')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Average Number of Target Counts')
        plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

def generate_cumulative_count_plot(df):
    if df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot([], [], marker='o', color='b', linestyle='-')
        plt.title('Cumulative Target Count (No Data)')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Count')
        plt.grid(True)
    else:
        df = df.sort_values('timestamp')
        df['cumulative_count'] = range(1, len(df) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamp'], df['cumulative_count'], marker='o', color='g', linestyle='-')
        plt.title('Cumulative Target Count Over Time')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Count')
        plt.xticks(rotation=45)
        plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url


def generate_class_distribution_plot(df):
    if df.empty:
        labels = ['Humans', 'Vehicles']
        sizes = [0, 0]
    else:
        human_count = df[df['class_id'] == '0'].shape[0]
        vehicle_count = df[df['class_id'].isin(['2', '3', '5', '7'])].shape[0]
        labels = ['Humans', 'Vehicles']
        sizes = [human_count, vehicle_count]

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
    plt.axis('equal')
    plt.gca().add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor="black", lw=2))  # Box around pie chart

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

def generate_direction_plot(df):
    if df.empty:
        labels = ['Direction 1', 'Direction 2']
        sizes = [0, 0]
    else:
        dir_1_count = df[df['direction'] == 1].shape[0]
        dir_2_count = df[df['direction'] == 2].shape[0]
        labels = ['Direction 1', 'Direction 2']
        sizes = [dir_1_count, dir_2_count]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes, color=['dodgerblue', 'orange'])
    plt.title('Activity by Direction')
    plt.ylabel('Number of Detections')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

@app.route('/')
def index():
    try:    
        df = load_data_as_dataframe()

        hourly_plot_url = generate_hourly_activity_plot(df)
        class_plot_url = generate_class_distribution_plot(df)
        direction_plot_url = generate_direction_plot(df)
        cumulative_count_plot_url = generate_cumulative_count_plot(df)  
        text_stats = calculate_text_stats(df)

        return render_template('index.html', 
                            hourly_plot_url=hourly_plot_url, 
                            class_plot_url=class_plot_url,
                            direction_plot_url=direction_plot_url,
                            cumulative_count_plot_url =cumulative_count_plot_url,
                            df=df.to_html(classes='table table-striped', index=False),
                            text_stats=text_stats) 
    except Exception as e:
        logging.error("An error occured: %s", e, exc_info=True)        

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
