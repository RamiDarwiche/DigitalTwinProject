import os, csv, time, random
import paho.mqtt.client as mqtt
import pandas as pd
from pathlib import Path


datafile_path = Path('PyTorchAnomalyDD.csv')
datasets_root = Path('working')
raw_dt = pd.read_csv(datafile_path, nrows = 28800)
raw_dt = raw_dt[raw_dt['Occupancy Mode Indicator'] > 0]

'''
Create new blank csv and wipe the old
blank csv
'''

from paho.mqtt import client as mqtt_client


broker = 'broker.emqx.io'
port = 1883
topic = "test"
# Generate a Client ID with the publish prefix.
client_id = f'publish-{random.randint(0, 1000)}'
username = 'unreal'
password = 'unreal'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client, csv_path="PyTorchAnomalyDD.csv"):
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for (row, row2) in zip(raw_dt['AHU: Outdoor Air Temperature'], raw_dt['Datetime']):
            temp_data_val = "Time: " + row2 + "\nTemperature: " + str(row)
            time.sleep(1)
            msg = f"messages: {line_count}"
            line_count += 1
            result = client.publish(topic, temp_data_val)
            status = result[0]
            '''
            Within loop, add new publish to blank CSV
            to build 'real time' data, after x amount of time (a day)
            close csv so that it can be read by anomaly detection
            (look into simultaneous read/write), after function has
            been given enough time to run, continue inserting
            new entries
            '''
            if status == 0:
                print(f"Send `{temp_data_val}` to topic `{topic}`")
            else:
                print(f"Failed to send message to topic {topic}")

def run():
    client = connect_mqtt()
    client.loop_start()
    publish(client)
    client.loop_stop()


if __name__ == '__main__':
    run()
