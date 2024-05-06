## YOLO v8 Multi-Class count and Data Ingestion

There are two examples of using YOLO v8 for counting entry and exits for a particular class, i.e., counting all the instances of a particular type of object or class in an image or video clip as they move in and out of a particular area. An application of this could be for traffic counts, you could count the people walking on the sidewalk, cars, bicycles, trucks, etc., and use that for better city planning.
There are two examples:

* A general example that generates a JSON payload with the counts of those items entering and exiting the area
* A data pipeline example that generates MQTT messages that could be picked up by an MQTT broker and the written to database like InfluxDB. MQTT was chosen because it’s a fairly low power protocol and it allows for two way communication, thus making it quite useful for a solution that could potentially be deployed on a low powered edge device.

### Multi-Class Counter w/ Data Pipeline – Additional Details

The data architecture is fairly simple: Node-RED functions as a sort of conductor, it picks up the messages from Eclipse-Mosquitto and then writes the data to InfluxDB. From there Grafana is used to display the data. 


**Note:** I just used the data collection infrastructure I built for my data platform project which runs on my Kubernetes (K3s distro) cluster. While an edge cluster “could” make sense in certain scenarios if this was being deployed around a city it would probably make more sense to use something like Hive MQ or a solution deployed on the cloud.*


