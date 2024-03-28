import uuid
from paho.mqtt import client as mqtt
from logging_utils import logger


class CommUtilities():

    def __init__(self):

        pass

    @staticmethod
    def getClientID():

        clientID = str(uuid.uuid4())

        return clientID

    @staticmethod
    def mqttClient(clientID: str, username: str, pwd: str,
                   host: str, port: int):

        def connectionStatus(client, userdata, flags, code):

            if code == 0:
                logger.info('connected')

            else:
                print(f'connection error: {code} retrying...')
                logger.DEBUG(f'connection error occured, return code: {code}')

        client = mqtt.Client(clientID)
        client.username_pw_set(username=username, password=pwd)
        client.on_connect = connectionStatus

        code = client.connect(host, port)

        # this is so that the client will attempt to reconnect automatically/
        # no need to add reconnect
        # logic.
        client.loop_start()

        return client, code
