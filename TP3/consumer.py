from kafka import KafkaConsumer

consumer = KafkaConsumer('exo1', bootstrap_servers=['nowledgeable.com:9092'])

for message in consumer:
    print(f"Received message: {message.value.decode('utf-8')}")