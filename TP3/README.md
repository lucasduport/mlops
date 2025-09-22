# Unified Kafka Consumer System

This directory has been reorganized to use a unified, configurable consumer system that consolidates all the functionality from the individual consumer files.

## Files Overview

### Original Consumer Files
- `ml_consumer.py` - Basic ML consumer that only consumes house data and makes predictions
- `ml_consumer_producer.py` - ML consumer that also produces predictions to another topic  
- `ml_consumer_group.py` - ML consumer with consumer group support for load balancing
- `prediction_consumer.py` - Simple consumer that reads and displays predictions
- `database_consumer.py` - Consumer that stores predictions in SQLite database

### New Unified System
- `kafka_consumer.py` - Single, configurable consumer that can replicate all original functionality
- `examples.py` - Usage examples showing how to use the unified consumer
- `README_kafka_consumer.md` - This documentation file

## Architecture

The unified consumer uses a modular, plugin-like architecture:

### Core Components

1. **ConsumerConfig** - Configuration dataclass that controls all consumer behavior
2. **MessageHandler** - Abstract base class for processing messages
3. **UnifiedKafkaConsumer** - Main consumer orchestrator

### Message Handlers

1. **MLPredictionHandler** - Handles machine learning predictions
2. **DatabaseHandler** - Handles SQLite database storage
3. **DisplayHandler** - Handles message display/logging

## Usage Examples

### 1. Simple ML Consumer (replaces `ml_consumer.py`)
```bash
python kafka_consumer.py --preset simple --input-topic house_data
```

### 2. ML Consumer with Producer (replaces `ml_consumer_producer.py`)
```bash
python kafka_consumer.py --preset ml-producer --input-topic house_data --output-topic prediction_duport
```

### 3. ML Consumer with Consumer Group (replaces `ml_consumer_group.py`)
```bash
python kafka_consumer.py --preset ml-group --consumer-group ml_prediction_group --consumer-id consumer_1
```

### 4. Prediction Display Consumer (replaces `prediction_consumer.py`)
```bash
python kafka_consumer.py --preset prediction-consumer --input-topic prediction_duport
```

### 5. Database Storage Consumer (replaces `database_consumer.py`)
```bash
python kafka_consumer.py --preset database-consumer --input-topic prediction_duport --db-path predictions.db
```

### 6. Custom Configurations

You can mix and match features:

```bash
# Full pipeline: ML + Producer + Database
python kafka_consumer.py \
  --input-topic house_data \
  --output-topic prediction_duport \
  --enable-producer \
  --enable-database \
  --consumer-id full_pipeline

# Load balanced pipeline
python kafka_consumer.py \
  --input-topic house_data \
  --output-topic prediction_duport \
  --enable-producer \
  --consumer-group ml_prediction_group \
  --consumer-id worker_1
```

## Command Line Options

### Basic Options
- `--bootstrap-servers`: Kafka bootstrap servers (default: nowledgeable.com:9092)
- `--input-topic`: Input topic to consume from (default: house_data)
- `--output-topic`: Output topic to produce to (default: prediction_duport)

### Consumer Options
- `--consumer-group`: Consumer group ID for load balancing
- `--consumer-id`: Unique consumer ID
- `--auto-offset-reset`: Auto offset reset policy (earliest/latest)

### Feature Flags
- `--enable-database`: Enable database storage
- `--db-path`: Database file path (default: predictions.db)
- `--enable-producer`: Enable message producing
- `--disable-ml`: Disable ML predictions
- `--disable-display`: Disable message display

### Presets
- `--preset simple`: Simple ML consumer only
- `--preset ml-producer`: ML consumer with producer
- `--preset ml-group`: ML consumer with consumer group
- `--preset prediction-consumer`: Display predictions only
- `--preset database-consumer`: Store predictions in database

## Running Examples

Use the examples script to try different configurations:

```bash
python examples.py
```

This will show you all available examples and let you choose which one to run.

## Benefits of the Unified System

1. **Single Source of Truth**: One file to maintain instead of five
2. **Configurable**: Mix and match functionality as needed
3. **Extensible**: Easy to add new handlers
4. **Consistent**: Same patterns and error handling throughout
5. **Maintainable**: Clear separation of concerns
6. **Backward Compatible**: Can replicate all original functionality

## Migration Guide

To migrate from the original files:

1. **Instead of `ml_consumer.py`**:
   ```bash
   python kafka_consumer.py --preset simple
   ```

2. **Instead of `ml_consumer_producer.py`**:
   ```bash
   python kafka_consumer.py --preset ml-producer
   ```

3. **Instead of `ml_consumer_group.py`**:
   ```bash
   python kafka_consumer.py --preset ml-group --consumer-group YOUR_GROUP --consumer-id YOUR_ID
   ```

4. **Instead of `prediction_consumer.py`**:
   ```bash
   python kafka_consumer.py --preset prediction-consumer
   ```

5. **Instead of `database_consumer.py`**:
   ```bash
   python kafka_consumer.py --preset database-consumer
   ```

## Extending the System

To add new functionality:

1. Create a new handler class inheriting from `MessageHandler`
2. Implement the required methods: `setup()`, `handle()`, `cleanup()`
3. Add configuration options to `ConsumerConfig`
4. Register the handler in `UnifiedKafkaConsumer._setup_handlers()`

Example:
```python
class EmailHandler(MessageHandler):
    def setup(self):
        # Setup email client
        pass
    
    def handle(self, message_data, consumer_id=None):
        # Send email notification
        return message_data
    
    def cleanup(self):
        # Close email client
        pass
```