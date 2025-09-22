#!/usr/bin/env python3
"""
Usage Examples for Unified Kafka Consumer

This file demonstrates how to use the kafka_consumer.py to replicate
the functionality of all the original consumer files.
"""

import subprocess
import sys
import os

def run_example(description: str, command: list):
    """Run an example with description"""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    print("Press Ctrl+C to stop this example and continue to the next one...")
    
    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        print("\nExample stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Example failed: {e}")

def main():
    """Main function to run all examples"""
    
    # Make sure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    examples = [
        {
            "description": "Simple ML Consumer (replaces ml_consumer.py)",
            "command": [
                sys.executable, "kafka_consumer.py",
                "--preset", "simple",
                "--input-topic", "house_data"
            ]
        },
        {
            "description": "ML Consumer with Producer (replaces ml_consumer_producer.py)",
            "command": [
                sys.executable, "kafka_consumer.py",
                "--preset", "ml-producer",
                "--input-topic", "house_data",
                "--output-topic", "prediction_duport"
            ]
        },
        {
            "description": "ML Consumer with Consumer Group (replaces ml_consumer_group.py)",
            "command": [
                sys.executable, "kafka_consumer.py",
                "--preset", "ml-group",
                "--consumer-group", "ml_prediction_group",
                "--consumer-id", "consumer_1",
                "--input-topic", "house_data",
                "--output-topic", "prediction_duport"
            ]
        },
        {
            "description": "Prediction Display Consumer (replaces prediction_consumer.py)",
            "command": [
                sys.executable, "kafka_consumer.py",
                "--preset", "prediction-consumer",
                "--input-topic", "prediction_duport"
            ]
        },
        {
            "description": "Database Storage Consumer (replaces database_consumer.py)",
            "command": [
                sys.executable, "kafka_consumer.py",
                "--preset", "database-consumer",
                "--input-topic", "prediction_duport",
                "--db-path", "predictions.db"
            ]
        },
        {
            "description": "Full Pipeline Consumer (ML + Producer + Database)",
            "command": [
                sys.executable, "kafka_consumer.py",
                "--input-topic", "house_data",
                "--output-topic", "prediction_duport",
                "--enable-producer",
                "--enable-database",
                "--consumer-id", "full_pipeline"
            ]
        },
        {
            "description": "Load Balanced ML Pipeline",
            "command": [
                sys.executable, "kafka_consumer.py",
                "--input-topic", "house_data",
                "--output-topic", "prediction_duport",
                "--enable-producer",
                "--consumer-group", "ml_prediction_group",
                "--consumer-id", "worker_1"
            ]
        }
    ]
    
    print("Unified Kafka Consumer Examples")
    print("=" * 60)
    print("This script demonstrates how to use the unified consumer")
    print("to replicate all the functionality of the original consumer files.")
    print("\nEach example will run until you press Ctrl+C")
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}")
    
    try:
        choice = input("\nEnter example number (1-7) or 'all' to run all: ").strip()
        
        if choice.lower() == 'all':
            for example in examples:
                run_example(example['description'], example['command'])
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                run_example(examples[idx]['description'], examples[idx]['command'])
            else:
                print("Invalid choice")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except ValueError:
        print("Invalid input")

if __name__ == "__main__":
    main()