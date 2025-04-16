from src.config import Config
from src.trainers import CircleTrainer

if __name__ == '__main__':
    # Load configuration
    config = Config.from_json("src/config_benchmarking.json")

    # Create trainer
    trainer = CircleTrainer(config)

    # Train model
    trainer.train()