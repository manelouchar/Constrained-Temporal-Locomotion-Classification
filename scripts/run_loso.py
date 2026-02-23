from scripts.run_loso import LOSOTrainer

def run():
    trainer = LOSOTrainer(
        model_type="cnn_lstm",
        cnn_variant="deep"
    )
    trainer.run_loso()

if __name__ == "__main__":
    run()