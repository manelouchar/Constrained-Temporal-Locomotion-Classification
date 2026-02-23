from src.ablation.trainer import AblationStudyRunner

def run():
    runner = AblationStudyRunner()
    runner.run_all_ablations(
        model_type="cnn_lstm",
        cnn_variant="deep"
    )

if __name__ == "__main__":
    run()