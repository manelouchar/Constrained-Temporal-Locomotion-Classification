from src.postprocessing.evaluate_postfilter import PostFilterEvaluator

def main():
    evaluator = PostFilterEvaluator(
        config_path="configs/config.yaml",
        model_type="cnn_lstm_deep",
    )
    evaluator.run()

if __name__ == "__main__":
    main()