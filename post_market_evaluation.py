import argparse

from src.expert_knowledge import do_expert_knowledge_check
from src.statistical_analysis import do_statistical_analysis
from src.adversarial_evaluation import do_adversarial_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Perform post-market evaluation of the synthetic data"
    )
    # Expert Knowledge Arguments
    parser.add_argument(
        "--synthetic_samples_path",
        type=str,
        default="data/synthetic_samples",
        help="Directory containing synthetic samples",
    )
    parser.add_argument(
        "--expert_knowledge_results_path",
        type=str,
        default="expert_knowledge_results.json",
        help="Path to the expert knowledge results",
    )
    # Statistical Analysis Arguments
    parser.add_argument(
        "--statistical_analysis_results_path",
        type=str,
        default="statistical_analysis_results.json",
        help="Path to the statistical analysis results",
    )
    # Adversarial Evaluation Arguments
    parser.add_argument(
        "--original_samples_path",
        type=str,
        default="data/original_samples",
        help="Path to the original samples",
    )
    parser.add_argument(
        "--adversarial_evaluation_results_path",
        type=str,
        default="adversarial_evaluation_results.json",
        help="Path to the adversarial evaluation results",
    )
    args = parser.parse_args()


    expert_knowledge_results = do_expert_knowledge_check(args.synthetic_samples_path)

    # do_statistical_analysis()

    # do_adversarial_evaluation()

    # with open(args.expert_knowledge_results_path, "r") as f:
    #     expert_knowledge_results = json.load(f)

    # with open(args.statistical_analysis_results_path, "r") as f:
    #     statistical_analysis_results = json.load(f)

    # with open(args.adversarial_evaluation_results_path, "r") as f:
    #     adversarial_evaluation_results = json.load(f)

    # combined_results = {
    #     "expert_knowledge_evaluation": expert_knowledge_results,
    #     "statistical_analysis_evaluation": statistical_analysis_results,
    #     "adversarial_evaluation": adversarial_evaluation_results,
    # }

    # with open("post_market_evaluation_results.json", "w") as f:
    #     json.dump(combined_results, f, indent=4)

    # os.remove(args.expert_knowledge_results_path)
    # os.remove(args.statistical_analysis_results_path)
    # os.remove(args.adversarial_evaluation_results_path)


if __name__ == "__main__":
    main()
