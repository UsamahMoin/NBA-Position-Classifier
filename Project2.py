"""Run the leakage-safe NBA position classification experiment."""

import json

from tools.train_model import OUTPUT_PATH, train


def main() -> None:
    artifact = train()
    OUTPUT_PATH.write_text(
        json.dumps(artifact, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )

    metrics = artifact["metrics"]
    print("NBA Position Classifier")
    print(f"Players: {artifact['playerCount']}")
    print(f"Selected features: {', '.join(artifact['selectedFeatures'])}")
    print(f"Best C: {metrics['bestC']}")
    print(f"Training accuracy: {metrics['trainingAccuracy']:.3f}")
    print(f"Test accuracy: {metrics['testAccuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balancedAccuracy']:.3f}")
    print(
        "Cross-validation balanced accuracy: "
        f"{metrics['crossValidationBalancedAccuracy']:.3f}"
    )
    print("\nConfusion matrix (actual rows, predicted columns):")
    print("     " + "  ".join(artifact["labels"]))
    for label, row in zip(artifact["labels"], artifact["confusionMatrix"]):
        print(f"{label:>2}  " + "  ".join(f"{value:>2}" for value in row))
    print(f"\nWrote {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
