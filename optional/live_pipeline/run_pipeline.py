# Optional historical provider pipeline; invoke explicitly after installing its requirements.
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import json
import time
import os
import datetime
import sys


def main():
    print("\n=== MLIRBench Pipeline ===\n")

    print("1. Load MLIR from file")
    print("2. Paste MLIR code")

    choice = input("\nChoose option (1/2): ").strip()

    if choice == "1":

        mlir_file = input(
            "\nEnter MLIR file path: "
        ).strip()

        try:
            with open(mlir_file, "r") as f:
                input_mlir = f.read()
        except FileNotFoundError:
            print(f"\nFile not found: {mlir_file}")
            return
        except Exception as e:
            print(f"\nError reading file: {e}")
            return

    elif choice == "2":

        print(
            "\nPaste MLIR below."
        )
        print(
            "Type END on a new line when finished.\n"
        )

        lines = []

        while True:
            line = input()

            if line.strip() == "END":
                break

            lines.append(line)

        input_mlir = "\n".join(lines)

    else:
        print("Invalid option.")
        exit(1)

    dialect = input(
        "Enter dialect (linalg/torch): "
    ).strip().lower()

    if dialect not in ["linalg", "torch"]:
        print("Invalid dialect.")
        return

    try:
        level = int(
            input(
                "Enter mutation level (1-5): "
            ).strip()
        )

        if level not in [1, 2, 3, 4, 5]:
            raise ValueError()

    except ValueError:
        print("Mutation level must be 1-5.")
        return

    from graph.pipeline import build_graph
    graph = build_graph()

    print("\nStarting pipeline...\n")
    print("\n========== CONFIG ==========")
    print(f"Dialect: {dialect}")
    print(f"Level:   {level}")
    print(f"Size:    {len(input_mlir)} chars")
    print("============================")
    start = time.time()

    try:
        result = graph.invoke(
            {
                "user_input": {
                    "input_mlir": input_mlir,
                    "dialect": dialect,
                    "level": level,
                },
                "logs": [],
            }
        )

    except Exception as e:

        print("\nPipeline failed:\n")
        print(str(e))
        return

    elapsed = time.time() - start
    os.makedirs("runs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime(
    "%Y%m%d_%H%M%S"
    )

    output_file = (
    f"runs/run_{dialect}_L{level}_{timestamp}.txt"
    )

    with open(output_file, "w") as f:
        f.write("========== MLIRBench Run ==========\n\n")
        f.write(f"Dialect: {dialect}\n")
        f.write(f"Level: {level}\n")
        f.write(f"Runtime: {elapsed:.2f}s\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("========== RESULT ==========\n\n")

        f.write(
            json.dumps(
                result,
                indent=2,
                default=str
            )
        )

    print("\n========== RUN COMPLETE ==========\n")
    print(f"Results saved to: {output_file}")
    print(f"Pipeline completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
