import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create a test set split from the full dataset")
    parser.add_argument('--input', type=str, default="musiccaps_processed.json", help="Path to full processed json dataset")
    parser.add_argument('--output', type=str, default="data/musiccaps_test_1001_1100.json", help="Output path for the test subset")
    parser.add_argument('--start', type=int, default=1000, help="Start index (0-indexed)")
    parser.add_argument('--end', type=int, default=1100, help="End index (exclusive)")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate indices
    start = max(0, args.start)
    end = min(len(data), args.end)
    
    test_data = data[start:end]

    print(f"Saving {len(test_data)} items to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
        
    print("Done!")

if __name__ == '__main__':
    main()
