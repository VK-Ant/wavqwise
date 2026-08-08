"""CLI: wavqwise forecast --input data.csv --model arima --horizon 30"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(prog="wavqwise", description="WavqWise - Sense. Forecast. Alert.")
    sub = parser.add_subparsers(dest="command")

    fc = sub.add_parser("forecast", help="Run forecasting pipeline")
    fc.add_argument("--input", "-i", required=True)
    fc.add_argument("--target", "-t", default=None)
    fc.add_argument("--time", default=None)
    fc.add_argument("--model", "-m", default="moving_average")
    fc.add_argument("--horizon", "-H", type=int, default=30)
    fc.add_argument("--plot", action="store_true")

    det = sub.add_parser("detect", help="Run anomaly detection")
    det.add_argument("--input", "-i", required=True)
    det.add_argument("--target", "-t", default=None)
    det.add_argument("--method", "-m", default="zscore")

    mod = sub.add_parser("models", help="List available models")

    args = parser.parse_args()

    if args.command == "forecast":
        from wavqwise import WavqPipeline
        p = WavqPipeline()
        p.load(args.input, target=args.target, time=args.time)
        result = p.forecast(horizon=args.horizon, model=args.model)
        print(result.summary())
        if args.plot:
            result.plot()

    elif args.command == "detect":
        from wavqwise import AnomalyPipeline
        p = AnomalyPipeline()
        p.load(args.input, target=args.target)
        result = p.detect(method=args.method)
        print(result.summary())

    elif args.command == "models":
        from wavqwise import WavqPipeline
        print("Available models:")
        for m in WavqPipeline.available_models():
            print(f"  - {m}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
