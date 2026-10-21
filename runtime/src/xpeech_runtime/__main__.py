import argparse
import os


def main() -> None:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    parser = argparse.ArgumentParser(description="Run the sandbox agent service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("xpeech_runtime.service:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
