"""FastAPI application for the Financial Fraud Detection Environment."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from models import FraudAction, FraudObservation
    from server.ff_env_environment import FfEnvironment
except ModuleNotFoundError:
    from ff_env.models import FraudAction, FraudObservation
    from ff_env.server.ff_env_environment import FfEnvironment

app = create_app(
    FfEnvironment,
    FraudAction,
    FraudObservation,
    env_name="ff_env",
    max_concurrent_envs=1,
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()