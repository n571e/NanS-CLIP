from pathlib import Path

from web_demo.backend.app import create_app
from web_demo.backend.demo_backend import DemoBackend


def main():
    repo_root = Path(__file__).resolve().parents[2]
    backend = DemoBackend.from_repo_root(repo_root=repo_root)
    backend.warm_up()
    app = create_app(backend)
    app.run(host="127.0.0.1", port=8765, debug=False)


if __name__ == "__main__":
    main()
