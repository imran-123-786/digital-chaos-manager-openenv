from digital_chaos_env.server.app import app as _app
from digital_chaos_env.server.app import main as _inner_main

app = _app


def main() -> None:
    _inner_main()

if __name__ == "__main__":
    main()
