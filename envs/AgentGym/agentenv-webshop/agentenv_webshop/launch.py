"""
Entrypoint for the webshop agent environment.
"""

import argparse

import uvicorn

from .utils import debug_flg


def launch():
    """entrypoint for `webshop` commond"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of uvicorn worker processes (default: 1)")
    parser.add_argument("--limit-concurrency", type=int, default=None,
                        help="Max simultaneous concurrent connections (default: unlimited)")
    parser.add_argument("--backlog", type=int, default=2048,
                        help="Max length of the connection queue (default: 2048)")
    parser.add_argument("--timeout-keep-alive", type=int, default=5,
                        help="Keep-alive timeout in seconds (default: 5)")
    args = parser.parse_args()

    uvicorn.run(
        "agentenv_webshop:app",
        host=args.host,
        port=args.port,
        reload=debug_flg,
        workers=args.workers,
        limit_concurrency=args.limit_concurrency,
        backlog=args.backlog,
        timeout_keep_alive=args.timeout_keep_alive,
    )
