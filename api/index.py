from main import app as _app


class _StripPrefix:
    """Strip /api prefix so FastAPI routes (defined without it) match correctly."""

    def __init__(self, app, prefix: str):
        self.app = app
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path.startswith(self.prefix):
                stripped = path[len(self.prefix):] or "/"
                scope = {**scope, "path": stripped, "raw_path": stripped.encode()}
        await self.app(scope, receive, send)


app = _StripPrefix(_app, "/api")
