from io import BytesIO

from flask import Flask, jsonify, request, send_file, send_from_directory


def create_app(demo_backend=None):
    if demo_backend is None:
        from web_demo.backend.demo_backend import DemoBackend

        demo_backend = DemoBackend.from_repo_root()

    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.route("/api/search/compare", methods=["OPTIONS"])
    def compare_search_options():
        return ("", 204)

    @app.route("/api/search/image-to-text", methods=["OPTIONS"])
    def image_to_text_options():
        return ("", 204)

    @app.get("/api/summary")
    def summary():
        return jsonify(demo_backend.get_summary())

    @app.get("/api/examples")
    def examples():
        return jsonify(demo_backend.get_examples())

    @app.get("/api/item/<path:filename>")
    def item(filename):
        payload = demo_backend.get_item(filename)
        if payload is None:
            return jsonify({"error": "item_not_found", "filename": filename}), 404
        return jsonify(payload)

    @app.post("/api/search/compare")
    def compare_search():
        payload = request.get_json(silent=True) or {}
        query = payload.get("query", "")
        top_k = min(max(int(payload.get("top_k", 5)), 1), 20)
        try:
            result = demo_backend.compare_search(query, top_k)
        except ValueError as exc:
            return jsonify({"error": "invalid_query", "message": str(exc)}), 400
        return jsonify(result | {"top_k": top_k})

    @app.post("/api/search/image-to-text")
    def image_to_text_search():
        payload = request.get_json(silent=True) or {}
        top_k_raw = request.form.get("top_k", payload.get("top_k", 5))
        top_k = min(max(int(top_k_raw), 1), 20)
        filename = request.form.get("filename") or payload.get("filename")
        image_id_raw = request.form.get("image_id", payload.get("image_id"))
        image_id = int(image_id_raw) if image_id_raw is not None else None
        uploaded = request.files.get("image")
        image_bytes = uploaded.read() if uploaded else None
        try:
            result = demo_backend.compare_image_to_text(
                top_k=top_k,
                filename=filename,
                image_id=image_id,
                image_bytes=image_bytes,
            )
        except ValueError as exc:
            return jsonify({"error": "invalid_image_query", "message": str(exc)}), 400
        return jsonify(result | {"top_k": top_k})

    @app.get("/api/images/<path:filename>")
    def image_asset(filename):
        image_dir = demo_backend.get_image_directory()
        return send_from_directory(image_dir, filename)

    @app.get("/api/eval-images/<int:image_id>")
    def benchmark_image_asset(image_id):
        image_bytes = demo_backend.get_benchmark_image_bytes(image_id)
        if image_bytes is None:
            return jsonify({"error": "image_not_found", "image_id": image_id}), 404
        return send_file(BytesIO(image_bytes), mimetype="image/jpeg")

    return app
