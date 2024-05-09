from flask import Flask, json, request, jsonify, stream_with_context, Response
import os
from werkzeug.utils import secure_filename
from llm import generate_llama_response, generate_vector_space


app = Flask(__name__)


# app.config["UPLOAD_FOLDER"] = os.path.join(app.instance_path, "uploads")
# os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# @app.route("/api/upload", methods=["POST"])
# def upload_file():
#     for uploaded_file in request.files.getlist("file"):
#         uploaded_file.save(
#             os.path.join(
#                 app.config["UPLOAD_FOLDER"], secure_filename(uploaded_file.filename)
#             )
#         )

#     return "Success"


@app.route("/api/response", methods=["GET", "POST"])
def response():
    input = request.get_json()

    response_gen = generate_llama_response(**input)

    def send_tokens():
        for token in response_gen:
            yield f"{token}"

    return Response(send_tokens(), content_type="text/event-stream")


@app.route("/api/generate_context", methods=["GET", "POST"])
def generate_context():
    context = request.get_json()

    generate_vector_space(context)

    return json.dumps({"success": True}), 200, {"ContentType": "application/json"}


if __name__ == "__main__":
    app.run(debug=True)
