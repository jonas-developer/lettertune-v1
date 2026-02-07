from flask import Flask, request, jsonify, render_template
from model import (
    llama_response,
    granite_response,
    mistral_response,
    openai_response,
    deepseek_response,
)

from usage_counter import (
    init_counter,
    check_and_increment,
    get_usage,
    MAX_GENERATIONS,
)

import time

app = Flask(__name__)

# Initialize counter table on startup
try:
    init_counter()
except Exception as e:
    print("⚠️ Generation counter init failed:", e)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/usage", methods=["GET"])
def usage():
    return jsonify({
        "used": get_usage(),
        "limit": MAX_GENERATIONS
    })

@app.route("/robots.txt")
def robots():
    return send_from_directory(app.static_folder, "robots.txt")

@app.route("/sitemap.xml")
def sitemap():
    return send_from_directory(app.static_folder, "sitemap.xml")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}

    model_name = (data.get("model") or "").strip().lower()
    company_job_info = (data.get("company_job_info") or "").strip()
    applicant_background = (data.get("applicant_background") or "").strip()
    previous_cover_letter = (data.get("previous_cover_letter") or "").strip()
    additional_instructions = (data.get("additional_instructions") or "").strip()

    missing = []
    if not model_name:
        missing.append("model")
    if not company_job_info:
        missing.append("Company + job offer info")
    if not applicant_background:
        missing.append("Applicant background / resume")
    if not previous_cover_letter:
        missing.append("Previous cover letter (style reference)")

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    # 🔒 Enforce global generation limit
    allowed, new_total = check_and_increment()
    if not allowed:
        return jsonify({
            "error": "Generation limit reached",
            "used": new_total,
            "limit": MAX_GENERATIONS
        }), 429

    start_time = time.time()

    try:
        if model_name == "llama":
            result = llama_response(
                company_job_info,
                applicant_background,
                previous_cover_letter,
                additional_instructions,
            )
        elif model_name == "granite":
            result = granite_response(
                company_job_info,
                applicant_background,
                previous_cover_letter,
                additional_instructions,
            )
        elif model_name == "mistral":
            result = mistral_response(
                company_job_info,
                applicant_background,
                previous_cover_letter,
                additional_instructions,
            )
        elif model_name == "openai":
            result = openai_response(
                company_job_info,
                applicant_background,
                previous_cover_letter,
                additional_instructions,
            )
        elif model_name == "deepseek":
            result = deepseek_response(
                company_job_info,
                applicant_background,
                previous_cover_letter,
                additional_instructions,
            )
        else:
            return jsonify({"error": "Invalid model selection"}), 400

        result["duration"] = round(time.time() - start_time, 3)
        result["model"] = model_name
        result["used"] = new_total
        result["limit"] = MAX_GENERATIONS
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
