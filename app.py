from flask import Flask, request, jsonify, render_template
from model import (
    llama_response,
    granite_response,
    mistral_response,
    openai_response,
    deepseek_response,
)

import time

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # Accept JSON from fetch()
    data = request.get_json(silent=True) or {}

    model_name = (data.get("model") or "").strip().lower()
    company_job_info = (data.get("company_job_info") or "").strip()
    applicant_background = (data.get("applicant_background") or "").strip()
    previous_cover_letter = (data.get("previous_cover_letter") or "").strip()
    additional_instructions = (data.get("additional_instructions") or "").strip()

    # Validate required fields
    missing = []
    if not model_name:
        missing.append("model")
    if not company_job_info:
        missing.append("company_job_info")
    if not applicant_background:
        missing.append("applicant_background")
    if not previous_cover_letter:
        missing.append("previous_cover_letter")

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    start_time = time.time()

    try:
        if model_name == "llama":
            result = llama_response(
                company_job_info=company_job_info,
                applicant_background=applicant_background,
                previous_cover_letter=previous_cover_letter,
                additional_instructions=additional_instructions,
            )
        elif model_name == "granite":
            result = granite_response(
                company_job_info=company_job_info,
                applicant_background=applicant_background,
                previous_cover_letter=previous_cover_letter,
                additional_instructions=additional_instructions,
            )
        elif model_name == "mistral":
            result = mistral_response(
                company_job_info=company_job_info,
                applicant_background=applicant_background,
                previous_cover_letter=previous_cover_letter,
                additional_instructions=additional_instructions,
            )
        elif model_name == "openai":
            result = openai_response(
                company_job_info=company_job_info,
                applicant_background=applicant_background,
                previous_cover_letter=previous_cover_letter,
                additional_instructions=additional_instructions,
            )
        elif model_name == "deepseek":
            result = deepseek_response(
                company_job_info=company_job_info,
                applicant_background=applicant_background,
                previous_cover_letter=previous_cover_letter,
                additional_instructions=additional_instructions,
            )
        else:
            return jsonify({"error": "Invalid model selection"}), 400

        result["duration"] = round(time.time() - start_time, 3)
        result["model"] = model_name
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    
    app.run(debug=True)
