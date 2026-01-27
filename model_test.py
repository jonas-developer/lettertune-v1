# model_test.py

from mockup_data import job_text, resume_text, old_letter_text, old_letter_text_young
import model


def run_test():
    # Choose which model to test:
    # - model.llama_response
    # - model.granite_response
    # - model.mistral_response
    # - model.openai_response
    # - model.deepseek_response
    response_fn = model.llama_response

    additional_instructions = (
        "Keep it under 250 words. "
        "Include a short opening line referencing the company's mission."
    )

    result = response_fn(
        company_job_info=job_text,
        applicant_background=resume_text,
        previous_cover_letter=old_letter_text,
        additional_instructions=additional_instructions,
    )

    print("=== GENERATED COVER LETTER ===\n")
    print(result.get("cover_letter", ""))

    print("\n=== KEY MATCHES ===")
    for match in result.get("key_matches", []):
        print("-", match)

    print("\n=== STYLE NOTES ===")
    print(result.get("style_notes", ""))

    # Optional field
    suggested_subject = result.get("suggested_subject")
    if suggested_subject:
        print("\n=== SUGGESTED SUBJECT ===")
        print(suggested_subject)


if __name__ == "__main__":
    run_test()
