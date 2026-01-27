from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

from config import (
    PARAMETERS, CREDENTIALS,
    LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID,
    OPENAI_API_KEY, OPENAI_MODEL_ID,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL_ID, DEEPSEEK_BASE_URL
)

from mockup_data import job_text, resume_text, old_letter_text



# 1) Output schema (JSON)
class CoverLetterResponse(BaseModel):
    cover_letter: str = Field(
        description="The final cover letter text, ready to copy/paste."
    )
    key_matches: List[str] = Field(
        description="Bullet list of the strongest matches between job requirements and applicant background."
    )
    style_notes: str = Field(
        description="Short notes describing the writing style you imitated from the previous cover letter (tone, structure, phrasing)."
    )
    suggested_subject: Optional[str] = Field(
        default=None,
        description="Optional email subject line for sending the cover letter."
    )


json_parser = JsonOutputParser(pydantic_object=CoverLetterResponse)



# 2) Model initialization
def initialize_model(model_id: str) -> ChatWatsonx:
    return ChatWatsonx(
        model_id=model_id,
        url="https://eu-de.ml.cloud.ibm.com",
        project_id="c61b10b6-b6f7-474a-8641-34cc9f57f9c8",
        params=PARAMETERS,
        api_key=CREDENTIALS,
    )


llama_llm = initialize_model(LLAMA_MODEL_ID)
granite_llm = initialize_model(GRANITE_MODEL_ID)
mistral_llm = initialize_model(MISTRAL_MODEL_ID)


# OpenAI model
openai_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL_ID,
    temperature=PARAMETERS.get("temperature", 0.7),
)

# DeepSeek model (OpenAI-compatible)
deepseek_llm = ChatOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model=DEEPSEEK_MODEL_ID,
    temperature=PARAMETERS.get("temperature", 0.7),
)


# 3) System prompt (fixed)
SYSTEM_PROMPT = """You are LetterTune, an assistant that writes highly personalized cover letters.

Goals:
- Use the COMPANY + JOB info to understand the role, responsibilities, and requirements.
- Use the APPLICANT BACKGROUND/RESUME to select relevant experiences, skills, achievements, and keywords.
- Mimic the LANGUAGE STYLE of the APPLICANT'S PREVIOUS COVER LETTER (tone, phrasing, sentence length, structure, formality, personality).
- Do NOT copy content verbatim from the previous letter if it doesn't fit; mimic style, not facts.
- Keep it truthful: only claim what is supported by the applicant background.
- Output MUST follow the JSON schema exactly.

Quality requirements:
- Make the letter sound human, specific, and not generic.
- Prefer concrete details (projects, metrics, tools, outcomes) when available.
- If something is missing, make a reasonable neutral statement rather than inventing specifics.
"""



# 4) Prompt templates (model-specific wrappers)
#    IMPORTANT: At invoke-time we only pass 4 user inputs.

BASE_USER_PROMPT = """
COMPANY + JOB OFFER INFO:
{company_job_info}

APPLICANT BACKGROUND / RESUME:
{applicant_background}

APPLICANT'S PREVIOUS COVER LETTER (STYLE, GRAMMAR and SPELLING REFERENCE):
{previous_cover_letter}

ADDITIONAL INSTRUCTIONS (optional):
{additional_instructions}

Write a cover letter tailored to the job and company, using the applicant's background,
in the same language style as the previous cover letter.
"""

# Llama-style wrapper
llama_template = PromptTemplate(
    template=(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "{system_prompt}\n"
        "{format_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        + BASE_USER_PROMPT
        + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    ),
    input_variables=[
        "system_prompt",
        "format_prompt",
        "company_job_info",
        "applicant_background",
        "previous_cover_letter",
        "additional_instructions",
    ],
)

# Granite-style wrapper
granite_template = PromptTemplate(
    template=(
        "System: {system_prompt}\n"
        "{format_prompt}\n"
        "Human:\n"
        + BASE_USER_PROMPT
        + "\nAI:"
    ),
    input_variables=[
        "system_prompt",
        "format_prompt",
        "company_job_info",
        "applicant_background",
        "previous_cover_letter",
        "additional_instructions",
    ],
)

# Mistral-style wrapper
mistral_template = PromptTemplate(
    template=(
        "<s>[INST]{system_prompt}\n"
        "{format_prompt}\n"
        + BASE_USER_PROMPT
        + "[/INST]"
    ),
    input_variables=[
        "system_prompt",
        "format_prompt",
        "company_job_info",
        "applicant_background",
        "previous_cover_letter",
        "additional_instructions",
    ],
)


openai_template = PromptTemplate(
    template=(
        "{system_prompt}\n\n"
        "{format_prompt}\n\n"
        "{company_job_info}\n\n"
        "{applicant_background}\n\n"
        "{previous_cover_letter}\n\n"
        "{additional_instructions}"
    ),
    input_variables=[
        "system_prompt",
        "format_prompt",
        "company_job_info",
        "applicant_background",
        "previous_cover_letter",
        "additional_instructions",
    ],
)




# 5) Chain builder: partial() removes the need to pass system/format each time
#    So the public functions truly take only 4 user inputs.

def _make_chain(model: ChatWatsonx, template: PromptTemplate):
    prompt = template.partial(
        system_prompt=SYSTEM_PROMPT,
        format_prompt=json_parser.get_format_instructions(),
    )
    return prompt | model | json_parser


_llama_chain = _make_chain(llama_llm, llama_template)
_granite_chain = _make_chain(granite_llm, granite_template)
_mistral_chain = _make_chain(mistral_llm, mistral_template)

_openai_chain = _make_chain(openai_llm, openai_template)
_deepseek_chain = _make_chain(deepseek_llm, openai_template)



# 6) Public API: each model takes 4 inputs
def _generate_cover_letter(chain, company_job_info: str, applicant_background: str, previous_cover_letter: str, additional_instructions: str = ""):
    payload = {
        "company_job_info": company_job_info,
        "applicant_background": applicant_background,
        "previous_cover_letter": previous_cover_letter,
        "additional_instructions": additional_instructions or "",
    }
    return chain.invoke(payload)


def llama_response(company_job_info: str, applicant_background: str, previous_cover_letter: str, additional_instructions: str = ""):
    return _generate_cover_letter(_llama_chain, company_job_info, applicant_background, previous_cover_letter, additional_instructions)


def granite_response(company_job_info: str, applicant_background: str, previous_cover_letter: str, additional_instructions: str = ""):
    return _generate_cover_letter(_granite_chain, company_job_info, applicant_background, previous_cover_letter, additional_instructions)


def mistral_response(company_job_info: str, applicant_background: str, previous_cover_letter: str, additional_instructions: str = ""):
    return _generate_cover_letter(_mistral_chain, company_job_info, applicant_background, previous_cover_letter, additional_instructions)



def openai_response(company_job_info, applicant_background, previous_cover_letter, additional_instructions=""):
    return _generate_cover_letter(
        _openai_chain,
        company_job_info,
        applicant_background,
        previous_cover_letter,
        additional_instructions,
    )

def deepseek_response(company_job_info, applicant_background, previous_cover_letter, additional_instructions=""):
    return _generate_cover_letter(
        _deepseek_chain,
        company_job_info,
        applicant_background,
        previous_cover_letter,
        additional_instructions,
    )

