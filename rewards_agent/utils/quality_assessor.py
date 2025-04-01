import re
from typing import Dict, Any
from naptha_sdk.inference import InferenceClient
from naptha_sdk.schemas import ChatCompletionRequest, ModelResponse
from rewards_agent.schemas import ContentSchema, QualityAssessmentSchema
from naptha_sdk.utils import get_logger

logger = get_logger(__name__)

class QualityAssessor:
    def __init__(self, module_run: dict):
        self.deployment = module_run.deployment
        self.llm_client = InferenceClient(self.deployment.node)

    async def assess_content(self, content_data: ContentSchema) -> QualityAssessmentSchema:
        """
        Assess the quality of content using LLM.
        Returns a QualityAssessmentSchema object with quality score and feedback.
        """
        prompt = self._create_assessment_prompt(content_data)

        request = ChatCompletionRequest(
            model=self.deployment.config.llm_config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.deployment.config.llm_config.temperature,
            max_tokens=self.deployment.config.llm_config.max_tokens,
        )
        response = await self.llm_client.run_inference(request)
        return self._process_llm_response(response)

    def _create_assessment_prompt(self, content_data: ContentSchema) -> str:
        """Create a prompt for the LLM to assess content quality."""
        return (
            "Please assess the following content and provide:\n"
            "1. A quality score between 0 and 10\n"
            "2. Specific feedback about the content's strengths and areas for improvement\n"
            f"Content Type: {content_data.content_type}\n"
            f"Content: {content_data.content}\n"
            "Provide your response in the following format:\n"
            "Score: [0-10]\n"
            "Feedback: [Your detailed feedback]\n"
            "Always format feedback in markdown format. Use - for bullet points and ** for bold text. Do not use any other markdown formatting."
        )

    def _process_llm_response(self, response: ModelResponse) -> QualityAssessmentSchema:
        """Process LLM response and extract quality score and feedback."""
        content = response.choices[0].message.content
        cleaned_content = self._clean_content(content)
        quality_score = self._extract_quality_score(content)
        feedback = self._extract_feedback(cleaned_content)

        return QualityAssessmentSchema(
            quality_score=quality_score,
            feedback=feedback,
        )

    def _clean_content(self, content: str) -> str:
        """Remove the score line and content before the feedback section."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        is_feedback_section = False
        cleaned_lines = []

        for line in lines:
            if 'Feedback:' in line:
                is_feedback_section = True
            if is_feedback_section:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _extract_quality_score(self, content: str) -> float:
        """Extract the quality score from the content."""
        score_line = content.split("\n")[0]
        score = float(score_line.split(":")[1].strip())

        return score
    
    def _extract_feedback(self, content: str) -> str:
        """Extract feedback from the LLM response."""
        try:
            content = content.replace('Feedback:', '', 1).strip()
            
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            feedback_sections = {}
            current_section = None

            for line in lines:
                cleaned_line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                cleaned_line = re.sub(r'\*(.*?)\*', r'\1', cleaned_line)
                
                if cleaned_line.endswith(':'):
                    current_section = cleaned_line.rstrip(':')
                    feedback_sections[current_section] = []
                    continue
                
                if current_section and cleaned_line:
                    cleaned_line = re.sub(r'^[\*\-\d\.\s]+', '', cleaned_line).strip()
                    if cleaned_line:
                        feedback_sections[current_section].append(cleaned_line)

            if not feedback_sections:
                feedback_sections['General Feedback'] = [line for line in lines if line]

            formatted_feedback = []
            for section, points in feedback_sections.items():
                if points:
                    formatted_feedback.append(f"{section}:")
                    for point in points:
                        formatted_feedback.append(f"- {point}")
                    formatted_feedback.append("")

            return '\n'.join(formatted_feedback).strip()
        
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return "Error processing feedback. Please check the content format."