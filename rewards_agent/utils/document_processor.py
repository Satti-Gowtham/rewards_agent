from langchain_experimental.text_splitter import SemanticChunker
from rewards_agent.utils.ollama_embedder import OllamaNomicEmbedder
from rewards_agent.utils.quality_assessor import QualityAssessor
from rewards_agent.schemas import ContentSchema
from typing import List, Dict
import filetype
import fitz
from docx import Document
from naptha_sdk.utils import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    def __init__(
            self,
            module_run: Dict,
            embedding_model: OllamaNomicEmbedder,
            breakpoint_threshold: float = 95,
        ):
        self.chunker = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold
        )
        self.assessor = QualityAssessor(module_run)
    
    def load_pdf(self, file_path: str) -> str:
        """ Loads a PDF file and returns the text content. """
        with fitz.open(file_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        
        return text

    def load_docx(self, file_path: str) -> str:
        """ Loads a DOCX file and returns the text content. """
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        return text

    def load_unknown(self, file_path: str) -> str:
        """ Handles files that cannot be processed. """
        return f"Unsupported file type: {file_path}"
    
    def load_file(self, file_path: str) -> str:
        """
        Determines the file type and loads the content accordingly.

        :param file_path: Path to the file to be loaded.
        :return: Content of the file as a string, or an error message if unsupported.
        """
        kind = filetype.guess(file_path)
        if kind is None:
            return f"Cannot determine file type for {file_path}"

        mime_type_map = {
            'application/pdf': self.load_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self.load_docx,
        }

        load_function = mime_type_map.get(kind.mime, self.load_unknown)
        return load_function(file_path)

    def chunk_text(self, text: str) -> List[str]:
        """ Chunks text into smaller chunks. """
        document = self.chunker.create_documents([text])

        return [doc.page_content for doc in document]

    async def process_document(self, file_path: str = None, agent_id: str = None, text: str = None) -> Dict:
        """ Processes a document and returns a quality assessment. """
        if file_path:
            content = self.load_file(file_path)
        else:
            content = text

        chunks = self.chunk_text(content)

        total_score = 0
        feedback_list = []
        
        for chunk in chunks:
            assessment = await self.assessor.assess_content(ContentSchema(content=chunk, agent_id=agent_id))
            score = assessment.quality_score
            total_score += score
            feedback_list.append(assessment.feedback)

        average_score = total_score / len(chunks) if chunks else 0
        combined_feedback = "\n\n".join(feedback_list)

        return {
            "quality_score": average_score,
            "feedback": combined_feedback
        }