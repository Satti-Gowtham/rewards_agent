from pydantic import BaseModel
from typing import Union, Dict, Any, List, Optional

class ContentSchema(BaseModel):
    content: str
    agent_id: str
    content_type: Optional[str] = "text"
    metadata: Optional[Dict[str, Any]] = None

class QualityAssessmentSchema(BaseModel):
    quality_score: float
    feedback: str
    reward_amount: Optional[float] = None

class InputSchema(BaseModel):
    func_input_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]], str]] = None
    quality_threshold: float = 7
    base_reward: float = 10