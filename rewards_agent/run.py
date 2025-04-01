from dotenv import load_dotenv
from typing import Dict
from naptha_sdk.schemas import AgentRunInput
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from rewards_agent.schemas import InputSchema, QualityAssessmentSchema
from rewards_agent.utils.reward_calculator import RewardCalculator
from rewards_agent.utils.document_processor import DocumentProcessor
from langchain_openai import OpenAIEmbeddings

load_dotenv()
logger = get_logger(__name__)

class RewardsAgent:
    def __init__(self, module_run: Dict):
        self.module_run = module_run
        self.reward_calculator = RewardCalculator(
            min_quality_threshold=module_run.inputs.quality_threshold, 
            base_reward=module_run.inputs.base_reward
        )
        self.document_processor = DocumentProcessor(module_run, OpenAIEmbeddings())

    async def assess_and_reward(self, input_data: Dict) -> Dict:
        """Assess content quality and provide rewards if merited."""
        logger.info(f"Processing content for assessment and rewards")
        
        if "file_path" in input_data:
            quality_assessment = await self.document_processor.process_document(file_path=input_data["file_path"], agent_id=input_data["agent_id"])
        elif "text" in input_data:
            quality_assessment = await self.document_processor.process_document(text=input_data["text"], agent_id=input_data["agent_id"])
        else:
            raise ValueError("No file path or text provided")

        quality_assessment = QualityAssessmentSchema(**quality_assessment)
            
        if self.reward_calculator.should_reward(quality_assessment):
            reward_amount = self.reward_calculator.calculate_reward(quality_assessment)
            quality_assessment.reward_amount = reward_amount
            logger.info(f"Reward of {reward_amount} calculated for agent {input_data['agent_id']}")
        else:
            logger.info(f"Content quality below threshold for agent {input_data['agent_id']}")
        
        return quality_assessment.dict()

async def run(module_run: Dict):
    module_run = AgentRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    rewards_agent = RewardsAgent(module_run)

    return await rewards_agent.assess_and_reward(module_run.inputs.func_input_data)

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment(
        "agent",
        "rewards_agent/configs/deployment.json",
        node_url=os.getenv("NODE_URL")
    ))

    input_params = {
        "func_input_data": {
            "agent_id": "test_agent",
            "file_path": "rewards_agent/test_files/test_file.pdf"
        },
        "quality_threshold": 3,
        "base_reward": 100
    }

    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY_FULL_PATH"))
    }

    response = asyncio.run(run(module_run))
    print("Response: ", response)