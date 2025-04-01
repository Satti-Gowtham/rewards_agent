from rewards_agent.schemas import QualityAssessmentSchema

class RewardCalculator:
    def __init__(self, base_reward: float = 10.0, min_quality_threshold: float = 5.0):
        self.base_reward = base_reward
        self.min_quality_threshold = min_quality_threshold

    def calculate_reward(self, quality_assessment: QualityAssessmentSchema) -> float:
        """
        Calculate reward amount based on quality assessment.
        
        Returns 0 if quality is below threshold.
        """
        if quality_assessment.quality_score < self.min_quality_threshold:
            return 0.0
        
        reward_multiplier = quality_assessment.quality_score / 10.0
        return self.base_reward * reward_multiplier

    def should_reward(self, quality_assessment: QualityAssessmentSchema) -> bool:
        """Determine if the content quality merits a reward."""
        return quality_assessment.quality_score >= self.min_quality_threshold 