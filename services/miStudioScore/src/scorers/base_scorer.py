# src/scorers/base_scorer.py
"""
Defines the abstract base class for all scoring modules.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseScorer(ABC):
    """
    Abstract base class for a scorer. All scorers must inherit from this
    class and implement the score method.
    """

    @abstractmethod
    def score(self, features: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Applies a scoring algorithm to a list of features.

        This method should modify the features list in-place by adding
        a new score field to each feature dictionary.

        Args:
            features: A list of feature data dictionaries.
            **kwargs: Scorer-specific parameters passed from the orchestrator.

        Returns:
            The list of feature data dictionaries, enriched with new scores.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the scorer.
        """
        pass
