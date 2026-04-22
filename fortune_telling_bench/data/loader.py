"""
Data loader for benchmark questions.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..utils.logger import get_logger
from ..utils.path_utils import find_data_file

logger = get_logger(__name__)


class DataLoader:
    """Load and manage benchmark data."""

    BASE_BENCHMARK_YEAR = 2022
    QUESTIONS_PER_BENCHMARK_YEAR = 40
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data file
        """
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Try to find data file
            found_path = find_data_file("data.json")
            if not found_path:
                # Try alternative names
                found_path = find_data_file("fortune_telling_bench_standard.json")
            
            if found_path:
                self.data_path = found_path
            else:
                # Default fallback
                self.data_path = Path("data/data.json")
        
        self.data = None
        self.astro_data_path = Path("data/data_with_astro.json")
        self.fortune_data = None
        self.fortune_data_path = Path("data/fortune_api_results.json")
        self.raw_data_path = Path("data/raw")
        
    def load_questions(self, 
                      use_astro: bool = False,
                      sample_size: Optional[int] = None,
                      year: Optional[int] = None,
                      categories: Optional[List[str]] = None,
                      shuffle: bool = True,
                      shuffle_options: bool = False) -> List[Dict[str, Any]]:
        """
        Load questions from data file.
        
        Args:
            use_astro: Whether to use astronomical data
            sample_size: Number of questions to sample
            year: Filter by benchmark year inferred from question_number
            categories: Filter by categories
            shuffle: Whether to shuffle questions
            shuffle_options: Whether to shuffle options within each question
            
        Returns:
            List of question dictionaries
        """
        # Choose data file
        if use_astro and self.astro_data_path.exists():
            data_path = self.astro_data_path
            logger.info(f"Loading astronomical data from {data_path}")
        else:
            data_path = self.data_path
            logger.info(f"Loading standard data from {data_path}")
        
        # Load data
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Load fortune data if requested
        if use_astro:
            self._load_fortune_data()
        
        # Get questions
        questions = self.data.get('questions', [])
        
        # Validate question format
        valid_questions = []
        for q in questions:
            if self._validate_question(q):
                normalized_question = q.copy()
                benchmark_year = self.infer_benchmark_year(normalized_question)
                if benchmark_year is not None:
                    normalized_question['benchmark_year'] = benchmark_year
                valid_questions.append(normalized_question)
            else:
                logger.warning(f"Skipping invalid question: {q.get('id', 'unknown')}")

        questions = valid_questions
        logger.info(f"Loaded {len(questions)} valid questions")

        # Filter by benchmark year if specified
        if year is not None:
            available_years = sorted({
                q['benchmark_year']
                for q in questions
                if q.get('benchmark_year') is not None
            })
            filtered_questions = [
                q for q in questions
                if q.get('benchmark_year') == year
            ]

            if not filtered_questions and year not in available_years:
                raw_years = self.get_available_raw_years()
                if year in raw_years:
                    raise ValueError(
                        f"Benchmark year {year} exists in raw data "
                        f"({self.raw_data_path / f'{year}.txt'}), but it has not been "
                        f"standardized into {data_path} yet."
                    )
                raise ValueError(
                    f"Benchmark year {year} is not available in {data_path}. "
                    f"Available years: {available_years}"
                )

            questions = filtered_questions
            logger.info(f"Filtered to {len(questions)} questions in benchmark year: {year}")
        
        # Filter by categories if specified (improved version)
        if categories:
            filtered_questions = []
            for q in questions:
                q_category = q.get('category')
                if q_category and q_category in categories:
                    filtered_questions.append(q)
            questions = filtered_questions
            logger.info(f"Filtered to {len(questions)} questions in categories: {categories}")
        
        # Shuffle options if requested
        if shuffle_options:
            questions = [self.shuffle_question_options(q) for q in questions]
            logger.info(f"Shuffled options for {len(questions)} questions")
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(questions)
        
        # Sample if specified
        if sample_size and sample_size < len(questions):
            questions = questions[:sample_size]
            logger.info(f"Sampled {sample_size} questions")

        return questions

    def infer_benchmark_year(self, question: Dict[str, Any]) -> Optional[int]:
        """
        Infer benchmark year from the question metadata.

        Args:
            question: Question dictionary

        Returns:
            Inferred benchmark year, or None if it cannot be determined
        """
        benchmark_year = question.get('benchmark_year')
        if benchmark_year is not None:
            try:
                return int(benchmark_year)
            except (TypeError, ValueError):
                return None

        question_number = question.get('question_number')
        try:
            question_number = int(question_number)
        except (TypeError, ValueError):
            return None

        if question_number < 1:
            return None

        year_offset = (question_number - 1) // self.QUESTIONS_PER_BENCHMARK_YEAR
        return self.BASE_BENCHMARK_YEAR + year_offset

    def get_available_years(self, use_astro: bool = False) -> List[int]:
        """
        Get benchmark years currently available in the standardized dataset.

        Args:
            use_astro: Whether to inspect the astro dataset variant

        Returns:
            Sorted list of benchmark years
        """
        questions = self.load_questions(use_astro=use_astro, shuffle=False)
        years = {
            q['benchmark_year']
            for q in questions
            if q.get('benchmark_year') is not None
        }
        return sorted(years)

    def get_available_raw_years(self) -> List[int]:
        """
        Get benchmark years available as raw text files under data/raw.

        Returns:
            Sorted list of raw benchmark years
        """
        if not self.raw_data_path.exists():
            return []

        years = []
        for raw_file in self.raw_data_path.glob("*.txt"):
            if raw_file.stem.isdigit():
                years.append(int(raw_file.stem))

        return sorted(set(years))
    
    def shuffle_question_options(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shuffle the options of a question and update the correct answer accordingly.
        
        Args:
            question: Original question dictionary
            
        Returns:
            Question dictionary with shuffled options and updated answer
        """
        # Create a deep copy to avoid modifying the original
        shuffled_question = question.copy()
        options = question.get('options', [])
        original_answer = question.get('answer')
        
        if not options or not original_answer:
            logger.warning(f"Question {question.get('id')} has no options or answer, skipping shuffle")
            return shuffled_question
        
        # Create list of option letters and corresponding options
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:len(options)]
        
        # Create mapping from original position to option data
        original_options = []
        for i, option in enumerate(options):
            if isinstance(option, dict) and 'letter' in option:
                original_options.append(option)
            else:
                # Handle case where option might be just text
                original_options.append({
                    'letter': option_letters[i],
                    'text': option if isinstance(option, str) else str(option)
                })
        
        # Find the correct answer option
        correct_option_index = None
        for i, option in enumerate(original_options):
            if option['letter'] == original_answer:
                correct_option_index = i
                break
        
        if correct_option_index is None:
            logger.warning(f"Cannot find correct answer '{original_answer}' in options for question {question.get('id')}")
            return shuffled_question
        
        # Create shuffled order
        shuffled_indices = list(range(len(original_options)))
        # Use question ID hash as seed for consistent shuffling per question
        question_id = question.get('id')
        base_seed = hash(question_id) % (2**32)  # Ensure seed is within int32 range
        
        # Keep shuffling until no option maps to itself
        seed_offset = 0
        max_attempts = 100  # Prevent infinite loop
        
        while seed_offset < max_attempts:
            current_seed = (base_seed + seed_offset) % (2**32)
            local_random = random.Random(current_seed)
            shuffled_indices = list(range(len(original_options)))
            local_random.shuffle(shuffled_indices)
            
            # Check if any index maps to itself (derangement check)
            has_fixed_point = any(i == shuffled_indices[i] for i in range(len(shuffled_indices)))
            
            if not has_fixed_point:
                # Found a valid shuffling with no fixed points
                break
            
            seed_offset += 1
        
        if seed_offset >= max_attempts:
            logger.warning(f"Could not find derangement for question {question_id} after {max_attempts} attempts, using last attempt")
        else:
            logger.debug(f"Question {question_id}: Found valid derangement after {seed_offset + 1} attempts")
        
        # Create new options with new letters
        new_options = []
        new_answer = None
        option_mapping = {}  # Track how options were shuffled
        
        for new_index, original_index in enumerate(shuffled_indices):
            new_letter = option_letters[new_index]
            original_option = original_options[original_index]
            
            new_option = {
                'letter': new_letter,
                'text': original_option['text']
            }
            new_options.append(new_option)
            
            # Track the mapping
            option_mapping[original_option['letter']] = new_letter
            
            # Update the correct answer
            if original_index == correct_option_index:
                new_answer = new_letter
        
        # Update the question
        shuffled_question['options'] = new_options
        shuffled_question['answer'] = new_answer
        
        # Add metadata about the shuffling
        shuffled_question['_option_shuffle_info'] = {
            'original_answer': original_answer,
            'new_answer': new_answer,
            'option_mapping': option_mapping,  # Maps original letter -> new letter
            'shuffled': True
        }
        
        logger.debug(f"Question {question.get('id')}: Shuffled options. Original answer: {original_answer} -> New answer: {new_answer}")
        
        return shuffled_question
    
    def _validate_question(self, question: Dict[str, Any]) -> bool:
        """
        Validate question data structure.
        
        Args:
            question: Question dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['id', 'question', 'options', 'answer']
        
        # Check required fields
        for field in required_fields:
            if field not in question:
                logger.warning(f"Question missing required field '{field}': {question.get('id', 'unknown')}")
                return False
        
        # Check options format
        options = question['options']
        if not isinstance(options, list) or len(options) < 2:
            logger.warning(f"Invalid options format for question {question['id']}")
            return False
        
        # Check answer is valid
        answer = question['answer']
        if not isinstance(answer, str) or answer not in 'ABCDEFGH'[:len(options)]:
            logger.warning(f"Invalid answer '{answer}' for question {question['id']}")
            return False
        
        return True
    
    def get_categories(self) -> List[str]:
        """
        Get list of available categories.
        
        Returns:
            List of category names
        """
        if not self.data:
            # Load all data but don't shuffle
            self.load_questions(shuffle=False)
        
        categories = set()
        for q in self.data.get('questions', []):
            category = q.get('category')
            if category:  # Only add non-empty categories
                categories.add(category)
        
        return sorted(list(categories))
    
    def get_statistics(self, year: Optional[int] = None, use_astro: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive data statistics including validation info.

        Returns:
            Dictionary with detailed statistics
        """
        valid_questions = self.load_questions(
            use_astro=use_astro,
            year=year,
            shuffle=False
        )

        questions = self.data.get('questions', [])
        available_years = {}
        for q in questions:
            benchmark_year = self.infer_benchmark_year(q)
            if benchmark_year is not None:
                available_years[benchmark_year] = available_years.get(benchmark_year, 0) + 1

        if year is not None:
            questions = [
                q for q in questions
                if self.infer_benchmark_year(q) == year
            ]

        # Calculate various statistics
        category_counts = {}
        missing_answers = 0
        invalid_options = 0
        
        for q in questions:
            # Category statistics
            cat = q.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Check data integrity
            if not q.get('answer'):
                missing_answers += 1
            
            options = q.get('options', [])
            if not isinstance(options, list) or len(options) < 2:
                invalid_options += 1
        
        return {
            'total_questions': len(questions),
            'valid_questions': len(valid_questions),
            'categories': category_counts,
            'missing_answers': missing_answers,
            'invalid_options': invalid_options,
            'selected_year': year,
            'available_years': sorted(available_years),
            'year_counts': available_years,
            'data_version': self.data.get('version', 'unknown'),
            'benchmark_name': self.data.get('benchmark_name', 'FortuneTellingBench')
        }
    
    def _load_fortune_data(self):
        """Load fortune data from API results file."""
        if self.fortune_data_path.exists():
            logger.info(f"Loading fortune data from {self.fortune_data_path}")
            with open(self.fortune_data_path, 'r', encoding='utf-8') as f:
                fortune_results = json.load(f)
            
            # Convert list to dictionary by case_id
            self.fortune_data = {}
            for result in fortune_results:
                case_id = result.get('case_id')
                if case_id and result.get('status') == 'success':
                    self.fortune_data[case_id] = result
            
            # Add fortune data to main data structure
            if not self.data:
                self.data = {}
            self.data['fortune_data'] = self.fortune_data
            
            logger.info(f"Loaded fortune data for {len(self.fortune_data)} cases")
        else:
            logger.warning(f"Fortune data file not found: {self.fortune_data_path}")
