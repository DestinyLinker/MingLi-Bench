"""
Core benchmark framework for evaluating LLMs on fortune telling tasks.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .models.base import ModelClient
from .utils.logger import get_logger
from .data.loader import DataLoader
from .data.schema import ANSWER_PATTERNS, VALID_OPTIONS, EvaluationResult
from dataclasses import asdict

logger = get_logger(__name__)


class FortuneTellingBenchmark:
    """Main benchmark class for evaluating LLMs on fortune telling tasks."""
    
    def __init__(self, model_client: ModelClient, data_path: Optional[str] = None):
        """
        Initialize the benchmark.
        
        Args:
            model_client: ModelClient instance for LLM API calls
            data_path: Path to the benchmark data file
        """
        self.model_client = model_client
        self.data_loader = DataLoader(data_path)
        self.results = []
        
    def evaluate(self, 
                 use_cot: bool = False,
                 use_astro: bool = False,
                 sample_size: Optional[int] = None,
                 year: Optional[int] = None,
                 categories: Optional[List[str]] = None,
                 shuffle_options: bool = True,
                 max_workers: int = 5) -> Dict[str, Any]:
        """
        Run the benchmark evaluation.
        
        Args:
            use_cot: Whether to use Chain-of-Thought reasoning
            use_astro: Whether to use astronomical data
            sample_size: Number of questions to evaluate (None for all)
            year: Specific benchmark year to evaluate
            categories: Specific categories to evaluate
            shuffle_options: Whether to shuffle options within each question
            max_workers: Maximum number of concurrent API calls
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation with model: {self.model_client.model_name}")
        logger.info(
            f"Settings: CoT={use_cot}, Astro={use_astro}, Year={year}, "
            f"Sample={sample_size}, ShuffleOptions={shuffle_options}"
        )
        
        # Load questions
        questions = self.data_loader.load_questions(
            use_astro=use_astro,
            sample_size=sample_size,
            year=year,
            categories=categories,
            shuffle_options=shuffle_options
        )
        
        logger.info(f"Loaded {len(questions)} questions")
        
        # Evaluate questions
        start_time = time.time()
        self.results = self._evaluate_questions(questions, use_cot, use_astro, max_workers)
        end_time = time.time()
        
        # Calculate statistics
        stats = self._calculate_statistics(self.results)
        stats['evaluation_time'] = end_time - start_time
        stats['model_name'] = self.model_client.model_name
        stats['use_cot'] = use_cot
        stats['use_astro'] = use_astro
        stats['selected_year'] = year
        stats['evaluated_years'] = sorted({
            q.get('benchmark_year')
            for q in questions
            if q.get('benchmark_year') is not None
        })
        stats['shuffle_options'] = shuffle_options
        stats['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Evaluation completed in {stats['evaluation_time']:.2f} seconds")
        logger.info(f"Overall accuracy: {stats['overall_accuracy']:.2%}")
        
        return stats
    
    def _evaluate_questions(self, 
                           questions: List[Dict[str, Any]], 
                           use_cot: bool,
                           use_astro: bool,
                           max_workers: int) -> List[Dict[str, Any]]:
        """
        Evaluate a list of questions using concurrent API calls.
        
        Args:
            questions: List of questions to evaluate
            use_cot: Whether to use Chain-of-Thought reasoning
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of evaluation results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(self._evaluate_single_question, q, use_cot, use_astro): q
                for q in questions
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(questions), desc="Evaluating") as pbar:
                for future in as_completed(future_to_question):
                    question = future_to_question[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error evaluating question {question['id']}: {e}")
                        results.append({
                            'question_id': question['id'],
                            'correct': False,
                            'error': str(e)
                        })
                    pbar.update(1)
        
        return results
    
    def _evaluate_single_question(self, 
                                 question: Dict[str, Any], 
                                 use_cot: bool,
                                 use_astro: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single question with improved error handling.
        
        Args:
            question: Question data
            use_cot: Whether to use Chain-of-Thought reasoning
            use_astro: Whether to use astronomical/fortune data
            
        Returns:
            Evaluation result dictionary
        """
        start_time = time.time()
        
        # Initialize result with defaults to ensure consistent structure
        result_data = {
            'question_id': question.get('id', 'unknown'),
            'category': question.get('category', 'unknown'),
            'benchmark_year': question.get('benchmark_year'),
            'question': question.get('question', ''),
            'options': question.get('options', []),
            'correct_answer': question.get('answer', None),
            'predicted_answer': None,
            'correct': False,
            'response': '',
            'response_time': 0,
            'error': None,
            'prompt': None
        }
        
        # Add option shuffle information if available
        if '_option_shuffle_info' in question:
            result_data['option_shuffle_info'] = question['_option_shuffle_info']
        
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(question, use_cot, use_astro)
            result_data['prompt'] = prompt

            # Get model prediction
            response = self.model_client.generate(prompt)
            result_data['response'] = response

            # Extract answer
            prediction = self._extract_answer(response, use_cot)
            result_data['predicted_answer'] = prediction
            
            # Check if correct
            if prediction and result_data['correct_answer']:
                result_data['correct'] = prediction == result_data['correct_answer']
            
        except Exception as e:
            logger.error(f"Error evaluating question {result_data['question_id']}: {e}")
            result_data['error'] = str(e)
        
        result_data['response_time'] = time.time() - start_time
        
        # Create EvaluationResult for consistency with rest of codebase
        result = EvaluationResult(**result_data)
        return asdict(result)
    
    def _prepare_prompt(self, question: Dict[str, Any], use_cot: bool, use_astro: bool = False) -> str:
        """
        Prepare the prompt for the model. 
        
        Args:
            question: Question data
            use_cot: Whether to use Chain-of-Thought reasoning
            use_astro: Whether to use astronomical/fortune data
            
        Returns:
            Formatted prompt string
        """
        birth_info = question['birth_info']
        
        # Check if fortune data is available
        fortune_data = None
        if use_astro:
            logger.debug(f"use_astro is True, checking for fortune data for case: {question.get('case_id')}")
            if hasattr(self.data_loader, 'data') and self.data_loader.data:
                fortune_data = self.data_loader.data.get('fortune_data', {}).get(question.get('case_id'))
                logger.debug(f"Fortune data found: {fortune_data is not None}")
            else:
                logger.debug("data_loader.data not available")
        
        prompt = f"""以下是一道关于中国传统命理的题目。

命主信息：
{birth_info.get('raw', birth_info)}"""
        if use_cot:
            prompt += "\n结合中国传统命理学（包括但不限于四柱八字、紫微斗数等），请先分析推理过程，然后给出答案。最后用'答案：X'的格式给出你的选择（X为A、B、C或D）。"
        else:
            prompt += "\n结合中国传统命理学（包括但不限于四柱八字、紫微斗数等）进行推算，请直接给出答案，用'答案：X'的格式（X为A、B、C或D）。"
         
        # Add fortune data if available
        if fortune_data and fortune_data.get('api_response'):
            api_response = fortune_data['api_response']
            if api_response.get('success') and api_response.get('data'):
                chart = api_response['data'].get('data') or {}

                if isinstance(chart, dict) and chart:
                    chinese_date = chart.get('chineseDate') or '未知'
                    time_info = chart.get('time') or '未知'
                    five_elements = chart.get('fiveElementsClass') or '未知'
                    zodiac = chart.get('zodiac') or '未知'

                    prompt += f"""

八字命盘信息：
八字：{chinese_date}
时辰：{time_info}
五行局：{five_elements}
生肖：{zodiac}

紫微命盘信息：
十二宫位星曜分布："""

                    palace_order = ['命宫', '兄弟', '夫妻', '子女', '财帛', '疾厄',
                                   '迁移', '仆役', '官禄', '田宅', '福德', '父母']

                    palace_info = {}
                    for palace in chart.get('palaces') or []:
                        name = palace.get('name')
                        if not name:
                            continue
                        major = [s.get('name', '') for s in palace.get('majorStars') or [] if s.get('name')]
                        minor = [s.get('name', '') for s in palace.get('minorStars') or [] if s.get('name')]
                        all_stars = major + minor
                        if all_stars:
                            palace_info[name] = ' '.join(all_stars)

                    for palace in palace_order:
                        if palace in palace_info:
                            prompt += f"\n{palace}：{palace_info[palace]}"

        prompt += f"""

问题：{question['question']}

选项：
"""
        
        # Validate and build options
        options = question.get('options', [])
        if not options:
            raise ValueError(f"Question {question.get('id')} has no options")
        
        # Handle different option formats and sort by letter
        formatted_options = []
        for option in options:
            if isinstance(option, dict) and 'letter' in option:
                # Option has explicit letter
                formatted_options.append(option)
            else:
                # Need to assign letters
                for i, opt in enumerate(options[:4]):  # Max 4 options
                    letter = chr(65 + i)  # A, B, C, D
                    if isinstance(opt, dict):
                        text = opt.get('text', str(opt))
                    else:
                        text = str(opt)
                    formatted_options.append({'letter': letter, 'text': text})
                break
        
        # Sort by letter to ensure consistent order
        sorted_options = sorted(formatted_options, key=lambda x: x.get('letter', 'Z'))
        
        for opt in sorted_options:
            letter = opt.get('letter', '?')
            text = opt.get('text', '')
            prompt += f"{letter}. {text}\n"
        

        return prompt
    
    def _extract_answer(self, response: str, use_cot: bool = False) -> Optional[str]:
        """
        Extract the answer from model response with improved logic.
        
        Args:
            response: Model response text
            use_cot: Whether CoT was used
            
        Returns:
            Extracted answer (A, B, C, or D) or None
        """
        import re

        # Clean response text
        response = response.strip()
        response = re.sub(r'[\*_`]+', '', response)
        response = re.sub(r'[\"\'"""‘’「」『』（）\(\)\[\]【】<>《》]', '', response)

        # Answer extraction patterns (in priority order)
        patterns = [
            # Explicit answer formats
            r'答案[：:]\s*([A-Za-z])',
            r'答案是[：:]\s*([A-Za-z])',
            r'选择[：:]\s*([A-Za-z])',
            r'选[：:]\s*([A-Za-z])',
            # Single letter on its own line
            r'^([A-Za-z])$',
            # Answer at end of sentence
            r'[。，]([A-Za-z])[。]?$',
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, response, re.MULTILINE))
            if matches:
                # If multiple matches, take the last one (final answer)
                return matches[-1].group(1).upper()
        
        # Last resort: find all standalone letters and take the last valid one
        all_letters = re.findall(r'\b([A-Za-z])\b', response)
        if all_letters:
            # Filter for valid option letters
            valid_letters = [l.upper() for l in all_letters if l.upper() in ['A', 'B', 'C', 'D', 'E', 'F']]
            if valid_letters:
                return valid_letters[-1]
        
        return None
    
    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate evaluation statistics.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary containing statistics
        """
        # Handle empty results
        if not results:
            logger.warning("No results to calculate statistics")
            return {
                'total_questions': 0,
                'correct_answers': 0,
                'errors': 0,
                'overall_accuracy': 0.0,
                'category_stats': {},
                'average_response_time': 0.0,
                'results': []
            }
        
        total = len(results)
        correct = sum(1 for r in results if r.get('correct', False))
        errors = sum(1 for r in results if r.get('error') not in (None, ''))
        
        # Calculate category-wise accuracy
        category_stats = {}
        for result in results:
            category = result.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'correct': 0}
            category_stats[category]['total'] += 1
            if result.get('correct', False):
                category_stats[category]['correct'] += 1
        
        # Calculate accuracies
        for category in category_stats:
            stats = category_stats[category]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        # Calculate average response time
        response_times = [r.get('response_time', 0) for r in results if 'response_time' in r]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'total_questions': total,
            'correct_answers': correct,
            'errors': errors,
            'overall_accuracy': correct / total if total > 0 else 0,
            'category_stats': category_stats,
            'average_response_time': avg_response_time,
            'results': results
        }
    
    def save_results(self, stats: Dict[str, Any], output_dir: str = "logs"):
        """
        Save evaluation results to files.

        Args:
            stats: Evaluation statistics
            output_dir: Directory to save results (default: logs)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_client.model_name.replace("/", "-")
        run_dir = output_path / f"{model_name}_{timestamp}"
        run_dir.mkdir(exist_ok=True)

        results_file = run_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved results to {results_file}")

        summary_file = run_dir / "summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_summary(stats))
        logger.info(f"Saved summary to {summary_file}")

        responses_dir = run_dir / "responses"
        responses_dir.mkdir(exist_ok=True)
        
        if 'results' in stats:
            for i, result in enumerate(stats['results']):
                question_id = result.get('question_id', f'unknown_{i}')
                # Create individual file for each question
                response_file = responses_dir / f"question_{question_id}.txt"
                
                with open(response_file, 'w', encoding='utf-8') as f:
                    f.write(f"题目ID: {question_id}\n")
                    f.write(f"类别: {result.get('category', 'unknown')}\n")
                    if result.get('benchmark_year') is not None:
                        f.write(f"评测年份: {result['benchmark_year']}\n")
                    f.write("="*60 + "\n\n")
                    
                    f.write("【题目】\n")
                    f.write(f"{result.get('question', '')}\n\n")
                    
                    f.write("【选项】\n")
                    for option in result.get('options', []):
                        if isinstance(option, dict):
                            f.write(f"{option.get('letter', '?')}. {option.get('text', '')}\n")
                        else:
                            f.write(f"{option}\n")
                    
                    if result.get('prompt'):
                        f.write("\n" + "="*60 + "\n\n")
                        f.write("【完整Prompt】\n")
                        f.write(f"{result['prompt']}\n\n")

                    f.write("\n" + "="*60 + "\n\n")
                    f.write("【模型回复】\n")
                    f.write(f"{result.get('response', '')}\n\n")
                    
                    f.write("="*60 + "\n")
                    f.write("【评估结果】\n")
                    f.write(f"预测答案: {result.get('predicted_answer', 'None')}\n")
                    f.write(f"正确答案: {result.get('correct_answer', 'None')}\n")
                    f.write(f"是否正确: {'✓ 正确' if result.get('correct', False) else '✗ 错误'}\n")
                    f.write(f"响应时间: {result.get('response_time', 0):.2f}秒\n")
                    
                    # Add option shuffle information if available
                    if 'option_shuffle_info' in result and result['option_shuffle_info'] is not None:
                        shuffle_info = result['option_shuffle_info']
                        f.write(f"\n【选项打乱信息】\n")
                        f.write(f"选项已打乱: {'是' if shuffle_info.get('shuffled', False) else '否'}\n")
                        if shuffle_info.get('shuffled', False):
                            f.write(f"原始答案: {shuffle_info.get('original_answer', 'None')}\n")
                            f.write(f"打乱后答案: {shuffle_info.get('new_answer', 'None')}\n")
                            f.write("选项映射: ")
                            mapping = shuffle_info.get('option_mapping', {})
                            mapping_str = ", ".join([f"{orig}→{new}" for orig, new in mapping.items()])
                            f.write(f"{mapping_str}\n")
                    
                    if result.get('error'):
                        f.write(f"\n错误信息: {result['error']}\n")
            
            logger.info(f"Saved {len(stats['results'])} individual responses to {responses_dir}")
    
    def _generate_summary(self, stats: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of results.
        
        Args:
            stats: Evaluation statistics
            
        Returns:
            Summary text
        """
        summary = f"""Chinese Fortune Telling Benchmark Evaluation Summary
==========================================

Model: {stats['model_name']}
Date: {stats['timestamp']}
Settings: CoT={stats['use_cot']}, Astro={stats['use_astro']}, Year={stats.get('selected_year', 'all')}, ShuffleOptions={stats.get('shuffle_options', False)}

Overall Results:
--------------
Total Questions: {stats['total_questions']}
Correct Answers: {stats['correct_answers']}
Errors: {stats['errors']}
Overall Accuracy: {stats['overall_accuracy']:.2%}

Category Breakdown:
-----------------
"""
        
        for category, cat_stats in stats['category_stats'].items():
            summary += f"{category:12s}: {cat_stats['accuracy']:6.2%} ({cat_stats['correct']}/{cat_stats['total']})\n"
        
        summary += f"\nAverage Response Time: {stats['average_response_time']:.2f}s\n"
        summary += f"Total Evaluation Time: {stats['evaluation_time']:.2f}s\n"
        
        return summary
