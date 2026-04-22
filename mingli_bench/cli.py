"""
Command-line interface for Fortune Telling Benchmark.
"""

import argparse
import sys

from .benchmark import FortuneTellingBenchmark
from .models.factory import ModelFactory
from .utils import get_logger, load_config
from .data import DataLoader

logger = get_logger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chinese Fortune Telling Benchmark - Evaluate LLMs on Chinese fortune telling tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate GPT-4
  python -m mingli_bench.cli --model gpt-4

  # Evaluate only 2023 benchmark questions
  python -m mingli_bench.cli --model gpt-4 --year 2023
  
  # Use Chain-of-Thought reasoning
  python -m mingli_bench.cli --model gpt-4 --cot
  
  # Test with 10 sample questions
  python -m mingli_bench.cli --model claude-3-sonnet --sample 10
  
  # Filter by category
  python -m mingli_bench.cli --model gemini-pro --categories 事件 婚姻
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        help="Model to evaluate (e.g., gpt-4, claude-3-sonnet, gemini-pro)"
    )
    
    # Evaluation options
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use Chain-of-Thought reasoning"
    )
    
    parser.add_argument(
        "--astro",
        action="store_true",
        help="Use astronomical/astrological data"
    )
    
    parser.add_argument(
        "--shuffle-options",
        action="store_true",
        help="Randomly shuffle options within each question"
    )
    
    parser.add_argument(
        "--sample", "-s",
        type=int,
        metavar="N",
        help="Evaluate only N sample questions"
    )

    parser.add_argument(
        "--year", "-y",
        type=int,
        help="Evaluate only questions from a specific benchmark year (e.g., 2022)"
    )
    
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        choices=["事业", "健康", "外貌", "婚姻", "子女", "学业", "官非", "家庭", "性格", "灾劫", "财运", "运势"],
        help="Filter questions by categories"
    )
    
    # Data options
    parser.add_argument(
        "--data-path",
        help="Path to benchmark data file"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        default="logs",
        help="Directory to save results (default: logs)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    
    # Performance options
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum concurrent API calls (default: 5)"
    )
    
    # Configuration
    parser.add_argument(
        "--env-file",
        help="Path to .env file"
    )

    parser.add_argument(
        "--platform",
        choices=["openai", "openrouter", "anthropic", "google", "deepseek", "doubao"],
        help="Force routing platform (overrides auto-detection from model name prefix)",
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported models and exit"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics and exit"
    )
    
    args = parser.parse_args()
    
    # Handle special actions
    if args.list_models:
        print("Supported models by provider:")
        for provider, models in ModelFactory.list_supported_models().items():
            print(f"\n{provider.capitalize()}:")
            for model in models:
                print(f"  - {model}")
        return 0
    
    if args.stats:
        try:
            loader = DataLoader(args.data_path)
            stats = loader.get_statistics(year=args.year)
        except Exception as e:
            logger.error(f"Failed to load dataset statistics: {e}")
            return 1
        print(f"\nDataset Statistics:")
        print(f"  Name: {stats['benchmark_name']}")
        print(f"  Version: {stats['data_version']}")
        print(f"  Available Years: {', '.join(map(str, stats['available_years']))}")
        if args.year is not None:
            print(f"  Selected Year: {args.year}")
        print(f"  Total Questions: {stats['total_questions']}")
        print(f"\n  Categories:")
        for cat, count in stats['categories'].items():
            print(f"    - {cat}: {count}")
        return 0
    
    # Check if model is required for remaining operations
    if not args.model:
        parser.error("--model is required unless using --list-models or --stats")
    
    # Load configuration
    try:
        config = load_config(args.env_file)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create model client
    try:
        logger.info(f"Creating model client for {args.model}")
        model_client = ModelFactory.create(args.model, provider=args.platform, config=config)
    except Exception as e:
        logger.error(f"Failed to create model client: {e}")
        return 1
    
    # Validate API key
    if not model_client.validate_api_key():
        logger.error("Invalid API key. Please check your configuration.")
        return 1
    
    # Create benchmark
    benchmark = FortuneTellingBenchmark(model_client, args.data_path)
    
    # Run evaluation
    try:
        logger.info("Starting benchmark evaluation...")
        results = benchmark.evaluate(
            use_cot=args.cot,
            use_astro=args.astro,
            sample_size=args.sample,
            year=args.year,
            categories=args.categories,
            shuffle_options=getattr(args, 'shuffle_options', False),
            max_workers=args.max_workers
        )
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Evaluation Results - {args.model}")
        print(f"{'='*50}")
        if args.year is not None:
            print(f"Benchmark Year: {args.year}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        
        if results['errors'] > 0:
            print(f"Errors: {results['errors']}")
        
        print(f"\nCategory Breakdown:")
        for cat, stats in results['category_stats'].items():
            print(f"  {cat:12s}: {stats['accuracy']:6.2%} ({stats['correct']}/{stats['total']})")
        
        print(f"\nEvaluation Time: {results['evaluation_time']:.2f}s")
        print(f"Avg Response Time: {results['average_response_time']:.2f}s")
        
        # Save results
        if not args.no_save:
            benchmark.save_results(results, args.output_dir)
            print(f"\nResults saved to {args.output_dir}/")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
