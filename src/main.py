import asyncio
import logging
import sys
from pathlib import Path

from infra.pipeline_executor import PipelineExecutor
from src.pipeline import StocksRecommenderPipeline


def setup_logger():
    """Initialize and configure the logger for the pipeline."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_dir / 'pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logger initialized successfully")
    return logger


def main():
    """Main entry point for executing the pipeline step by step."""
    # Setup logger
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Starting Stocks Recommender Pipeline")
    logger.info("=" * 60)
    
    try:
        # Define pipeline steps
        pipeline_steps = [
            StocksRecommenderPipeline.fetch_tickers,
            StocksRecommenderPipeline.run_agents,
            StocksRecommenderPipeline.process_output,
            StocksRecommenderPipeline.publish
        ]
        
        # Initialize PipelineExecutor
        executor = PipelineExecutor(
            pipeline_class=StocksRecommenderPipeline,
            pipeline_steps=pipeline_steps,
            load_pipeline=False,  # Set to True to load from saved state
            save_pipeline=False,  # Set to True to save pipeline state
            pipeline_path=None,  # Path to save/load pipeline pickle
            start_from_saved_state=False,
            save_state_on_every_step=False
        )
        
        # Create step functions that the executor can call
        # The executor expects functions with __name__ attribute matching the method name
        def create_step_function(step_name):
            """Create a function wrapper for a pipeline step."""
            # Capture step_name in closure
            name = step_name
            if name == 'run_agents':
                # Handle async method
                def step_func():
                    return asyncio.run(getattr(executor.pipeline, name)())
            else:
                # Handle sync methods
                def step_func():
                    return getattr(executor.pipeline, name)()
            
            step_func.__name__ = name
            return step_func
        
        # Create step functions for each pipeline step
        steps_to_execute = [create_step_function(step_name) for step_name in pipeline_steps]
        
        # Execute pipeline steps
        logger.info(f"Executing {len(pipeline_steps)} pipeline steps: {', '.join(pipeline_steps)}")
        executor.execute(steps_to_execute)
        
        logger.info("=" * 60)
        logger.info("Pipeline execution completed successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
