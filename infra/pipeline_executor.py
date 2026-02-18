import logging


from infra.base_pipeline import BasePipeline
from infra.timer import Timer
from infra.utils import save_to_pickle, is_file_exist, load_pickle


class PipelineExecutor:
    """
    this class gets a list of steps, and the class itself of the pipeline,
    and exec every step.
    it also save every step into a pickle, the pipeline
    and can load the pickle and continue from the wanted step
    """

    def __init__(self,
             pipeline_class,
             pipeline_steps: list[str],
             load_pipeline: bool = False,
             save_pipeline: bool = False,
             pipeline_path: str = None,
             start_from_saved_state: bool = False,
             save_state_on_every_step: bool = False):

        self.pipeline_class = pipeline_class
        self.pipeline_steps = pipeline_steps
        self.load_pipeline = load_pipeline
        self.save_pipeline = save_pipeline
        self.pipeline_path = pipeline_path
        self.start_from_saved_state = start_from_saved_state
        self.save_state_on_every_step = save_state_on_every_step
        
        # Initialize pipeline
        self.pipeline = self._get_pipeline(pipeline_class)
        
        # Validate input arguments
        self._validate_input_args()


    def execute(self):
        pipeline_timer = Timer('Starting pipeline').start()
        for step in self.pipeline_steps:
            step_name = step.__name__
            logging.info(f'================ Executing: {step_name} ===========')

            timer = Timer(f'Executing {step_name}').start()
            getattr(self.pipeline, step_name)()
            timer.end()
            self._save_pipeline_if_needed()

            logging.info(f'================ Finished: {step_name} ===========')
        pipeline_timer.end()

    def _get_pipeline(self, pipeline_class):
        if self.start_from_saved_state and is_file_exist(self.pipeline_path):
            pipeline = self._load_pipeline()
            if isinstance(pipeline, BasePipeline):
                pipeline.setup()
            return pipeline
        else:
            return pipeline_class()

    def _save_pipeline_if_needed(self):
        if self.save_state_on_every_step:
            timer = Timer('Saving pipeline').start()
            if isinstance(self.pipeline, BasePipeline):
                self.pipeline.tear_down()
            save_to_pickle(self.pipeline, self.pipeline_path)
            timer.end()

    def _load_pipeline(self):
        logging.info('Starting pipeline')
        timer = Timer('Loading pipeline').start()
        pipeline = load_pickle(self.pipeline_path)
        timer.end()
        return pipeline

    def _validate_input_args(self):
        if self.save_state_on_every_step and self.pipeline_path is None:
            raise ValueError('Please specify pipeline pickle path')


