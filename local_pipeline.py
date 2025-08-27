import os
import sys
from typing import Text

from absl import logging
from tfx.orchestration import pipeline, metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = 'pp'

# pipeline input parameters
DATA_ROOT = os.path.abspath('data')
TRANSFORM_MODULE_FILE = os.path.abspath("modules/customer_churn_transform.py")
TRAINER_MODULE_FILE = os.path.abspath("modules/customer_churn_trainer.py")

# pipeline output parameters
OUTPUT_DIR = os.path.abspath('op')
SERVING_MODEL_DIR = os.path.join(OUTPUT_DIR, 'serving_model')
pipeline_root = os.path.join(OUTPUT_DIR, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')


def init_local_pipeline(
        components, pipeline_root: Text,
) -> pipeline.Pipeline:
    """
    Membuat dan mengembalikan pipeline TFX lokal.
    
    arguments:
        components: Daftar komponen TFX.
        pipeline_root: Direktori root untuk pipeline.
    
    returns:
        Objek pipeline TFX.
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=1",
        "--number_of_worker_harness_threads=1",
        "--disk_cache_size=0",
    ]


    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ), 
        beam_pipeline_args=beam_args 
    )

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    
    from modules.components import init_components
    
    components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        training_steps=700,
        eval_steps=300,
        serving_model_dir=SERVING_MODEL_DIR,
    )
    
    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)