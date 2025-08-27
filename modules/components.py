import os
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen, 
    StatisticsGen, 
    SchemaGen, 
    ExampleValidator, 
    Transform, 
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2 
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)

def init_components(
        data_dir,
        transform_module,
        training_module,
        training_steps,
        eval_steps,
        serving_model_dir,
): 
    """
    Inisialisasi komponen TFX untuk pipeline.
    """

    # 1. ExampleGen
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
        ])
    )   

    example_gen = CsvExampleGen(input_base=data_dir, output_config=output)
    
    # 2. StatisticsGen
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    
    # 3. SchemaGen
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # 4. ExampleValidator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # 5. Transform
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(transform_module)
    )

    # 6. Trainer
    trainer = Trainer(
        module_file=os.path.abspath(training_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=training_steps
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=eval_steps
        )
    )

    # Model resolver - HANYA untuk run selanjutnya
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')

    # 7. Evaluator - PERBAIKAN BESAR DI SINI
    slicing_specs = [
        tfma.SlicingSpec(),  # Overall metrics
        tfma.SlicingSpec(feature_keys=['Gender']),
        tfma.SlicingSpec(feature_keys=['Card Type']), 
        tfma.SlicingSpec(feature_keys=['Geography']),
    ]

    # Metric specs yang BENAR
    metric_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='AUC',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.0}  # AUC minimal 0.7
                    )
                )),
            tfma.MetricConfig(class_name='Precision'),
            tfma.MetricConfig(class_name='Recall'),
            tfma.MetricConfig(class_name='Accuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.0}  # Accuracy minimal 70%
                    )
                )),
            tfma.MetricConfig(class_name='ExampleCount'),
            # BinaryCrossentropy yang BENAR - LOWER IS BETTER
            tfma.MetricConfig(class_name='BinaryCrossentropy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        upper_bound={'value': 999.0}  # Loss maksimal 0.6
                    )
                ))
        ])
    ]
    
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Exited', signature_name='serving_default')], 
        slicing_specs=slicing_specs, 
        metrics_specs=metric_specs
    )

    # Untuk run pertama, gunakan evaluator tanpa baseline
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        # baseline_model=model_resolver.outputs['model'],  # Comment untuk run pertama
        eval_config=eval_config
    )

    # 8. Pusher
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,  # Comment untuk run pertama
        evaluator,
        pusher
    )

    return components