import tensorflow as tf
import tensorflow_transform as tft

CAT_FEATURES = {
    "Geography": 3,
    "Gender" : 2,
    "Card Type": 4,
}

NUM_FEATURES = [
    'CreditScore', 
    'Age', 
    'Tenure', 
    'Balance', 
    'NumOfProducts', 
    'HasCrCard',
    'IsActiveMember', 
    'EstimatedSalary', 
    'Complain',
    'Satisfaction Score', 
    'Point Earned'
]

LABEL_KEY = 'Exited'

def transformed_name(key):
    "Renaming the transformed features"
    return key + '_xf'

def convert_num_to_one_hot(label_tensor, num_labels=2):
    """ 
    Convert a label (1 or 0) tensor to one-hot vector
    
    args:
     int: label_tensor (0 or 1)
     
    return 
    label tensor.
    """

    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

def preprocessing_fn(inputs):
    outputs = {}

    # Transform categorical features
    for key, dim in CAT_FEATURES.items():
        int_val = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1)

        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_val, num_labels=dim + 1
        )
    
    # Transform numerical features (ensure float32)
    for feature in NUM_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(
            tf.cast(inputs[feature], tf.float32)
        )

    # Label
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        inputs[LABEL_KEY], tf.int64
    )
    
    return outputs
