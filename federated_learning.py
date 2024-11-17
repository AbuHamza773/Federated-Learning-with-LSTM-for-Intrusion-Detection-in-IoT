
import numpy as np

def federated_aggregation(local_weights):
    """Aggregate weights from multiple local models in federated learning."""
    aggregated_weights = []
    for weights_list_tuple in zip(*local_weights):
        aggregated_weights.append(
            np.mean([np.array(weights) for weights in weights_list_tuple], axis=0)
        )
    return aggregated_weights

def update_global_model(global_model, aggregated_weights):
    """Update the global model with aggregated weights."""
    global_model.set_weights(aggregated_weights)
