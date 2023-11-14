"""
> The gain in loss if we assume change in the decay rates is greater than sensitivity threshold
"""

def sensitivity(estimated_loss, current_loss, threshold):
    """
    If predicted loss is greater than

    :return:
    """
    if abs((estimated_loss - current_loss)/current_loss) > threshold:
        return True
    return False

"""
Some thoughts on this
Since the loss function is monotonic on the decay function, then if we already have a threshold for the decay params (trigger event), 
    then this by itself is already a measure for sensitivity.
IOTW, having a sensitivity threshold for the loss can be redundant.
Or perhaps, we can compute the threshold needed for the decay params by inverting the sensitivity threshold based on the loss function
(Otherwise, this will be an added complexity.)
"""